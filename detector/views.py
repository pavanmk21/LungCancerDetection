import os
import json
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image

from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator

import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .models import Prediction, TrainingSession
from .forms import ImageUploadForm, TrainingForm
from django.http import HttpRequest, HttpResponse


# -------------------------------
# PyTorch Model Loader / Detector
# -------------------------------
class Detector:
    # UPDATED: Default model changed to 'efficientnet_b3' (Matches stem size 40)
    def __init__(self, model_path=None, model_name='efficientnet_b3', classes=None, device=None):
        self.model_path = model_path
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # FIX 1: Changed to LOWERCASE to match string comparisons in result.html
        self.classes = classes or ['benign', 'malignant', 'normal'] 
        self.model = None

        if model_path:
            self.load_model(model_path)

        # UPDATED: EfficientNet-B3 requires 300x300 resolution for optimal performance
        self.val_transform = A.Compose([
            A.Resize(300, 300),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Loading model from: {model_path}")
        print(f"Using Architecture: {self.model_name}")
        
        checkpoint = torch.load(model_path, map_location=self.device)

        # Initialize the correct architecture (EfficientNet-B3)
        self.model = timm.create_model(self.model_name, pretrained=False, num_classes=len(self.classes))

        # Flexible state_dict loading to handle different saving methods
        if isinstance(checkpoint, dict):
            if "model_state" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state"])
            elif "state_dict" in checkpoint:
                self.model.load_state_dict(self.strip_model_prefix(checkpoint["state_dict"]))
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
        
    def strip_model_prefix(self, state_dict):
        """Removes the 'model.' prefix if present in the state dict keys."""
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    def preprocess(self, image_path):
        img = np.array(Image.open(image_path).convert("RGB"))
        tensor = self.val_transform(image=img)["image"].unsqueeze(0)
        return tensor

    def predict(self, image_path):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        img_tensor = self.preprocess(image_path).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        
        # Ensure the predicted class is in the correct case for saving
        predicted_class_name = self.classes[idx]

        return {
            "predicted_class": predicted_class_name,
            "confidence": float(probs[idx]),
            "all_probabilities": {cls: float(p) for cls, p in zip(self.classes, probs)}
        }


# Global Detector Instance
MODEL_PATH = "models/saved_models/best_model.pth"

# Initialize detector safely
try:
    detector = Detector(model_path=MODEL_PATH)
except Exception as e:
    print(f"WARNING: Failed to load model at startup. Error: {e}")
    detector = Detector() # Initialize empty so server doesn't crash immediately


# -------------------------------
# Django Views
# -------------------------------
def home(request: HttpRequest) -> HttpResponse:
    recent_predictions = Prediction.objects.all().order_by('-created_at')[:5]
    total_predictions = Prediction.objects.count()

    return render(request, "detector/home.html", {
        "recent_predictions": recent_predictions,
        "total_predictions": total_predictions
    })


def upload_image(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            image = form.cleaned_data["image"]

            if detector.model is None:
                # Try reloading in case it was fixed since startup
                try:
                    detector.load_model(MODEL_PATH)
                except Exception:
                    messages.error(request, "Model could not be loaded. Please check server logs.")
                    return redirect("upload")

            try:
                # Create a temporary file to handle image processing
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{image.name}") as temp_file:
                    for chunk in image.chunks():
                        temp_file.write(chunk)
                    temp_file_path = temp_file.name

                # Perform prediction
                result = detector.predict(temp_file_path)

                # Clean up temp file
                os.remove(temp_file_path)

                # FIX 2: Multiply confidence by 100 before saving to store it as a percentage
                pred_obj = Prediction.objects.create(
                    image=image,
                    predicted_class=result["predicted_class"],
                    confidence=result["confidence"] * 100,
                    all_probabilities=json.dumps(result["all_probabilities"])
                )

                messages.success(request, "Prediction successful!")
                return redirect("result", pk=pred_obj.pk)

            except Exception as e:
                messages.error(request, f"Prediction error: {str(e)}")
                return redirect("upload")

    else:
        form = ImageUploadForm()

    return render(request, "detector/upload.html", {"form": form})


def result(request, pk):
    try:
        prediction = Prediction.objects.get(pk=pk)
        probabilities = prediction.get_probabilities_dict()

        # The chart data is already calculating percentages (p * 100)
        chart_data = {
            "labels": list(probabilities.keys()),
            "data": [p * 100 for p in probabilities.values()]
        }

        return render(request, "detector/result.html", {
            "prediction": prediction,
            "probabilities": probabilities,
            "chart_data": json.dumps(chart_data)
        })
    except Prediction.DoesNotExist:
        messages.error(request, "Prediction not found.")
        return redirect("home")


def history(request):
    items = Prediction.objects.all().order_by('-created_at')
    paginator = Paginator(items, 10)

    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    return render(request, "detector/history.html", {"predictions": page_obj})


def delete_prediction(request, pk):
    try:
        pred = Prediction.objects.get(pk=pk)
        if pred.image and os.path.exists(pred.image.path):
            os.remove(pred.image.path)
        pred.delete()
        messages.success(request, "Deleted successfully.")
    except Prediction.DoesNotExist:
        messages.error(request, "Prediction does not exist.")

    return redirect("history")


def train_model(request):
    if request.method == "POST":
        form = TrainingForm(request.POST)
        if form.is_valid():
            messages.info(request, "Training functionality is currently a placeholder.")
            return redirect("home")
    else:
        form = TrainingForm()

    sessions = TrainingSession.objects.all()[:5]

    return render(request, "detector/train.html", {
        "form": form,
        "training_sessions": sessions
    })


@require_http_methods(["POST"])
def api_predict(request):
    if "image" not in request.FILES:
        return JsonResponse({"error": "Image missing"}, status=400)

    image = request.FILES["image"]

    if detector.model is None:
        try:
            detector.load_model(MODEL_PATH)
        except Exception:
            return JsonResponse({"error": "Model not loaded on server"}, status=503)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{image.name}") as temp_file:
            for chunk in image.chunks():
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        result = detector.predict(temp_file_path)
        os.remove(temp_file_path)

        # FIX 2: Multiply confidence by 100 before saving to store it as a percentage
        pred_obj = Prediction.objects.create(
            image=image,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"] * 100,
            all_probabilities=json.dumps(result["all_probabilities"])
        )

        return JsonResponse({
            "success": True,
            "prediction_id": pred_obj.pk,
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "all_probabilities": result["all_probabilities"]
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def base(request):
    # Renders the template located at "your_app_name/templates/detector/base.html"
    return render(request, "detector/base.html")