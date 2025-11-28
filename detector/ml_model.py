import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


class LungCancerDetector:

    def __init__(self, model_path=None, model_name='efficientnet_b3', device=None):
        self.model = None

        # Model path
        self.model_path = model_path or getattr(
            settings,
            'ML_MODEL_PATH',
            'models/saved_models/best_model.pth'
        )

        self.model_name = model_name
        self.classes = None

        # Device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Default EfficientNet-B3 size (will auto-update after loading model)
        self.img_size = 300

        # Define transforms (will be rebuilt if img_size changes after loading)
        self.val_transform = self.build_transform(self.img_size)

        # Auto-load if exists
        if os.path.exists(self.model_path):
            self.load_model(self.model_path)
        else:
            logger.warning(f"Model file not found at {self.model_path}")

    def build_transform(self, size):
        """Return validation/inference transform."""
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def load_model(self, model_path=None):
        model_path = model_path or self.model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # Load metadata
        self.classes = checkpoint["classes"]
        self.model_name = checkpoint["model_name"]

        # Load image size if included in checkpoint
        if "img_size" in checkpoint:
            self.img_size = checkpoint["img_size"]

        # Rebuild transforms with correct size
        self.val_transform = self.build_transform(self.img_size)

        # Build model
        self.model = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=len(self.classes)
        )

        # Load weights STRICTLY
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.to(self.device).eval()

        logger.info(f"Loaded model: {self.model_name}")
        logger.info(f"Classes: {self.classes}")
        logger.info(f"Image size: {self.img_size}")

        return True

    def preprocess_image(self, image_path):
        """Read + preprocess image."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = np.array(Image.open(image_path).convert('RGB'))
        tensor = self.val_transform(image=img)["image"].unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image_path):
        """Run inference + return probabilities."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model().")

        tensor = self.preprocess_image(image_path)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))

        return {
            "predicted_class": self.classes[idx],
            "confidence": float(probs[idx]),
            "all_probabilities": {
                cls: float(p) for cls, p in zip(self.classes, probs)
            }
        }


# Global singleton instance
detector = LungCancerDetector()

# Auto-load if file exists
if os.path.exists(detector.model_path):
    detector.load_model()
