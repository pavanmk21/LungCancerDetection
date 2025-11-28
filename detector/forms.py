from django import forms
from .models import Prediction

class ImageUploadForm(forms.Form):
    """Form for uploading images for prediction"""
    
    image = forms.ImageField(
        label='Select Chest X-ray or CT Scan',
        help_text='Supported formats: JPG, PNG (Max size: 10MB)',
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'image/*'
        })
    )
    
    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            # Validate file size (10MB max)
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError("Image file too large ( > 10MB )")
            
            # Validate file type
            if not image.content_type in ['image/jpeg', 'image/png', 'image/jpg']:
                raise forms.ValidationError("Unsupported file type. Please upload JPG or PNG.")
        
        return image


class TrainingForm(forms.Form):
    """Form for training model"""
    
    MODEL_CHOICES = [
        ('cnn', 'Custom CNN'),
        ('efficientnet', 'EfficientNetB0 (Transfer Learning)'),
        ('resnet', 'ResNet50 (Transfer Learning)'),
    ]
    
    model_type = forms.ChoiceField(
        choices=MODEL_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    epochs = forms.IntegerField(
        min_value=1,
        max_value=100,
        initial=50,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    batch_size = forms.IntegerField(
        min_value=8,
        max_value=128,
        initial=32,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    learning_rate = forms.FloatField(
        min_value=0.00001,
        max_value=0.1,
        initial=0.001,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.0001'})
    )