from django.db import models
from django.utils import timezone
import json

class Prediction(models.Model):
    """Store prediction history"""
    
    image = models.ImageField(upload_to='uploads/%Y/%m/%d/')
    predicted_class = models.CharField(max_length=50)
    confidence = models.FloatField()
    all_probabilities = models.JSONField()
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.predicted_class} - {self.confidence:.2%} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"
    
    def get_probabilities_dict(self):
        """Convert JSON probabilities to dict"""
        if isinstance(self.all_probabilities, str):
            return json.loads(self.all_probabilities)
        return self.all_probabilities


class TrainingSession(models.Model):
    """Store training session information"""
    
    model_type = models.CharField(max_length=50)
    base_model = models.CharField(max_length=50, blank=True)
    epochs = models.IntegerField()
    final_accuracy = models.FloatField(null=True, blank=True)
    final_loss = models.FloatField(null=True, blank=True)
    training_time = models.DurationField(null=True, blank=True)
    model_path = models.CharField(max_length=500)
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.model_type} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"