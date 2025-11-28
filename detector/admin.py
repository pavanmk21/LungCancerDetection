from django.contrib import admin
from .models import Prediction, TrainingSession

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['predicted_class', 'confidence', 'created_at']
    list_filter = ['predicted_class', 'created_at']
    search_fields = ['predicted_class']
    readonly_fields = ['created_at']

@admin.register(TrainingSession)
class TrainingSessionAdmin(admin.ModelAdmin):
    list_display = ['model_type', 'base_model', 'final_accuracy', 'created_at']
    list_filter = ['model_type', 'created_at']
    readonly_fields = ['created_at']