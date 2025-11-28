from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('base', views.base, name='base'),
    path('upload/', views.upload_image, name='upload'),
    path('result/<int:pk>/', views.result, name='result'),
    path('history/', views.history, name='history'),
    path('delete/<int:pk>/', views.delete_prediction, name='delete_prediction'),
    path('train/', views.train_model, name='train'),
    path('api/predict/', views.api_predict, name='api_predict'),
]