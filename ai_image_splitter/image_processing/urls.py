from django.urls import path
from .views import ProcessImageView

urlpatterns = [
    path('process/', ProcessImageView.as_view(), name='process_image'),
]
