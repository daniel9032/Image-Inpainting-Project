from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views
from .views import post_request, return_result

app_name = 'image_inpainting'
urlpatterns = [
	# Two paths: with or without given image
	path('', views.index, name='index'),
	path('register/', post_request, name='register'),
	path('result/', return_result, name='result'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)