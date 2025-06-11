from django.urls import path # type: ignore
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('index/', views.home, name='index'),
    path('upload/', views.upload_and_predict, name='upload'),
    path('loading/', views.loading, name='loading'),
    path('result/', views.result, name='result'),
    path('features/', views.features, name='features'),
]
