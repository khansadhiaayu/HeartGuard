from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('prediksi/', views.predict, name='predict'),
    path('riwayat/', views.history, name='history'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('chatbot/', views.chatbot, name='chatbot'),
]