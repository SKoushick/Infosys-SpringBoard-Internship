"""
URL configuration for TEDMind REST API.
"""

from django.urls import path
from apps.api import views

app_name = "api"

urlpatterns = [
    path("health/", views.api_health_check, name="health"),
    path("talks/", views.api_talks_list, name="talks_list"),
    path("talks/<int:talk_id>/", views.api_talk_detail, name="talk_detail"),
    path("talks/<int:talk_id>/similar/", views.api_talk_similar, name="talk_similar"),
    path("recommendations/generate/", views.api_generate_recommendations, name="generate_recommendations"),
    path("feedback/", views.api_feedback, name="feedback"),
    path("users/me/", views.api_user_me, name="user_me"),
    path("users/me/saved/", views.api_user_saved, name="user_saved"),
    path("users/me/saved/<int:talk_id>/", views.api_user_saved, name="user_saved_delete"),
]
