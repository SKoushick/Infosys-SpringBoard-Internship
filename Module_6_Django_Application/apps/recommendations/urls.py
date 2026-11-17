"""
URL routes for recommendations.
"""
from django.urls import path
from apps.recommendations import views

app_name = "recommendations"

urlpatterns = [
    path("", views.recommend_form_view, name="form"),
    path("results/", views.recommend_results_view, name="results"),
    path("api/search-titles/", views.api_search_titles_view, name="api_search_titles"),
]
