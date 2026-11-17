"""
URL routes for talks catalog, details, and saved talks.
"""
from django.urls import path
from apps.talks import views

app_name = "talks"

urlpatterns = [
    path("", views.home_view, name="home"),
    path("explore/", views.explore_view, name="explore"),
    path("talks/<int:talk_id>/", views.talk_detail_view, name="detail"),
    path("talks/<int:talk_id>/toggle-save/", views.toggle_save_talk_view, name="toggle_save"),
    path("saved/", views.saved_talks_view, name="saved"),
    path("about/", views.about_view, name="about"),
]
