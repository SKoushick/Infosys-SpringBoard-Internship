"""
URL routes for talks catalog, details, collections, and saved talks.
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
    path("collections/create/", views.create_collection_view, name="create_collection"),
    path("collections/<int:collection_id>/", views.collection_detail_view, name="collection_detail"),
    path("api/collections/add/", views.add_to_collection_ajax, name="add_to_collection_ajax"),
    path("api/feedback/", views.record_feedback_ajax, name="record_feedback_ajax"),
    path("about/", views.about_view, name="about"),
]
