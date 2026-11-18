"""
URL configuration for TEDMind TED Talks Recommendation System.
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("apps.api.urls")),
    path("accounts/", include("apps.accounts.urls")),
    path("", include("apps.talks.urls")),
    path("recommendations/", include("apps.recommendations.urls")),
]
