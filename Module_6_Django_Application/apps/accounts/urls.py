"""
URL configuration for apps.accounts.
"""

from django.urls import path
from apps.accounts import views

app_name = "accounts"

urlpatterns = [
    path("register/", views.register_view, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("profile/", views.profile_view, name="profile"),
    path("onboarding/", views.onboarding_view, name="onboarding"),
    path("api/toggle-topic/", views.toggle_topic_interest_ajax, name="toggle_topic_ajax"),
]
