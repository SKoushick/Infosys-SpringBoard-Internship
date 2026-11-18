"""
Django Admin registration for UserProfile.
"""

from django.contrib import admin
from apps.accounts.models import UserProfile


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "onboarding_completed", "recommendations_enabled", "created_at")
    list_filter = ("onboarding_completed", "recommendations_enabled", "created_at")
    search_fields = ("user__username", "user__email", "bio")
    readonly_fields = ("created_at", "updated_at")
