"""
User Account and Profile Models for TEDMind.
Provides persistent user-level topic preferences and personalization settings.
"""

from django.conf import settings
from django.db import models


class UserProfile(models.Model):
    """
    Extends Django's User model with personalized TED discovery preferences.
    """
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="profile",
    )
    bio = models.TextField(blank=True, max_length=500)
    preferred_topics = models.JSONField(
        default=list,
        blank=True,
        help_text="Curated topic interests selected by user (e.g., ['technology', 'education']).",
    )
    onboarding_completed = models.BooleanField(
        default=False,
        help_text="Flag indicating whether user completed the interest selection onboarding.",
    )
    recommendations_enabled = models.BooleanField(
        default=True,
        help_text="Toggle allowing user to pause/resume algorithmic personalization.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "User Profile"
        verbose_name_plural = "User Profiles"

    def __str__(self):
        return f"Profile of {self.user.username}"

    def get_preferred_topics_list(self):
        if isinstance(self.preferred_topics, list):
            return self.preferred_topics
        return []

    def set_preferred_topics_list(self, topics):
        cleaned = [t.strip().lower() for t in topics if isinstance(t, str) and t.strip()]
        self.preferred_topics = sorted(list(set(cleaned)))
