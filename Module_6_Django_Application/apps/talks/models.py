"""
Models for TEDMind Talks Application.
Stores user interactions, persistent bookmarks, custom collections, viewing history, and feedback.
"""

from django.conf import settings
from django.db import models


class SavedTalk(models.Model):
    """
    User bookmarked talk. Supports both registered users and guest sessions.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="saved_talks",
        db_index=True,
    )
    session_key = models.CharField(max_length=64, blank=True, null=True, db_index=True)
    talk_id = models.IntegerField(db_index=True)
    title = models.CharField(max_length=255)
    speaker = models.CharField(max_length=255)
    tags = models.CharField(max_length=255, blank=True)
    is_liked = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        owner = self.user.username if self.user else f"Guest ({self.session_key[:8]}...)"
        return f"{self.speaker}: {self.title} (Owner: {owner})"


class Collection(models.Model):
    """
    User-curated playlist / learning path of TED talks.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="collections",
    )
    title = models.CharField(max_length=150)
    description = models.TextField(blank=True)
    is_public = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        unique_together = ("user", "title")

    def __str__(self):
        return f"{self.title} (by {self.user.username})"

    @property
    def item_count(self):
        return self.items.count()


class CollectionItem(models.Model):
    """
    Individual talk item within a collection.
    """
    collection = models.ForeignKey(
        Collection,
        on_delete=models.CASCADE,
        related_name="items",
    )
    talk_id = models.IntegerField(db_index=True)
    notes = models.TextField(blank=True, help_text="Personal learning notes for this talk.")
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["added_at"]
        unique_together = ("collection", "talk_id")

    def __str__(self):
        return f"Talk #{self.talk_id} in {self.collection.title}"


class ViewingHistory(models.Model):
    """
    Tracks talk views for user history and transparent personalization.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="viewing_history",
        db_index=True,
    )
    session_key = models.CharField(max_length=64, blank=True, null=True, db_index=True)
    talk_id = models.IntegerField(db_index=True)
    viewed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-viewed_at"]

    def __str__(self):
        user_repr = self.user.username if self.user else f"Guest ({self.session_key[:8]}...)"
        return f"Talk #{self.talk_id} viewed by {user_repr} at {self.viewed_at}"


class UserFeedback(models.Model):
    """
    Explicit user feedback (like, dislike, not interested) on recommendations.
    """
    FEEDBACK_CHOICES = [
        ("like", "Like"),
        ("dislike", "Dislike"),
        ("not_interested", "Not Interested"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="talk_feedback",
    )
    session_key = models.CharField(max_length=64, blank=True, null=True, db_index=True)
    talk_id = models.IntegerField(db_index=True)
    feedback_type = models.CharField(max_length=20, choices=FEEDBACK_CHOICES)
    reason = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.feedback_type.upper()} on Talk #{self.talk_id}"


class RecommendationEvent(models.Model):
    """
    Audit log of recommendation executions for latency analysis and user behavior.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="recommendation_events",
    )
    session_key = models.CharField(max_length=64, blank=True, null=True)
    seed_talk_id = models.IntegerField(null=True, blank=True)
    query_text = models.CharField(max_length=255, blank=True)
    method = models.CharField(max_length=30, default="hybrid")
    recommended_talk_ids = models.JSONField(default=list)
    latency_ms = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Rec Event ({self.method}) at {self.created_at}"
