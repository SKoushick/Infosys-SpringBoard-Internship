"""
Models for TEDMind Talks Application.
Stores user interactions (bookmarks, liked talks) per browser session.
"""

from django.db import models


class SavedTalk(models.Model):
    """User bookmarked/liked talk stored against their anonymous session."""
    session_key = models.CharField(max_length=64, db_index=True)
    talk_id = models.IntegerField(db_index=True)
    title = models.CharField(max_length=255)
    speaker = models.CharField(max_length=255)
    tags = models.CharField(max_length=255, blank=True)
    is_liked = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        unique_together = ("session_key", "talk_id")

    def __str__(self):
        return f"{self.speaker}: {self.title} (Session: {self.session_key[:8]}...)"
