"""
Models for TEDMind Talks Application.
Stores user interactions, persistent bookmarks, custom collections, viewing history, feedback,
learning goals, structured learning paths, item sequences, progress tracking, and heartbeat telemetry.
"""

import uuid
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models
from django.utils import timezone
from .managers import (
    LearningGoalManager,
    GoalPathManager,
    TalkProgressManager,
    TalkProgressHeartbeatEventManager,
)


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


# ==============================================================================
# Phase 2 — Goal-Oriented Learning Platform & Progress Tracking Models
# ==============================================================================

class LearningGoal(models.Model):
    """
    User-defined learning goal with thematic focus and target pedagogical stage.
    """
    STAGE_CHOICES = [
        ("foundational", "Foundational"),
        ("intermediate", "Intermediate"),
        ("specialized", "Specialized"),
        ("all", "All Stages"),
    ]
    STATUS_CHOICES = [
        ("active", "Active"),
        ("completed", "Completed"),
        ("abandoned", "Abandoned"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="learning_goals",
        db_index=True,
    )
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    target_topics = models.JSONField(
        default=list,
        blank=True,
        help_text="Curated topic interests (e.g. ['artificial intelligence', 'neuroscience']).",
    )
    target_stage = models.CharField(
        max_length=20,
        choices=STAGE_CHOICES,
        default="all",
        help_text="Target difficulty progression stage.",
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="active",
        db_index=True,
        help_text="Authoritative lifecycle status: active, completed, or abandoned.",
    )
    is_active = models.BooleanField(
        default=True,
        db_index=True,
        help_text="Backward-compatible boolean indicator (True iff status == 'active').",
    )
    target_completion_date = models.DateField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = LearningGoalManager()

    class Meta:
        ordering = ["-status", "-created_at"]
        verbose_name = "Learning Goal"
        verbose_name_plural = "Learning Goals"
        constraints = [
            models.CheckConstraint(
                condition=(
                    (models.Q(status="active") & models.Q(is_active=True)) |
                    (models.Q(status__in=["completed", "abandoned"]) & models.Q(is_active=False))
                ),
                name="check_learninggoal_status_is_active_consistency",
            ),
        ]

    def __str__(self):
        return f"{self.title} ({self.user.username} - {self.get_status_display()})"

    def clean(self):
        super().clean()
        if not self.title or not self.title.strip():
            raise ValidationError({"title": "Goal title cannot be empty."})

        # Synchronize status and is_active to avoid contradictory values
        if self.status == "active":
            self.is_active = True
        else:
            self.is_active = False

        # Strict JSON topic validation
        if not isinstance(self.target_topics, list):
            raise ValidationError({"target_topics": "target_topics must be a JSON list of topic strings."})

        cleaned_topics = []
        seen = set()
        for idx, item in enumerate(self.target_topics):
            if type(item) is not str:
                raise ValidationError({"target_topics": f"Item at index {idx} must be a string, got {type(item).__name__}."})
            trimmed = item.strip().lower()
            if not trimmed:
                raise ValidationError({"target_topics": f"Item at index {idx} cannot be empty or pure whitespace."})
            if trimmed not in seen:
                seen.add(trimmed)
                cleaned_topics.append(trimmed)
        self.target_topics = sorted(cleaned_topics)

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)


class GoalPath(models.Model):
    """
    Curated learning path consisting of ordered TED Talks to fulfill a specific LearningGoal.
    """
    PATH_STAGE_CHOICES = [
        ("foundational", "Foundational"),
        ("intermediate", "Intermediate"),
        ("specialized", "Specialized"),
        ("mixed", "Full Progressive Journey"),
    ]
    STATUS_CHOICES = [
        ("draft", "Draft"),
        ("active", "Active"),
        ("completed", "Completed"),
        ("archived", "Archived"),
    ]

    goal = models.ForeignKey(
        LearningGoal,
        on_delete=models.CASCADE,
        related_name="paths",
        db_index=True,
    )
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    stage_focus = models.CharField(
        max_length=20,
        choices=PATH_STAGE_CHOICES,
        default="mixed",
        help_text="Pedagogical stage focus. 'mixed' represents multi-stage beginner-to-advanced progression.",
    )
    is_adaptive = models.BooleanField(
        default=True,
        help_text="Whether this path dynamically adjusts sequence based on learner engagement.",
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="active",
        db_index=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = GoalPathManager()

    class Meta:
        ordering = ["-updated_at"]
        verbose_name = "Goal Learning Path"
        verbose_name_plural = "Goal Learning Paths"

    def __str__(self):
        return f"{self.title} (Goal: {self.goal.title})"

    @property
    def total_items(self):
        return self.items.count()

    def clean(self):
        super().clean()
        if not self.title or not self.title.strip():
            raise ValidationError({"title": "Path title cannot be empty."})

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)


class GoalPathItem(models.Model):
    """
    Ordered TED Talk node inside a GoalPath with predicted pedagogical metadata.
    """
    STAGE_CHOICES = [
        ("foundational", "Foundational"),
        ("intermediate", "Intermediate"),
        ("specialized", "Specialized"),
    ]
    EVIDENCE_CHOICES = [
        ("high", "High"),
        ("medium", "Medium"),
        ("low", "Low"),
    ]

    goal_path = models.ForeignKey(
        GoalPath,
        on_delete=models.CASCADE,
        related_name="items",
        db_index=True,
    )
    talk_id = models.IntegerField(
        db_index=True,
        validators=[MinValueValidator(1)],
    )
    sequence_order = models.PositiveIntegerField(
        default=1,
        db_index=True,
        validators=[MinValueValidator(1)],
    )
    assigned_stage = models.CharField(
        max_length=20,
        choices=STAGE_CHOICES,
        default="foundational",
    )
    estimated_complexity = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Predicted pedagogical complexity score (0.0 to 1.0).",
    )
    evidence_quality = models.CharField(
        max_length=20,
        choices=EVIDENCE_CHOICES,
        default="high",
    )
    rationale = models.TextField(
        blank=True,
        help_text="Explainability reason why this talk was selected for this learning sequence.",
    )
    notes = models.TextField(blank=True, help_text="Learner's personal study notes.")
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["sequence_order"]
        verbose_name = "Goal Path Item"
        verbose_name_plural = "Goal Path Items"
        constraints = [
            models.UniqueConstraint(
                fields=["goal_path", "talk_id"],
                name="unique_talk_per_goal_path",
            ),
            models.UniqueConstraint(
                fields=["goal_path", "sequence_order"],
                name="unique_sequence_order_per_goal_path",
            ),
            models.CheckConstraint(
                condition=models.Q(sequence_order__gte=1),
                name="check_goalpathitem_sequence_order_positive",
            ),
            models.CheckConstraint(
                condition=models.Q(talk_id__gte=1),
                name="check_goalpathitem_talk_id_positive",
            ),
            models.CheckConstraint(
                condition=models.Q(estimated_complexity__isnull=True) | (
                    models.Q(estimated_complexity__gte=0.0) & models.Q(estimated_complexity__lte=1.0)
                ),
                name="check_goalpathitem_complexity_range",
            ),
        ]

    def __str__(self):
        return f"Item #{self.sequence_order} (Talk #{self.talk_id}) in {self.goal_path.title}"

    def clean(self):
        super().clean()
        if self.estimated_complexity is not None:
            import math
            if math.isnan(self.estimated_complexity) or math.isinf(self.estimated_complexity):
                raise ValidationError(
                    {"estimated_complexity": "Estimated complexity cannot be NaN or infinity."}
                )
            if not (0.0 <= self.estimated_complexity <= 1.0):
                raise ValidationError(
                    {"estimated_complexity": "Estimated complexity must be between 0.0 and 1.0."}
                )

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)


class TalkProgress(models.Model):
    """
    Tracks learner engagement, cumulative dwell time, completion status, and ratings.
    Supports path-associated progress (goal_path is set) and general catalog progress (goal_path is None).
    Uses on_delete=RESTRICT on goal_path to ensure safe transactional merging before path deletion.
    """
    STATUS_CHOICES = [
        ("not_started", "Not Started"),
        ("in_progress", "In Progress"),
        ("completed", "Completed"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="talk_progress",
        db_index=True,
    )
    talk_id = models.IntegerField(
        db_index=True,
        validators=[MinValueValidator(1)],
    )
    goal_path = models.ForeignKey(
        GoalPath,
        null=True,
        blank=True,
        on_delete=models.RESTRICT,
        related_name="progress_records",
        db_index=True,
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="not_started",
        db_index=True,
    )
    dwell_time_seconds = models.PositiveIntegerField(
        default=0,
        validators=[MinValueValidator(0)],
        help_text="Cumulative viewing duration in seconds.",
    )
    completion_percent = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
        help_text="Learner progress percentage (0.0 to 100.0).",
    )
    learner_rating = models.PositiveSmallIntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text="Subjective learner rating from 1 to 5.",
    )
    rated_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Timestamp when the learner rating was submitted or updated.",
    )
    last_engaged_at = models.DateTimeField(
        default=timezone.now,
        db_index=True,
        help_text="Timestamp of the most recent genuine user interaction with this talk.",
    )
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    objects = TalkProgressManager()

    class Meta:
        ordering = ["-last_engaged_at"]
        verbose_name = "Talk Progress"
        verbose_name_plural = "Talk Progress Records"
        constraints = [
            # 1. Path-specific progress uniqueness
            models.UniqueConstraint(
                fields=["user", "talk_id", "goal_path"],
                condition=models.Q(goal_path__isnull=False),
                name="unique_user_talk_per_goal_path",
            ),
            # 2. General catalog progress uniqueness (goal_path IS NULL)
            models.UniqueConstraint(
                fields=["user", "talk_id"],
                condition=models.Q(goal_path__isnull=True),
                name="unique_user_talk_general_progress",
            ),
            # 3. Status and completion percentage strict invariant
            models.CheckConstraint(
                condition=(
                    (models.Q(status="not_started") & models.Q(completion_percent=0.0)) |
                    (models.Q(status="in_progress") & models.Q(completion_percent__gte=0.0) & models.Q(completion_percent__lt=100.0)) |
                    (models.Q(status="completed") & models.Q(completion_percent=100.0))
                ),
                name="check_talkprogress_status_completion_consistency",
            ),
            # 4. Rating and rated_at bidirectional consistency
            models.CheckConstraint(
                condition=(
                    (models.Q(learner_rating__isnull=True) & models.Q(rated_at__isnull=True)) |
                    (models.Q(learner_rating__isnull=False) & models.Q(rated_at__isnull=False))
                ),
                name="check_talkprogress_rating_rated_at_consistency",
            ),
            # 5. Positive talk ID
            models.CheckConstraint(
                condition=models.Q(talk_id__gte=1),
                name="check_talkprogress_talk_id_positive",
            ),
            # 6. Non-negative dwell time
            models.CheckConstraint(
                condition=models.Q(dwell_time_seconds__gte=0),
                name="check_talkprogress_dwell_time_non_negative",
            ),
            # 7. Rating bounds (1 to 5 or null)
            models.CheckConstraint(
                condition=models.Q(learner_rating__isnull=True) | (
                    models.Q(learner_rating__gte=1) & models.Q(learner_rating__lte=5)
                ),
                name="check_talkprogress_learner_rating_range",
            ),
        ]

    def __str__(self):
        context = f"Path {self.goal_path_id}" if self.goal_path_id else "General Catalog"
        return f"Progress on Talk #{self.talk_id} by {self.user.username} ({context}: {self.status})"

    @property
    def completion_percentage(self):
        """Alias for completion_percent to ensure seamless compatibility across naming conventions."""
        return self.completion_percent

    @completion_percentage.setter
    def completion_percentage(self, value):
        self.completion_percent = value

    def clean(self):
        super().clean()
        # Cross-table ownership validation
        if self.goal_path_id is not None:
            if self.goal_path.goal.user_id != self.user_id:
                raise ValidationError(
                    {"goal_path": "Cross-user security violation: You cannot record progress against another user's learning path."}
                )

        # Rating / rated_at synchronization
        if self.learner_rating is not None and self.rated_at is None:
            self.rated_at = timezone.now()
        elif self.learner_rating is None:
            self.rated_at = None

        # Status / completion consistency validation
        if self.status == "completed":
            self.completion_percent = 100.0
            if not self.completed_at:
                self.completed_at = timezone.now()
        elif self.completion_percent >= 100.0:
            self.status = "completed"
            self.completion_percent = 100.0
            if not self.completed_at:
                self.completed_at = timezone.now()
        elif self.completion_percent > 0.0:
            self.status = "in_progress"
            self.completed_at = None
        elif self.status == "in_progress" and self.completion_percent == 0.0:
            # Valid: user started watching with 0% completion recorded
            self.completed_at = None
        elif self.status == "not_started":
            self.completion_percent = 0.0
            self.completed_at = None

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)


class TalkProgressHeartbeatEvent(models.Model):
    """
    Idempotent telemetry event log for video player heartbeat dwell increments.
    Guarantees exactly-once processing per (user, client_event_id) pair.
    """
    event_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="heartbeat_events",
        db_index=True,
    )
    client_event_id = models.CharField(
        max_length=64,
        db_index=True,
        help_text="Client-generated unique event UUID string for idempotent telemetry deduplication.",
    )
    talk_id = models.IntegerField(
        db_index=True,
        validators=[MinValueValidator(1)],
    )
    goal_path_id = models.IntegerField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Path context ID if viewing occurred inside a learning path.",
    )
    dwell_increment = models.PositiveIntegerField(
        validators=[MinValueValidator(1)],
        help_text="Viewing seconds increment reported by client player.",
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
    )

    objects = TalkProgressHeartbeatEventManager()

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Heartbeat Event"
        verbose_name_plural = "Heartbeat Events"
        constraints = [
            models.CheckConstraint(
                condition=models.Q(dwell_increment__gte=1),
                name="check_heartbeat_dwell_increment_positive",
            ),
            models.UniqueConstraint(
                fields=["user", "client_event_id"],
                name="unique_user_heartbeat_client_event",
            ),
        ]

    def __str__(self):
        return f"Event {self.event_id} ({self.dwell_increment}s on Talk #{self.talk_id} by {self.user.username})"

    def clean(self):
        super().clean()
        if not self.client_event_id or not str(self.client_event_id).strip():
            raise ValidationError({"client_event_id": "Client event ID is required and cannot be empty or pure whitespace."})
        try:
            uuid.UUID(str(self.client_event_id).strip())
        except (ValueError, AttributeError, TypeError):
            raise ValidationError({"client_event_id": f"Client event ID must be a valid UUID string, got {self.client_event_id!r}."})

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

