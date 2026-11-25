"""
Custom QuerySets and Managers for Phase 2 Learning Platform Models.
Enforces strict user scoping, composable filters, and zero data leakage.
"""
from django.db import models


class LearningGoalQuerySet(models.QuerySet):
    def for_user(self, user):
        """Scope query to a specific user (User instance or user ID)."""
        user_id = user.id if hasattr(user, "id") else user
        return self.filter(user_id=user_id)

    def active(self):
        """Active learning goals."""
        return self.filter(status="active")

    def completed(self):
        """Completed learning goals."""
        return self.filter(status="completed")

    def abandoned(self):
        """Abandoned learning goals."""
        return self.filter(status="abandoned")


class LearningGoalManager(models.Manager.from_queryset(LearningGoalQuerySet)):
    pass


class GoalPathQuerySet(models.QuerySet):
    def for_user(self, user):
        """Scope paths to those belonging to the user's learning goals."""
        user_id = user.id if hasattr(user, "id") else user
        return self.filter(goal__user_id=user_id)

    def active(self):
        """Active learning paths."""
        return self.filter(status="active")

    def completed(self):
        """Completed learning paths."""
        return self.filter(status="completed")

    def archived(self):
        """Archived learning paths."""
        return self.filter(status="archived")


class GoalPathManager(models.Manager.from_queryset(GoalPathQuerySet)):
    pass


class TalkProgressQuerySet(models.QuerySet):
    def for_user(self, user):
        """Scope progress records to a specific user."""
        user_id = user.id if hasattr(user, "id") else user
        return self.filter(user_id=user_id)

    def general(self):
        """Progress records across the general catalog (not tied to any learning path)."""
        return self.filter(goal_path__isnull=True)

    def path_associated(self):
        """Progress records bound to specific learning paths."""
        return self.filter(goal_path__isnull=False)

    def for_path(self, path):
        """Scope progress records to a specific GoalPath instance or ID."""
        path_id = path.id if hasattr(path, "id") else path
        return self.filter(goal_path_id=path_id)

    def completed(self):
        """Completed talk progress."""
        return self.filter(status="completed")

    def in_progress(self):
        """In-progress talk records."""
        return self.filter(status="in_progress")

    def not_started(self):
        """Not-started talk progress records."""
        return self.filter(status="not_started")


class TalkProgressManager(models.Manager.from_queryset(TalkProgressQuerySet)):
    pass


class TalkProgressHeartbeatEventQuerySet(models.QuerySet):
    def for_user(self, user):
        """Scope heartbeat events to a specific user."""
        user_id = user.id if hasattr(user, "id") else user
        return self.filter(user_id=user_id)

    def for_talk(self, talk_id):
        """Filter events for a specific catalog talk."""
        return self.filter(talk_id=talk_id)

    def for_path(self, path):
        """Filter events for a specific path ID or GoalPath instance."""
        path_id = path.id if hasattr(path, "id") else path
        return self.filter(goal_path_id=path_id)


class TalkProgressHeartbeatEventManager(models.Manager.from_queryset(TalkProgressHeartbeatEventQuerySet)):
    pass
