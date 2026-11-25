"""
Django Admin configuration for TEDMind talks, collections, and interactions.
"""

from django.contrib import admin, messages
from apps.talks.models import (
    SavedTalk,
    Collection,
    CollectionItem,
    ViewingHistory,
    UserFeedback,
    RecommendationEvent,
    LearningGoal,
    GoalPath,
    GoalPathItem,
    TalkProgress,
    TalkProgressHeartbeatEvent,
)
from apps.talks.services import delete_goal_path, delete_learning_goal


@admin.register(SavedTalk)
class SavedTalkAdmin(admin.ModelAdmin):
    list_display = ("title", "speaker", "user", "session_key", "created_at")
    list_filter = ("created_at", "is_liked")
    search_fields = ("title", "speaker", "user__username", "session_key")


class CollectionItemInline(admin.TabularInline):
    model = CollectionItem
    extra = 1


@admin.register(Collection)
class CollectionAdmin(admin.ModelAdmin):
    list_display = ("title", "user", "is_public", "item_count", "created_at")
    list_filter = ("is_public", "created_at")
    search_fields = ("title", "description", "user__username")
    inlines = [CollectionItemInline]


@admin.register(ViewingHistory)
class ViewingHistoryAdmin(admin.ModelAdmin):
    list_display = ("talk_id", "user", "session_key", "viewed_at")
    list_filter = ("viewed_at",)
    search_fields = ("talk_id", "user__username", "session_key")


@admin.register(UserFeedback)
class UserFeedbackAdmin(admin.ModelAdmin):
    list_display = ("talk_id", "feedback_type", "user", "reason", "created_at")
    list_filter = ("feedback_type", "created_at")
    search_fields = ("talk_id", "user__username", "reason")


@admin.register(RecommendationEvent)
class RecommendationEventAdmin(admin.ModelAdmin):
    list_display = ("method", "user", "seed_talk_id", "latency_ms", "created_at")
    list_filter = ("method", "created_at")
    search_fields = ("query_text", "user__username")


# ==============================================================================
# Phase 2 — Goal-Oriented Learning Platform & Safeguarded Admin Registrations
# ==============================================================================

class GoalPathItemInline(admin.TabularInline):
    model = GoalPathItem
    extra = 0
    ordering = ["sequence_order"]
    fields = ("sequence_order", "talk_id", "assigned_stage", "estimated_complexity", "evidence_quality", "rationale", "notes")


class GoalPathInline(admin.TabularInline):
    model = GoalPath
    extra = 0
    fields = ("title", "stage_focus", "status", "is_adaptive", "created_at")
    readonly_fields = ("created_at",)
    show_change_link = True


@admin.register(LearningGoal)
class LearningGoalAdmin(admin.ModelAdmin):
    list_display = ("title", "user", "status", "is_active", "target_stage", "target_completion_date", "created_at")
    list_filter = ("status", "is_active", "target_stage", "created_at")
    search_fields = ("title", "description", "user__username")
    readonly_fields = ("is_active", "created_at", "updated_at")
    inlines = [GoalPathInline]
    actions = ["safe_delete_selected_goals"]

    def delete_model(self, request, obj):
        """Route single LearningGoal deletion through delete_learning_goal service."""
        try:
            res = delete_learning_goal(user=obj.user, goal_id=obj.id)
            self.message_user(
                request,
                f"Successfully deleted LearningGoal '{obj.title}'. Deleted {res['deleted_paths_count']} child path(s) "
                f"and safely merged {res['merged_records_count']} progress records into general catalog.",
                messages.SUCCESS,
            )
        except Exception as e:
            self.message_user(request, f"Error deleting LearningGoal '{obj.title}': {e}", messages.ERROR)
            raise

    def delete_queryset(self, request, queryset):
        """Route bulk LearningGoal deletion through delete_learning_goal service."""
        total_deleted = 0
        total_paths = 0
        total_merged = 0
        for obj in list(queryset):
            try:
                res = delete_learning_goal(user=obj.user, goal_id=obj.id)
                total_deleted += 1
                total_paths += res["deleted_paths_count"]
                total_merged += res["merged_records_count"]
            except Exception as e:
                self.message_user(request, f"Error deleting LearningGoal #{obj.id}: {e}", messages.ERROR)
        if total_deleted > 0:
            self.message_user(
                request,
                f"Safely deleted {total_deleted} LearningGoal(s). Deleted {total_paths} child path(s) and merged {total_merged} progress records.",
                messages.SUCCESS,
            )

    @admin.action(description="Safely delete selected LearningGoals (merge child path progress)")
    def safe_delete_selected_goals(self, request, queryset):
        self.delete_queryset(request, queryset)


@admin.register(GoalPath)
class GoalPathAdmin(admin.ModelAdmin):
    list_display = ("title", "goal", "get_user", "stage_focus", "status", "is_adaptive", "total_items", "updated_at")
    list_filter = ("status", "stage_focus", "is_adaptive")
    search_fields = ("title", "description", "goal__title", "goal__user__username")
    readonly_fields = ("created_at", "updated_at")
    inlines = [GoalPathItemInline]
    actions = ["safe_delete_selected_paths"]

    @admin.display(description="User")
    def get_user(self, obj):
        return obj.goal.user.username

    def delete_model(self, request, obj):
        """Route single GoalPath deletion through delete_goal_path service."""
        try:
            res = delete_goal_path(user=obj.goal.user, path_id=obj.id)
            self.message_user(
                request,
                f"Successfully deleted GoalPath '{obj.title}'. Merged {res['merged_records_count']} progress record(s) into general catalog.",
                messages.SUCCESS,
            )
        except Exception as e:
            self.message_user(request, f"Error deleting GoalPath '{obj.title}': {e}", messages.ERROR)
            raise

    def delete_queryset(self, request, queryset):
        """Route bulk GoalPath deletion through delete_goal_path service."""
        total_deleted = 0
        total_merged = 0
        for obj in list(queryset):
            try:
                res = delete_goal_path(user=obj.goal.user, path_id=obj.id)
                total_deleted += 1
                total_merged += res["merged_records_count"]
            except Exception as e:
                self.message_user(request, f"Error deleting GoalPath #{obj.id}: {e}", messages.ERROR)
        if total_deleted > 0:
            self.message_user(
                request,
                f"Safely deleted {total_deleted} GoalPath(s). Merged {total_merged} progress record(s) into general catalog.",
                messages.SUCCESS,
            )

    @admin.action(description="Safely delete selected GoalPaths (merge progress into general catalog)")
    def safe_delete_selected_paths(self, request, queryset):
        self.delete_queryset(request, queryset)


@admin.register(GoalPathItem)
class GoalPathItemAdmin(admin.ModelAdmin):
    list_display = ("id", "goal_path", "talk_id", "sequence_order", "assigned_stage", "estimated_complexity", "evidence_quality")
    list_filter = ("assigned_stage", "evidence_quality")
    search_fields = ("talk_id", "goal_path__title", "goal_path__goal__user__username")
    readonly_fields = ("added_at",)


@admin.register(TalkProgress)
class TalkProgressAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "talk_id", "get_path", "status", "dwell_time_seconds", "completion_percent", "learner_rating", "last_engaged_at")
    list_filter = ("status", "learner_rating")
    search_fields = ("talk_id", "user__username", "goal_path__title")
    readonly_fields = ("created_at", "last_engaged_at", "completed_at", "rated_at")

    @admin.display(description="Path Context")
    def get_path(self, obj):
        return obj.goal_path.title if obj.goal_path else "General Catalog"


@admin.register(TalkProgressHeartbeatEvent)
class TalkProgressHeartbeatEventAdmin(admin.ModelAdmin):
    list_display = ("event_id", "client_event_id", "user", "talk_id", "goal_path_id", "dwell_increment", "created_at")
    list_filter = ("created_at",)
    search_fields = ("client_event_id", "talk_id", "user__username")
    readonly_fields = ("event_id", "client_event_id", "user", "talk_id", "goal_path_id", "dwell_increment", "created_at")

    def has_add_permission(self, request):
        """Heartbeat events are immutable telemetry logs: cannot be manually created."""
        return False

    def has_change_permission(self, request, obj=None):
        """Heartbeat events are immutable telemetry logs: cannot be modified."""
        return False
