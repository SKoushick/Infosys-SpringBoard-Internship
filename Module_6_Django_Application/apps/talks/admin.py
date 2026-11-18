"""
Django Admin configuration for TEDMind talks, collections, and interactions.
"""

from django.contrib import admin
from apps.talks.models import (
    SavedTalk,
    Collection,
    CollectionItem,
    ViewingHistory,
    UserFeedback,
    RecommendationEvent,
)


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
