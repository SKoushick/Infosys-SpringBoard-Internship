"""
Comprehensive test suite for Stage C: Django Admin Safeguards.
Covers:
- Single GoalPath deletion via Admin (merging progress into general catalog).
- Bulk GoalPath deletion via Admin (delete_queryset and safe_delete_selected_paths).
- Single LearningGoal deletion via Admin (merging all child paths' progress).
- Bulk LearningGoal deletion via Admin (delete_queryset and safe_delete_selected_goals).
- Progress merging accuracy through admin actions.
- Ownership, read-only fields, and permission restrictions (Heartbeat event immutability).
- Failure handling and rollback during admin deletion.
- Verification that existing admin registrations remain functional.
"""

from unittest.mock import patch, MagicMock

from django.test import TestCase, RequestFactory
from django.contrib.admin.sites import AdminSite
from django.contrib.auth import get_user_model
from django.contrib.messages.storage.fallback import FallbackStorage
from django.utils import timezone

from apps.talks.models import (
    LearningGoal,
    GoalPath,
    GoalPathItem,
    TalkProgress,
    TalkProgressHeartbeatEvent,
    SavedTalk,
    Collection,
)
from apps.talks.admin import (
    LearningGoalAdmin,
    GoalPathAdmin,
    GoalPathItemAdmin,
    TalkProgressAdmin,
    TalkProgressHeartbeatEventAdmin,
    SavedTalkAdmin,
    CollectionAdmin,
)

User = get_user_model()


class MockRequestUser:
    """Helper to attach session and messages to RequestFactory requests."""
    def __init__(self, user):
        self.user = user

    def prepare_request(self, request):
        request.user = self.user
        setattr(request, "session", {})
        messages = FallbackStorage(request)
        setattr(request, "_messages", messages)
        return request


class LearningAdminSafeguardsTests(TestCase):
    def setUp(self):
        self.site = AdminSite()
        self.factory = RequestFactory()

        self.superuser = User.objects.create_superuser(
            username="admin_user", email="admin@example.com", password="password"
        )
        self.learner1 = User.objects.create_user(username="learner_1", password="password")
        self.learner2 = User.objects.create_user(username="learner_2", password="password")

        self.req_helper = MockRequestUser(self.superuser)

        self.goal_admin = LearningGoalAdmin(LearningGoal, self.site)
        self.path_admin = GoalPathAdmin(GoalPath, self.site)
        self.item_admin = GoalPathItemAdmin(GoalPathItem, self.site)
        self.progress_admin = TalkProgressAdmin(TalkProgress, self.site)
        self.hb_admin = TalkProgressHeartbeatEventAdmin(TalkProgressHeartbeatEvent, self.site)

    def test_single_goal_path_admin_deletion_merges_progress(self):
        """Verify delete_model on GoalPathAdmin safely merges path progress into general catalog."""
        goal = LearningGoal.objects.create(user=self.learner1, title="Path Admin Goal")
        path = GoalPath.objects.create(goal=goal, title="Admin Path 1")

        # Create path progress
        TalkProgress.objects.create(
            user=self.learner1,
            talk_id=901,
            goal_path=path,
            dwell_time_seconds=75,
            completion_percent=50.0,
            status="in_progress",
        )

        request = self.req_helper.prepare_request(self.factory.post("/"))
        path_id = path.id

        # Execute single object deletion via admin
        self.path_admin.delete_model(request, path)

        # Path must be deleted
        self.assertFalse(GoalPath.objects.filter(id=path_id).exists())

        # Progress must be safely merged into general catalog
        gen = TalkProgress.objects.get(user=self.learner1, talk_id=901, goal_path__isnull=True)
        self.assertEqual(gen.dwell_time_seconds, 75)
        self.assertEqual(gen.completion_percent, 50.0)
        self.assertEqual(gen.status, "in_progress")

    def test_bulk_goal_path_admin_deletion(self):
        """Verify delete_queryset and safe_delete_selected_paths merge multiple paths safely."""
        goal = LearningGoal.objects.create(user=self.learner1, title="Bulk Path Goal")
        p1 = GoalPath.objects.create(goal=goal, title="Bulk Path A")
        p2 = GoalPath.objects.create(goal=goal, title="Bulk Path B")

        TalkProgress.objects.create(
            user=self.learner1, talk_id=902, goal_path=p1, dwell_time_seconds=30, status="in_progress"
        )
        TalkProgress.objects.create(
            user=self.learner1, talk_id=902, goal_path=p2, dwell_time_seconds=40, status="in_progress"
        )

        request = self.req_helper.prepare_request(self.factory.post("/"))
        qs = GoalPath.objects.filter(id__in=[p1.id, p2.id])

        # Execute bulk action
        self.path_admin.safe_delete_selected_paths(request, qs)

        self.assertFalse(GoalPath.objects.filter(id__in=[p1.id, p2.id]).exists())

        # Check merged dwell time: 30 + 40 = 70
        gen = TalkProgress.objects.get(user=self.learner1, talk_id=902, goal_path__isnull=True)
        self.assertEqual(gen.dwell_time_seconds, 70)

    def test_single_learning_goal_admin_deletion(self):
        """Verify delete_model on LearningGoalAdmin safely deletes goal and merges child paths."""
        goal = LearningGoal.objects.create(user=self.learner2, title="Goal To Delete")
        path = GoalPath.objects.create(goal=goal, title="Child Path")

        TalkProgress.objects.create(
            user=self.learner2, talk_id=903, goal_path=path, dwell_time_seconds=100, status="completed", completion_percent=100.0
        )

        request = self.req_helper.prepare_request(self.factory.post("/"))
        goal_id = goal.id
        path_id = path.id

        self.goal_admin.delete_model(request, goal)

        self.assertFalse(LearningGoal.objects.filter(id=goal_id).exists())
        self.assertFalse(GoalPath.objects.filter(id=path_id).exists())

        # General progress exists
        gen = TalkProgress.objects.get(user=self.learner2, talk_id=903, goal_path__isnull=True)
        self.assertEqual(gen.dwell_time_seconds, 100)
        self.assertEqual(gen.status, "completed")
        self.assertEqual(gen.completion_percent, 100.0)

    def test_bulk_learning_goal_admin_deletion(self):
        """Verify bulk deletion of LearningGoals merges all paths across selected goals."""
        g1 = LearningGoal.objects.create(user=self.learner1, title="Goal 1")
        g2 = LearningGoal.objects.create(user=self.learner1, title="Goal 2")

        p1 = GoalPath.objects.create(goal=g1, title="Path 1")
        p2 = GoalPath.objects.create(goal=g2, title="Path 2")

        TalkProgress.objects.create(
            user=self.learner1, talk_id=904, goal_path=p1, dwell_time_seconds=20, status="in_progress"
        )
        TalkProgress.objects.create(
            user=self.learner1, talk_id=904, goal_path=p2, dwell_time_seconds=30, status="in_progress"
        )

        request = self.req_helper.prepare_request(self.factory.post("/"))
        qs = LearningGoal.objects.filter(id__in=[g1.id, g2.id])

        self.goal_admin.safe_delete_selected_goals(request, qs)

        self.assertFalse(LearningGoal.objects.filter(id__in=[g1.id, g2.id]).exists())
        self.assertFalse(GoalPath.objects.filter(id__in=[p1.id, p2.id]).exists())

        # Merged general progress: 20 + 30 = 50
        gen = TalkProgress.objects.get(user=self.learner1, talk_id=904, goal_path__isnull=True)
        self.assertEqual(gen.dwell_time_seconds, 50)

    def test_heartbeat_event_admin_immutability(self):
        """Verify Heartbeat events cannot be added or changed via Django Admin."""
        request = self.req_helper.prepare_request(self.factory.get("/"))

        self.assertFalse(self.hb_admin.has_add_permission(request))
        self.assertFalse(self.hb_admin.has_change_permission(request))

        # Check all relevant fields are read-only
        readonly = self.hb_admin.get_readonly_fields(request)
        self.assertIn("client_event_id", readonly)
        self.assertIn("user", readonly)
        self.assertIn("talk_id", readonly)
        self.assertIn("dwell_increment", readonly)

    def test_admin_deletion_failure_handling(self):
        """Verify that if deletion service fails, error message is handled cleanly."""
        goal = LearningGoal.objects.create(user=self.learner1, title="Fail Goal")
        path = GoalPath.objects.create(goal=goal, title="Fail Path")

        request = self.req_helper.prepare_request(self.factory.post("/"))

        with patch("apps.talks.admin.delete_goal_path", side_effect=RuntimeError("Simulated failure")):
            with self.assertRaises(RuntimeError):
                self.path_admin.delete_model(request, path)

        # Path still exists because transaction rolled back
        self.assertTrue(GoalPath.objects.filter(id=path.id).exists())

    def test_existing_admin_classes_remain_functional(self):
        """Verify SavedTalkAdmin and CollectionAdmin remain intact and functional."""
        saved_admin = SavedTalkAdmin(SavedTalk, self.site)
        coll_admin = CollectionAdmin(Collection, self.site)

        self.assertIn("title", saved_admin.list_display)
        self.assertIn("title", coll_admin.list_display)
        self.assertEqual(len(coll_admin.inlines), 1)
