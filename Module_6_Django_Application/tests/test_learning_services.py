"""
Comprehensive application-level test suite for Stage B Managers and Transactional Services.
Covers:
- User-scoped manager filtering and composition.
- Cross-user security and access rejection.
- General and path-specific progress creation.
- First heartbeat dwell accumulation and atomic increments.
- Heartbeat idempotency (identical duplicate replay vs. conflicting payload rejection).
- Missing/invalid client event UUIDs and invalid dwell increments.
- Heartbeat after path deletion (no silent redirection).
- Synchronized concurrency races: Heartbeat-first and Deletion-first.
- Progress merging accuracy (dwell time sum, max completion, rating timestamp resolution, zero double-counting).
- Safe transactional learning-goal deletion.
- User account deletion cascade behavior (models.RESTRICT vs cascade root).
- SQLite lock retry behavior.
"""

import threading
import time
import uuid
from unittest.mock import patch

from django.test import TestCase, TransactionTestCase
from django.contrib.auth import get_user_model
from django.db import OperationalError, connection
from django.utils import timezone

from apps.talks.models import (
    LearningGoal,
    GoalPath,
    GoalPathItem,
    TalkProgress,
    TalkProgressHeartbeatEvent,
)
from apps.talks.services import (
    delete_goal_path,
    delete_learning_goal,
    record_talk_heartbeat,
    GoalPathNotFoundError,
    LearningGoalNotFoundError,
    GoalAccessDeniedError,
    InvalidHeartbeatError,
    HeartbeatConflictError,
)

User = get_user_model()


class LearningManagerTests(TestCase):
    """Test custom QuerySets and Managers for learning models."""

    @classmethod
    def setUpTestData(cls):
        cls.user1 = User.objects.create_user(username="mgr_user_1", password="password")
        cls.user2 = User.objects.create_user(username="mgr_user_2", password="password")

        cls.goal1_active = LearningGoal.objects.create(
            user=cls.user1, title="Goal 1 Active", target_topics=["ai"], status="active"
        )
        cls.goal1_comp = LearningGoal.objects.create(
            user=cls.user1, title="Goal 1 Comp", target_topics=["neuro"], status="completed"
        )
        cls.goal2_active = LearningGoal.objects.create(
            user=cls.user2, title="Goal 2 Active", target_topics=["robotics"], status="active"
        )

        cls.path1 = GoalPath.objects.create(goal=cls.goal1_active, title="Path 1", status="active")
        cls.path2 = GoalPath.objects.create(goal=cls.goal2_active, title="Path 2", status="active")

        cls.prog_gen = TalkProgress.objects.create(
            user=cls.user1, talk_id=101, status="in_progress", completion_percent=25.0
        )
        cls.prog_path = TalkProgress.objects.create(
            user=cls.user1, talk_id=102, goal_path=cls.path1, status="completed", completion_percent=100.0
        )
        cls.prog_u2 = TalkProgress.objects.create(
            user=cls.user2, talk_id=101, status="not_started", completion_percent=0.0
        )

    def test_learning_goal_manager_for_user_and_status(self):
        """Verify LearningGoal.objects.for_user() scopes strictly to user."""
        u1_goals = LearningGoal.objects.for_user(self.user1)
        self.assertEqual(u1_goals.count(), 2)
        self.assertEqual(u1_goals.active().count(), 1)
        self.assertEqual(u1_goals.completed().count(), 1)

        u2_goals = LearningGoal.objects.for_user(self.user2)
        self.assertEqual(u2_goals.count(), 1)
        self.assertEqual(u2_goals.first().title, "Goal 2 Active")

    def test_goal_path_manager_for_user(self):
        """Verify GoalPath.objects.for_user() traverses goal relationship to user."""
        u1_paths = GoalPath.objects.for_user(self.user1)
        self.assertEqual(u1_paths.count(), 1)
        self.assertEqual(u1_paths.first().title, "Path 1")

        u2_paths = GoalPath.objects.for_user(self.user2)
        self.assertEqual(u2_paths.count(), 1)
        self.assertEqual(u2_paths.first().title, "Path 2")

    def test_talk_progress_manager_filtering(self):
        """Verify TalkProgress.objects methods: for_user, general, path_associated, for_path."""
        u1_prog = TalkProgress.objects.for_user(self.user1)
        self.assertEqual(u1_prog.count(), 2)
        self.assertEqual(u1_prog.general().count(), 1)
        self.assertEqual(u1_prog.path_associated().count(), 1)
        self.assertEqual(u1_prog.for_path(self.path1).count(), 1)
        self.assertEqual(u1_prog.completed().count(), 1)
        self.assertEqual(u1_prog.in_progress().count(), 1)

        # Cross-user check
        u2_prog = TalkProgress.objects.for_user(self.user2)
        self.assertEqual(u2_prog.count(), 1)
        self.assertEqual(u2_prog.not_started().count(), 1)


class HeartbeatServiceTests(TestCase):
    """Test record_talk_heartbeat service and edge cases."""

    def setUp(self):
        self.user = User.objects.create_user(username="hb_user", password="password")
        self.other_user = User.objects.create_user(username="other_hb_user", password="password")

        self.goal = LearningGoal.objects.create(
            user=self.user, title="HB Goal", target_topics=["science"]
        )
        self.path = GoalPath.objects.create(goal=self.goal, title="HB Path")

    def test_first_heartbeat_and_subsequent_increments(self):
        """Verify first heartbeat creates progress with full increment, next adds atomically."""
        evt1 = str(uuid.uuid4())
        res1 = record_talk_heartbeat(
            user=self.user,
            talk_id=201,
            goal_path_id=self.path.id,
            dwell_increment=15,
            client_event_id=evt1,
        )
        self.assertEqual(res1["status"], "success")
        self.assertTrue(res1["increment_applied"])
        p1 = res1["progress"]
        self.assertEqual(p1.dwell_time_seconds, 15)
        self.assertEqual(p1.status, "in_progress")

        # Second heartbeat for same talk on same path
        evt2 = str(uuid.uuid4())
        res2 = record_talk_heartbeat(
            user=self.user,
            talk_id=201,
            goal_path_id=self.path.id,
            dwell_increment=15,
            client_event_id=evt2,
        )
        self.assertEqual(res2["status"], "success")
        p2 = res2["progress"]
        self.assertEqual(p2.dwell_time_seconds, 30)
        self.assertEqual(p2.id, p1.id)

    def test_general_catalog_heartbeat(self):
        """Verify heartbeat without goal_path_id updates general catalog progress."""
        evt = str(uuid.uuid4())
        res = record_talk_heartbeat(
            user=self.user,
            talk_id=301,
            goal_path_id=None,
            dwell_increment=20,
            client_event_id=evt,
        )
        self.assertEqual(res["status"], "success")
        prog = res["progress"]
        self.assertIsNone(prog.goal_path)
        self.assertEqual(prog.dwell_time_seconds, 20)
        self.assertEqual(prog.status, "in_progress")

    def test_identical_duplicate_heartbeat_is_idempotent(self):
        """Verify identical duplicate heartbeat returns existing record without double-incrementing."""
        evt = str(uuid.uuid4())
        res1 = record_talk_heartbeat(
            user=self.user,
            talk_id=202,
            goal_path_id=self.path.id,
            dwell_increment=15,
            client_event_id=evt,
        )
        self.assertEqual(res1["status"], "success")
        self.assertEqual(res1["progress"].dwell_time_seconds, 15)

        # Replay identical event
        res2 = record_talk_heartbeat(
            user=self.user,
            talk_id=202,
            goal_path_id=self.path.id,
            dwell_increment=15,
            client_event_id=evt,
        )
        self.assertEqual(res2["status"], "duplicate")
        self.assertFalse(res2["increment_applied"])
        self.assertEqual(res2["progress"].dwell_time_seconds, 15)
        self.assertEqual(TalkProgressHeartbeatEvent.objects.count(), 1)

    def test_conflicting_duplicate_heartbeat_rejected(self):
        """Verify duplicate client_event_id with differing payload raises HeartbeatConflictError."""
        evt = str(uuid.uuid4())
        record_talk_heartbeat(
            user=self.user,
            talk_id=203,
            goal_path_id=self.path.id,
            dwell_increment=15,
            client_event_id=evt,
        )

        # Same event ID but different talk_id
        with self.assertRaises(HeartbeatConflictError):
            record_talk_heartbeat(
                user=self.user,
                talk_id=999,
                goal_path_id=self.path.id,
                dwell_increment=15,
                client_event_id=evt,
            )

        # Same event ID but different dwell_increment
        with self.assertRaises(HeartbeatConflictError):
            record_talk_heartbeat(
                user=self.user,
                talk_id=203,
                goal_path_id=self.path.id,
                dwell_increment=30,
                client_event_id=evt,
            )

    def test_client_event_uuid_validation(self):
        """
        Verify that client_event_id strictly requires a valid UUID string.
        Rejects missing, empty, whitespace-only, non-string, and malformed UUID values.
        """
        invalid_uuid_cases = [
            None,
            "",
            "   ",
            12345,
            "not-a-uuid",
            "12345678-1234-1234-1234",  # Truncated
            "g50e8400-e29b-41d4-a716-446655440000",  # Invalid hex char 'g'
            "550e8400-e29b-41d4-a716-446655440000-extra",
            "550e8400e29b41d4a716446655440000extra",
        ]
        for bad_id in invalid_uuid_cases:
            with self.subTest(bad_id=bad_id):
                with self.assertRaises(InvalidHeartbeatError):
                    record_talk_heartbeat(
                        user=self.user,
                        talk_id=204,
                        client_event_id=bad_id,
                    )

        # Model-level enforcement
        from django.core.exceptions import ValidationError
        with self.assertRaises(ValidationError):
            hb_bad = TalkProgressHeartbeatEvent(
                user=self.user,
                talk_id=204,
                dwell_increment=15,
                client_event_id="malformed-uuid-string",
            )
            hb_bad.save()

        # Valid UUID string succeeds
        valid_uuid = str(uuid.uuid4())
        res = record_talk_heartbeat(
            user=self.user,
            talk_id=204,
            dwell_increment=15,
            client_event_id=valid_uuid,
        )
        self.assertEqual(res["status"], "success")

    def test_strict_integer_validation_talk_id_and_dwell_increment(self):
        """
        Verify strict integer validation for talk_id and dwell_increment.
        Rejects:
        - True and False
        - Floats, including 1.0, 15.0
        - Numeric strings such as "15", "1"
        - Zero
        - Negative integers
        Accepts only genuine positive Python integers.
        """
        valid_uuid = str(uuid.uuid4())

        # 1. Invalid talk_id values
        invalid_talk_ids = [
            True,
            False,
            1.0,
            205.5,
            -1.0,
            "1",
            "205",
            0,
            -1,
            -999,
            None,
            [],
        ]
        for bad_talk_id in invalid_talk_ids:
            with self.subTest(bad_talk_id=bad_talk_id):
                with self.assertRaises(InvalidHeartbeatError):
                    record_talk_heartbeat(
                        user=self.user,
                        talk_id=bad_talk_id,
                        dwell_increment=15,
                        client_event_id=str(uuid.uuid4()),
                    )

        # 2. Invalid dwell_increment values
        invalid_dwell_increments = [
            True,
            False,
            1.0,
            15.0,
            0.5,
            -15.0,
            "15",
            "1",
            0,
            -1,
            -15,
            None,
            {},
        ]
        for bad_inc in invalid_dwell_increments:
            with self.subTest(bad_inc=bad_inc):
                with self.assertRaises(InvalidHeartbeatError):
                    record_talk_heartbeat(
                        user=self.user,
                        talk_id=205,
                        dwell_increment=bad_inc,
                        client_event_id=str(uuid.uuid4()),
                    )

        # 3. Genuine positive integers succeed
        res = record_talk_heartbeat(
            user=self.user,
            talk_id=int(205),
            dwell_increment=int(30),
            client_event_id=str(uuid.uuid4()),
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["progress"].dwell_time_seconds, 30)

    def test_cross_user_path_access_denied(self):
        """Verify heartbeat against another user's path raises GoalAccessDeniedError."""
        other_goal = LearningGoal.objects.create(user=self.other_user, title="Other Goal")
        other_path = GoalPath.objects.create(goal=other_goal, title="Other Path")

        with self.assertRaises(GoalAccessDeniedError):
            record_talk_heartbeat(
                user=self.user,
                talk_id=205,
                goal_path_id=other_path.id,
                dwell_increment=15,
                client_event_id=str(uuid.uuid4()),
            )

    def test_heartbeat_after_path_deletion_raises_not_found(self):
        """Verify heartbeat targeting deleted path raises GoalPathNotFoundError and is not redirected."""
        temp_path = GoalPath.objects.create(goal=self.goal, title="Temp Path")
        temp_path_id = temp_path.id
        delete_goal_path(self.user, temp_path_id)

        with self.assertRaises(GoalPathNotFoundError):
            record_talk_heartbeat(
                user=self.user,
                talk_id=206,
                goal_path_id=temp_path_id,
                dwell_increment=15,
                client_event_id=str(uuid.uuid4()),
            )
        # Ensure no general progress was created
        self.assertFalse(TalkProgress.objects.filter(user=self.user, talk_id=206).exists())


class DeletionAndMergeServiceTests(TestCase):
    """Test delete_goal_path and delete_learning_goal transactional services."""

    def setUp(self):
        self.user = User.objects.create_user(username="del_user", password="password")
        self.other_user = User.objects.create_user(username="other_del_user", password="password")

        self.goal = LearningGoal.objects.create(
            user=self.user, title="Main Goal", target_topics=["technology"]
        )
        self.path = GoalPath.objects.create(goal=self.goal, title="Main Path")

    def test_delete_goal_path_merges_progress_into_general_accurately(self):
        """Verify path progress merges cleanly: dwell sum, max completion, rating with latest timestamp."""
        t1 = timezone.now() - timezone.timedelta(hours=2)
        t2 = timezone.now() - timezone.timedelta(hours=1)

        # Existing general progress
        TalkProgress.objects.create(
            user=self.user,
            talk_id=501,
            goal_path=None,
            dwell_time_seconds=60,
            completion_percent=40.0,
            status="in_progress",
            learner_rating=3,
            rated_at=t1,
            last_engaged_at=t1,
        )

        # Path progress for same talk with higher completion and later rating
        TalkProgress.objects.create(
            user=self.user,
            talk_id=501,
            goal_path=self.path,
            dwell_time_seconds=45,
            completion_percent=85.0,
            status="in_progress",
            learner_rating=5,
            rated_at=t2,
            last_engaged_at=t2,
        )

        # Path progress for another talk (no existing general progress)
        TalkProgress.objects.create(
            user=self.user,
            talk_id=502,
            goal_path=self.path,
            dwell_time_seconds=120,
            completion_percent=100.0,
            status="completed",
            learner_rating=4,
            rated_at=t2,
            last_engaged_at=t2,
            completed_at=t2,
        )

        # Execute safe deletion
        res = delete_goal_path(self.user, self.path.id)
        self.assertTrue(res["deleted"])
        self.assertEqual(res["merged_records_count"], 2)

        # Path must be gone
        self.assertFalse(GoalPath.objects.filter(id=self.path.id).exists())

        # Check Talk 501 general progress
        gen_501 = TalkProgress.objects.get(user=self.user, talk_id=501, goal_path__isnull=True)
        self.assertEqual(gen_501.dwell_time_seconds, 105)  # 60 + 45
        self.assertEqual(gen_501.completion_percent, 85.0)  # max(40, 85)
        self.assertEqual(gen_501.status, "in_progress")
        self.assertEqual(gen_501.learner_rating, 5)  # later rating
        self.assertEqual(gen_501.rated_at, t2)
        self.assertEqual(gen_501.last_engaged_at, t2)

        # Check Talk 502 general progress (newly created from path)
        gen_502 = TalkProgress.objects.get(user=self.user, talk_id=502, goal_path__isnull=True)
        self.assertEqual(gen_502.dwell_time_seconds, 120)
        self.assertEqual(gen_502.completion_percent, 100.0)
        self.assertEqual(gen_502.status, "completed")
        self.assertEqual(gen_502.learner_rating, 4)

    def test_delete_goal_path_cross_user_rejection(self):
        """Verify attempting to delete another user's path raises GoalPathNotFoundError."""
        with self.assertRaises(GoalPathNotFoundError):
            delete_goal_path(self.other_user, self.path.id)
        self.assertTrue(GoalPath.objects.filter(id=self.path.id).exists())

    def test_delete_learning_goal_safely_merges_multiple_paths(self):
        """Verify delete_learning_goal merges all child paths and removes goal."""
        path2 = GoalPath.objects.create(goal=self.goal, title="Second Path")

        TalkProgress.objects.create(
            user=self.user, talk_id=601, goal_path=self.path, dwell_time_seconds=30, status="in_progress"
        )
        TalkProgress.objects.create(
            user=self.user, talk_id=601, goal_path=path2, dwell_time_seconds=20, status="in_progress"
        )

        res = delete_learning_goal(self.user, self.goal.id)
        self.assertTrue(res["deleted"])
        self.assertEqual(res["deleted_paths_count"], 2)

        # Check goal and paths deleted
        self.assertFalse(LearningGoal.objects.filter(id=self.goal.id).exists())
        self.assertFalse(GoalPath.objects.filter(id=self.path.id).exists())
        self.assertFalse(GoalPath.objects.filter(id=path2.id).exists())

        # Check merged general progress: 30 + 20 = 50
        gen = TalkProgress.objects.get(user=self.user, talk_id=601, goal_path__isnull=True)
        self.assertEqual(gen.dwell_time_seconds, 50)

    def test_user_deletion_cascades_safely_under_restrict(self):
        """Verify that deleting a User cascades through Goal, Path, and Progress under models.RESTRICT."""
        test_user = User.objects.create_user(username="cascade_user", password="password")
        g = LearningGoal.objects.create(user=test_user, title="Cascade Goal")
        p = GoalPath.objects.create(goal=g, title="Cascade Path")
        prog = TalkProgress.objects.create(user=test_user, talk_id=701, goal_path=p, dwell_time_seconds=10)
        evt = TalkProgressHeartbeatEvent.objects.create(
            user=test_user, talk_id=701, goal_path_id=p.id, dwell_increment=10, client_event_id=str(uuid.uuid4())
        )

        user_id = test_user.id
        test_user.delete()

        # All records should be cleanly removed by Django deletion collector
        self.assertFalse(LearningGoal.objects.filter(user_id=user_id).exists())
        self.assertFalse(GoalPath.objects.filter(goal__user_id=user_id).exists())
        self.assertFalse(TalkProgress.objects.filter(user_id=user_id).exists())
        self.assertFalse(TalkProgressHeartbeatEvent.objects.filter(user_id=user_id).exists())


class ConcurrencyAndRaceTests(TransactionTestCase):
    """
    True concurrent multi-connection race tests:
    1. Heartbeat-first vs Deletion race.
    2. Deletion-first vs Heartbeat race.
    3. SQLite database lock retry recovery.
    """

    def setUp(self):
        self.user = User.objects.create_user(username="race_user", password="password")
        self.goal = LearningGoal.objects.create(user=self.user, title="Race Goal")
        self.path = GoalPath.objects.create(goal=self.goal, title="Race Path")

    def tearDown(self):
        # Close any lingering connections
        connection.close()

    def test_heartbeat_first_deletion_race(self):
        """
        Thread A begins heartbeat, commits 15s.
        Thread B waits for heartbeat to commit, then deletes path and merges progress.
        Verify: Both succeed; general progress contains heartbeat dwell time.
        """
        path_id = self.path.id
        user_id = self.user.id
        talk_id = 801
        client_evt = str(uuid.uuid4())

        barrier = threading.Barrier(2)
        results = {}

        def thread_heartbeat():
            connection.close()
            try:
                # Synchronize start
                barrier.wait(timeout=5)
                res = record_talk_heartbeat(
                    user=user_id,
                    talk_id=talk_id,
                    goal_path_id=path_id,
                    dwell_increment=15,
                    client_event_id=client_evt,
                )
                results["hb"] = res
            except Exception as e:
                results["hb_err"] = e
            finally:
                connection.close()

        def thread_deletion():
            connection.close()
            try:
                barrier.wait(timeout=5)
                # Let heartbeat proceed first
                time.sleep(0.15)
                res = delete_goal_path(user=user_id, path_id=path_id)
                results["del"] = res
            except Exception as e:
                results["del_err"] = e
            finally:
                connection.close()

        t1 = threading.Thread(target=thread_heartbeat)
        t2 = threading.Thread(target=thread_deletion)

        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        self.assertNotIn("hb_err", results, f"Heartbeat error: {results.get('hb_err')}")
        self.assertNotIn("del_err", results, f"Deletion error: {results.get('del_err')}")

        # Check path deleted
        self.assertFalse(GoalPath.objects.filter(id=path_id).exists())

        # Check merged general progress: must have the 15 seconds
        gen = TalkProgress.objects.filter(user_id=user_id, talk_id=talk_id, goal_path__isnull=True).first()
        self.assertIsNotNone(gen, "General progress record should have been created by merge")
        self.assertEqual(gen.dwell_time_seconds, 15)

    def test_deletion_first_heartbeat_race(self):
        """
        Thread A deletes path.
        Thread B attempts heartbeat on path after deletion starts/completes.
        Verify: Deletion succeeds; Heartbeat raises GoalPathNotFoundError and does NOT orphan or redirect progress.
        """
        path_id = self.path.id
        user_id = self.user.id
        talk_id = 802
        client_evt = str(uuid.uuid4())

        barrier = threading.Barrier(2)
        results = {}

        def thread_deletion():
            connection.close()
            try:
                barrier.wait(timeout=5)
                res = delete_goal_path(user=user_id, path_id=path_id)
                results["del"] = res
            except Exception as e:
                results["del_err"] = e
            finally:
                connection.close()

        def thread_heartbeat():
            connection.close()
            try:
                barrier.wait(timeout=5)
                # Let deletion acquire lock and execute first
                time.sleep(0.2)
                res = record_talk_heartbeat(
                    user=user_id,
                    talk_id=talk_id,
                    goal_path_id=path_id,
                    dwell_increment=15,
                    client_event_id=client_evt,
                )
                results["hb"] = res
            except GoalPathNotFoundError as e:
                results["hb_expected_err"] = e
            except Exception as e:
                results["hb_unexpected_err"] = e
            finally:
                connection.close()

        t1 = threading.Thread(target=thread_deletion)
        t2 = threading.Thread(target=thread_heartbeat)

        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        self.assertNotIn("del_err", results, f"Deletion error: {results.get('del_err')}")
        self.assertNotIn("hb_unexpected_err", results, f"Unexpected HB error: {results.get('hb_unexpected_err')}")
        self.assertIn("hb_expected_err", results, "Heartbeat should have raised GoalPathNotFoundError")

        # Verify no orphan or general progress was created for talk 802
        self.assertFalse(TalkProgress.objects.filter(user_id=user_id, talk_id=talk_id).exists())

    def test_sqlite_lock_retry_behavior(self):
        """
        Verify that retry wrapper catches transient OperationalError('database is locked')
        and recovers on a subsequent attempt without raising to caller.
        """
        call_count = [0]
        original_record_talk_heartbeat = record_talk_heartbeat

        # We simulate 1 locked error followed by success
        with patch("apps.talks.services.TalkProgressHeartbeatEvent.objects.create") as mock_create:
            def side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise OperationalError("database is locked")
                # Return actual instance on attempt 2
                return TalkProgressHeartbeatEvent.objects.raw("SELECT 1")[0]

            # Test that retry recovers
            # We test retry directly on delete_goal_path by mocking select_for_update once
            with patch("apps.talks.models.GoalPath.objects.select_for_update") as mock_lock:
                lock_calls = [0]
                def lock_side_effect(*args, **kwargs):
                    lock_calls[0] += 1
                    if lock_calls[0] == 1:
                        raise OperationalError("database is locked")
                    return GoalPath.objects.filter(id=self.path.id)

                mock_lock.side_effect = lock_side_effect
                res = delete_goal_path(user=self.user, path_id=self.path.id, max_retries=3, backoff_base=0.01)
                self.assertTrue(res["deleted"])
                self.assertGreater(lock_calls[0], 1)
