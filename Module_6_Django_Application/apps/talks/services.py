"""
Transactional Service Layer for Phase 2 Learning Platform.
Implements:
1. Safe transactional learning-path deletion (delete_goal_path).
2. Safe transactional learning-goal deletion (delete_learning_goal).
3. Resilient, idempotent video heartbeat recording (record_talk_heartbeat).
"""

import time
import random
import uuid
import logging
from typing import Optional, Dict, Any, Union

from django.db import transaction, models, IntegrityError, OperationalError
from django.utils import timezone
from django.contrib.auth import get_user_model

from .models import LearningGoal, GoalPath, TalkProgress, TalkProgressHeartbeatEvent

logger = logging.getLogger(__name__)
User = get_user_model()


# ==============================================================================
# Domain Exceptions
# ==============================================================================

class LearningServiceError(Exception):
    """Base exception for learning service errors."""
    pass


class GoalPathNotFoundError(LearningServiceError):
    """Raised when a requested GoalPath does not exist or is inaccessible."""
    pass


class LearningGoalNotFoundError(LearningServiceError):
    """Raised when a requested LearningGoal does not exist or is inaccessible."""
    pass


class GoalAccessDeniedError(LearningServiceError):
    """Raised when a user attempts to access or modify resources owned by another user."""
    pass


class InvalidHeartbeatError(LearningServiceError):
    """Raised when heartbeat telemetry contains invalid parameters."""
    pass


class HeartbeatConflictError(LearningServiceError):
    """Raised when a client_event_id is reused with a conflicting payload."""
    pass


# ==============================================================================
# Helper / Validation Functions
# ==============================================================================

def _resolve_user_id(user: Any) -> int:
    """Extract integer user ID from User instance or int."""
    if hasattr(user, "id"):
        return user.id
    if isinstance(user, int) and user > 0:
        return user
    raise ValueError(f"Invalid user specification: {user}")


def validate_dwell_increment(dwell_increment: Any) -> int:
    """
    Validate that dwell increment is a genuine positive Python integer >= 1.
    Strictly rejects bool (True/False), float (1.0), numeric strings ("15"), zero, and negative values.
    """
    if type(dwell_increment) is not int or dwell_increment < 1:
        raise InvalidHeartbeatError(
            f"dwell_increment must be a genuine positive Python integer >= 1, got {dwell_increment!r} ({type(dwell_increment).__name__})"
        )
    return dwell_increment


def validate_talk_id(talk_id: Any) -> int:
    """
    Validate that talk_id is a genuine positive Python integer >= 1.
    Strictly rejects bool (True/False), float (1.0), numeric strings ("15"), zero, and negative values.
    """
    if type(talk_id) is not int or talk_id < 1:
        raise InvalidHeartbeatError(
            f"talk_id must be a genuine positive Python integer >= 1, got {talk_id!r} ({type(talk_id).__name__})"
        )
    return talk_id


def validate_client_event_id(client_event_id: Any) -> str:
    """
    Validate that client_event_id is a valid, non-empty UUID string.
    Rejects missing, empty, whitespace-only, and malformed UUID values.
    """
    if not client_event_id or not isinstance(client_event_id, str) or not client_event_id.strip():
        raise InvalidHeartbeatError(
            "client_event_id is required and cannot be empty or pure whitespace."
        )
    trimmed = client_event_id.strip()
    try:
        val = uuid.UUID(trimmed)
    except (ValueError, AttributeError, TypeError):
        raise InvalidHeartbeatError(
            f"client_event_id must be a valid UUID string, got {client_event_id!r}"
        )
    return str(val)


# ==============================================================================
# Safe Path & Goal Deletion Services
# ==============================================================================

def _merge_single_progress_record(user_id: int, p: TalkProgress) -> TalkProgress:
    """
    Safely merge a path-specific TalkProgress record into general catalog progress.
    Preserves cumulative dwell time, highest completion percentage, latest timestamps,
    and valid ratings.
    """
    general = TalkProgress.objects.select_for_update().filter(
        user_id=user_id,
        talk_id=p.talk_id,
        goal_path__isnull=True,
    ).first()

    if general is None:
        # Create general progress with the path record's existing values
        general = TalkProgress.objects.create(
            user_id=user_id,
            talk_id=p.talk_id,
            goal_path=None,
            status=p.status,
            dwell_time_seconds=p.dwell_time_seconds,
            completion_percent=p.completion_percent,
            learner_rating=p.learner_rating,
            rated_at=p.rated_at,
            last_engaged_at=p.last_engaged_at,
            completed_at=p.completed_at,
        )
        return general

    # General progress already exists -> merge atomically
    general.dwell_time_seconds = general.dwell_time_seconds + p.dwell_time_seconds
    general.completion_percent = max(general.completion_percent, p.completion_percent)

    # Resolve status & completion
    if (
        general.completion_percent >= 100.0
        or general.status == "completed"
        or p.status == "completed"
    ):
        general.status = "completed"
        general.completion_percent = 100.0
        if not general.completed_at:
            general.completed_at = p.completed_at or timezone.now()
    elif general.completion_percent > 0.0 or general.dwell_time_seconds > 0:
        general.status = "in_progress"
        general.completed_at = None
    else:
        general.status = "not_started"
        general.completed_at = None

    # Resolve timestamps
    general.last_engaged_at = max(general.last_engaged_at, p.last_engaged_at)

    # Resolve rating & rated_at
    if p.learner_rating is not None and general.learner_rating is not None:
        # Both rated: pick latest rated_at
        if p.rated_at and general.rated_at:
            if p.rated_at > general.rated_at:
                general.learner_rating = p.learner_rating
                general.rated_at = p.rated_at
        elif p.rated_at and not general.rated_at:
            general.learner_rating = p.learner_rating
            general.rated_at = p.rated_at
    elif p.learner_rating is not None and general.learner_rating is None:
        general.learner_rating = p.learner_rating
        general.rated_at = p.rated_at or timezone.now()

    general.save()
    return general


def delete_goal_path(
    user: Any,
    path_id: int,
    *,
    max_retries: int = 3,
    backoff_base: float = 0.05,
) -> Dict[str, Any]:
    """
    Safely and transactionally delete a GoalPath.
    1. Validates user ownership.
    2. Locks the path and its associated TalkProgress records.
    3. Merges path-specific progress into general catalog progress without loss or double-counting.
    4. Deletes path-specific progress records.
    5. Deletes the GoalPath record.
    """
    user_id = _resolve_user_id(user)

    for attempt in range(max_retries):
        try:
            with transaction.atomic():
                path = GoalPath.objects.select_for_update().filter(id=path_id).first()
                if path is None:
                    raise GoalPathNotFoundError(f"GoalPath #{path_id} not found.")

                if path.goal.user_id != user_id:
                    raise GoalPathNotFoundError(
                        f"GoalPath #{path_id} not found or access denied."
                    )

                path_progress_list = list(
                    TalkProgress.objects.select_for_update().filter(goal_path=path)
                )

                # Merge each path-specific progress record into general progress
                for p in path_progress_list:
                    _merge_single_progress_record(user_id, p)

                # Delete all path-specific progress records
                TalkProgress.objects.filter(goal_path=path).delete()

                # Delete the GoalPath itself
                path.delete()

                return {
                    "path_id": path_id,
                    "deleted": True,
                    "merged_records_count": len(path_progress_list),
                }

        except OperationalError as exc:
            if "locked" in str(exc).lower() and attempt < max_retries - 1:
                sleep_time = backoff_base * (2 ** attempt) + random.uniform(0, 0.05)
                logger.warning(
                    "SQLite database locked in delete_goal_path (attempt %d/%d). Retrying in %.3fs...",
                    attempt + 1,
                    max_retries,
                    sleep_time,
                )
                time.sleep(sleep_time)
                continue
            raise


def delete_learning_goal(
    user: Any,
    goal_id: int,
    *,
    max_retries: int = 3,
    backoff_base: float = 0.05,
) -> Dict[str, Any]:
    """
    Safely and transactionally delete a LearningGoal and all its child GoalPaths.
    1. Validates user ownership.
    2. Locks the goal and all child paths.
    3. Safely merges progress across all child paths into general catalog progress.
    4. Deletes all child path progress records, child paths, and the goal.
    """
    user_id = _resolve_user_id(user)

    for attempt in range(max_retries):
        try:
            with transaction.atomic():
                goal = LearningGoal.objects.select_for_update().filter(id=goal_id).first()
                if goal is None:
                    raise LearningGoalNotFoundError(f"LearningGoal #{goal_id} not found.")

                if goal.user_id != user_id:
                    raise LearningGoalNotFoundError(
                        f"LearningGoal #{goal_id} not found or access denied."
                    )

                paths = list(goal.paths.select_for_update().all())
                total_merged = 0

                for path in paths:
                    path_progress_list = list(
                        TalkProgress.objects.select_for_update().filter(goal_path=path)
                    )
                    for p in path_progress_list:
                        _merge_single_progress_record(user_id, p)
                        total_merged += 1

                    TalkProgress.objects.filter(goal_path=path).delete()
                    path.delete()

                goal.delete()

                return {
                    "goal_id": goal_id,
                    "deleted": True,
                    "deleted_paths_count": len(paths),
                    "merged_records_count": total_merged,
                }

        except OperationalError as exc:
            if "locked" in str(exc).lower() and attempt < max_retries - 1:
                sleep_time = backoff_base * (2 ** attempt) + random.uniform(0, 0.05)
                logger.warning(
                    "SQLite database locked in delete_learning_goal (attempt %d/%d). Retrying in %.3fs...",
                    attempt + 1,
                    max_retries,
                    sleep_time,
                )
                time.sleep(sleep_time)
                continue
            raise


# ==============================================================================
# Resilient & Idempotent Heartbeat Recording
# ==============================================================================

def record_talk_heartbeat(
    user: Any,
    talk_id: int,
    goal_path_id: Optional[int] = None,
    dwell_increment: int = 15,
    client_event_id: Optional[str] = None,
    *,
    max_retries: int = 3,
    backoff_base: float = 0.05,
) -> Dict[str, Any]:
    """
    Record an idempotent video viewing heartbeat telemetry event.
    Requirements:
    - Stable, client-supplied event UUID (client_event_id).
    - Positive integer dwell increment >= 1.
    - Verified user and talk_id.
    - Path context verification: if goal_path_id is given, path must exist and belong to user.
      Never redirects a heartbeat to general progress if its specified path is deleted/missing.
    - Atomic persistence of heartbeat event and progress dwell time increment.
    - Idempotent deduplication on (user, client_event_id).
    - Rejects identical client_event_id with differing payload.
    - Retry wrapper on SQLite database-lock errors with exponential backoff.
    """
    user_id = _resolve_user_id(user)
    valid_talk_id = validate_talk_id(talk_id)
    valid_inc = validate_dwell_increment(dwell_increment)
    valid_client_id = validate_client_event_id(client_event_id)

    for attempt in range(max_retries):
        try:
            with transaction.atomic():
                # 1. Path ownership validation (if path context is specified)
                if goal_path_id is not None:
                    path = GoalPath.objects.select_for_update().filter(id=goal_path_id).first()
                    if path is None:
                        raise GoalPathNotFoundError(
                            f"GoalPath #{goal_path_id} not found or has been deleted."
                        )
                    if path.goal.user_id != user_id:
                        raise GoalAccessDeniedError(
                            f"GoalPath #{goal_path_id} does not belong to user #{user_id}."
                        )

                # 2. Check for existing event with this client_event_id for this user
                existing_event = TalkProgressHeartbeatEvent.objects.filter(
                    user_id=user_id,
                    client_event_id=valid_client_id,
                ).first()

                if existing_event is not None:
                    # Payload comparison
                    payload_matches = (
                        existing_event.talk_id == valid_talk_id
                        and existing_event.goal_path_id == goal_path_id
                        and existing_event.dwell_increment == valid_inc
                    )
                    if payload_matches:
                        # Idempotent replay: return existing progress without incrementing
                        progress = TalkProgress.objects.filter(
                            user_id=user_id,
                            talk_id=valid_talk_id,
                            goal_path_id=goal_path_id,
                        ).first()
                        return {
                            "status": "duplicate",
                            "progress": progress,
                            "event_id": str(existing_event.event_id),
                            "increment_applied": False,
                        }
                    else:
                        raise HeartbeatConflictError(
                            f"Heartbeat client_event_id '{valid_client_id}' already registered with different payload."
                        )

                # 3. Register heartbeat event
                try:
                    event = TalkProgressHeartbeatEvent.objects.create(
                        user_id=user_id,
                        client_event_id=valid_client_id,
                        talk_id=valid_talk_id,
                        goal_path_id=goal_path_id,
                        dwell_increment=valid_inc,
                    )
                except (IntegrityError, models.ProtectedError):
                    # Concurrent duplicate race caught by unique constraint
                    existing_event = TalkProgressHeartbeatEvent.objects.filter(
                        user_id=user_id,
                        client_event_id=valid_client_id,
                    ).first()
                    if existing_event is not None:
                        if (
                            existing_event.talk_id == valid_talk_id
                            and existing_event.goal_path_id == goal_path_id
                            and existing_event.dwell_increment == valid_inc
                        ):
                            progress = TalkProgress.objects.filter(
                                user_id=user_id,
                                talk_id=valid_talk_id,
                                goal_path_id=goal_path_id,
                            ).first()
                            return {
                                "status": "duplicate",
                                "progress": progress,
                                "event_id": str(existing_event.event_id),
                                "increment_applied": False,
                            }
                        raise HeartbeatConflictError(
                            f"Heartbeat client_event_id '{valid_client_id}' already registered with different payload."
                        )
                    raise

                # 4. Apply dwell increment to TalkProgress
                qs = TalkProgress.objects.filter(
                    user_id=user_id,
                    talk_id=valid_talk_id,
                    goal_path_id=goal_path_id,
                )
                now = timezone.now()

                if qs.exists():
                    # Atomic F-expression increment
                    qs.update(
                        dwell_time_seconds=models.F("dwell_time_seconds") + valid_inc,
                        last_engaged_at=now,
                    )
                    progress = qs.first()
                    if progress.status == "not_started":
                        progress.status = "in_progress"
                        progress.save(update_fields=["status"])
                else:
                    # First heartbeat: create new progress record
                    try:
                        progress = TalkProgress.objects.create(
                            user_id=user_id,
                            talk_id=valid_talk_id,
                            goal_path_id=goal_path_id,
                            dwell_time_seconds=valid_inc,
                            status="in_progress",
                            completion_percent=0.0,
                            last_engaged_at=now,
                        )
                    except IntegrityError:
                        # Concurrently created by another transaction
                        qs.update(
                            dwell_time_seconds=models.F("dwell_time_seconds") + valid_inc,
                            last_engaged_at=now,
                        )
                        progress = qs.first()

                return {
                    "status": "success",
                    "progress": progress,
                    "event_id": str(event.event_id),
                    "increment_applied": True,
                }

        except OperationalError as exc:
            if "locked" in str(exc).lower() and attempt < max_retries - 1:
                sleep_time = backoff_base * (2 ** attempt) + random.uniform(0, 0.05)
                logger.warning(
                    "SQLite database locked in record_talk_heartbeat (attempt %d/%d). Retrying in %.3fs...",
                    attempt + 1,
                    max_retries,
                    sleep_time,
                )
                time.sleep(sleep_time)
                continue
            raise
