"""
REST API Views for TEDMind Platform.
Provides documented JSON endpoints for talks, recommendations, and user interactions.
"""

import json
import time
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST, require_http_methods
from django.contrib.auth.decorators import login_required

from apps.recommendations.services import get_recommender_service
from apps.talks.models import SavedTalk, UserFeedback, RecommendationEvent


@require_GET
def api_health_check(request):
    """
    GET /api/health/
    System observability and health check endpoint.
    """
    service = get_recommender_service()
    data = {
        "status": "healthy" if service.is_initialized else "degraded",
        "service": "TEDMind Recommender Engine",
        "catalog_size": len(service.id_to_talk),
        "models_loaded": {
            "tfidf_recommender": service.tfidf_rec is not None,
            "semantic_recommender": service.semantic_rec is not None,
            "hybrid_recommender": service.hybrid_rec is not None,
        },
        "timestamp": time.time(),
    }
    status_code = 200 if service.is_initialized else 503
    return JsonResponse(data, status=status_code)


@require_GET
def api_talks_list(request):
    """
    GET /api/talks/
    Query parameters: q (search term), tag, speaker, page (default 1), per_page (default 12).
    """
    service = get_recommender_service()
    query = request.GET.get("q", "").strip()
    tag = request.GET.get("tag", "").strip()
    speaker = request.GET.get("speaker", "").strip()

    try:
        page = int(request.GET.get("page", 1))
        per_page = min(50, max(1, int(request.GET.get("per_page", 12))))
    except (ValueError, TypeError):
        page = 1
        per_page = 12

    talks, total = service.search_talks(
        query=query,
        tag=tag,
        speaker=speaker,
        page=page,
        per_page=per_page,
    )

    total_pages = (total + per_page - 1) // per_page if total > 0 else 1

    return JsonResponse({
        "total_count": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "results": talks,
    })


@require_GET
def api_talk_detail(request, talk_id: int):
    """
    GET /api/talks/<talk_id>/
    Returns rich metadata for a single TED talk.
    """
    service = get_recommender_service()
    talk = service.get_talk_by_id(talk_id)
    if not talk:
        return JsonResponse({"error": f"Talk #{talk_id} not found"}, status=404)
    return JsonResponse(talk)


@require_GET
def api_talk_similar(request, talk_id: int):
    """
    GET /api/talks/<talk_id>/similar/
    Returns companion recommendations for a talk.
    """
    service = get_recommender_service()
    if talk_id not in service.id_to_talk:
        return JsonResponse({"error": f"Talk #{talk_id} not found"}, status=404)

    try:
        top_k = min(20, max(1, int(request.GET.get("top_k", 3))))
    except (ValueError, TypeError):
        top_k = 3

    method = request.GET.get("method", "hybrid").lower()
    apply_mmr = request.GET.get("apply_mmr", "1") in ("1", "true", "True")

    recs = service.recommend_for_talk(
        seed_talk_id=talk_id,
        method=method,
        top_k=top_k,
        apply_mmr=apply_mmr,
    )
    return JsonResponse({"seed_talk_id": talk_id, "recommendations": recs})


@csrf_exempt
@require_POST
def api_generate_recommendations(request):
    """
    POST /api/recommendations/generate/
    Accepts JSON body:
    {
      "seed_talk_id": 1,
      "query_text": "machine learning robotics",
      "interests": ["technology", "education"],
      "method": "hybrid" | "tfidf" | "semantic",
      "top_k": 5,
      "apply_mmr": true
    }
    """
    service = get_recommender_service()
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
    except json.JSONDecodeError:
        return JsonResponse({"error": "Malformed JSON payload"}, status=400)

    seed_talk_id = body.get("seed_talk_id")
    query_text = str(body.get("query_text", "")).strip()
    raw_interests = body.get("interests", [])
    if isinstance(raw_interests, str):
        selected_tags = [t.strip() for t in raw_interests.split(",") if t.strip()]
    elif isinstance(raw_interests, list):
        selected_tags = [str(t).strip() for t in raw_interests if str(t).strip()]
    else:
        selected_tags = []

    method = str(body.get("method", "hybrid")).lower().strip()
    if method not in ("tfidf", "semantic", "hybrid"):
        method = "hybrid"

    try:
        top_k = min(20, max(1, int(body.get("top_k", 10))))
    except (ValueError, TypeError):
        top_k = 10

    apply_mmr = bool(body.get("apply_mmr", True))

    if not seed_talk_id and not query_text and not selected_tags:
        return JsonResponse({
            "error": "Provide at least one input signal: seed_talk_id, query_text, or interests."
        }, status=400)

    t0 = time.perf_counter()

    if seed_talk_id:
        try:
            st_id = int(seed_talk_id)
        except (ValueError, TypeError):
            return JsonResponse({"error": "Invalid seed_talk_id"}, status=400)

        if st_id not in service.id_to_talk:
            return JsonResponse({"error": f"Seed talk #{st_id} not found"}, status=404)

        recommendations = service.recommend_for_talk(
            seed_talk_id=st_id,
            method=method,
            top_k=top_k,
            apply_mmr=apply_mmr,
        )
    else:
        recommendations = service.recommend_by_interests(
            user_interests=selected_tags,
            query_text=query_text,
            method=method,
            top_k=top_k,
            apply_mmr=apply_mmr,
        )

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    return JsonResponse({
        "status": "success",
        "method": method,
        "top_k": top_k,
        "apply_mmr": apply_mmr,
        "latency_ms": latency_ms,
        "count": len(recommendations),
        "recommendations": recommendations,
    })


@require_GET
def api_user_me(request):
    """
    GET /api/users/me/
    Returns current user profile information or anonymous status.
    """
    if not request.user.is_authenticated:
        return JsonResponse({"authenticated": False, "username": None, "session_key": request.session.session_key})

    profile = getattr(request.user, "profile", None)
    return JsonResponse({
        "authenticated": True,
        "username": request.user.username,
        "email": request.user.email,
        "preferred_topics": profile.get_preferred_topics_list() if profile else [],
        "recommendations_enabled": profile.recommendations_enabled if profile else True,
        "saved_count": SavedTalk.objects.filter(user=request.user).count(),
    })


@csrf_exempt
@require_http_methods(["GET", "POST", "DELETE"])
def api_user_saved(request, talk_id: int = None):
    """
    GET /api/users/me/saved/ -> List saved talks
    POST /api/users/me/saved/ -> Add talk to saved (body: {"talk_id": 42})
    DELETE /api/users/me/saved/<talk_id>/ -> Remove saved talk
    """
    user = request.user if request.user.is_authenticated else None
    s_key = request.session.session_key

    if request.method == "GET":
        qs = SavedTalk.objects.filter(user=user) if user else SavedTalk.objects.filter(session_key=s_key)
        service = get_recommender_service()
        items = []
        for b in qs:
            t = service.get_talk_by_id(b.talk_id)
            if t:
                items.append(t)
        return JsonResponse({"count": len(items), "saved_talks": items})

    elif request.method == "POST":
        try:
            body = json.loads(request.body.decode("utf-8")) if request.body else {}
        except json.JSONDecodeError:
            return JsonResponse({"error": "Malformed JSON payload"}, status=400)

        tid = body.get("talk_id")
        if not tid:
            return JsonResponse({"error": "talk_id required"}, status=400)

        service = get_recommender_service()
        talk = service.get_talk_by_id(int(tid))
        if not talk:
            return JsonResponse({"error": "Talk not found in catalog"}, status=404)

        if user:
            obj, created = SavedTalk.objects.get_or_create(
                user=user,
                talk_id=int(tid),
                defaults={
                    "title": talk["title"],
                    "speaker": talk["main_speaker"],
                    "tags": ", ".join(talk["tags"][:3]),
                },
            )
        else:
            if not s_key:
                request.session.create()
                s_key = request.session.session_key
            obj, created = SavedTalk.objects.get_or_create(
                session_key=s_key,
                talk_id=int(tid),
                defaults={
                    "title": talk["title"],
                    "speaker": talk["main_speaker"],
                    "tags": ", ".join(talk["tags"][:3]),
                },
            )

        return JsonResponse({"status": "saved", "talk_id": int(tid), "created": created}, status=201 if created else 200)

    elif request.method == "DELETE":
        if not talk_id:
            return JsonResponse({"error": "talk_id URL parameter required"}, status=400)

        qs = SavedTalk.objects.filter(user=user, talk_id=talk_id) if user else SavedTalk.objects.filter(session_key=s_key, talk_id=talk_id)
        deleted_count, _ = qs.delete()
        return JsonResponse({"status": "removed", "talk_id": talk_id, "deleted": deleted_count > 0})


@csrf_exempt
@require_POST
def api_feedback(request):
    """
    POST /api/feedback/
    Submit explicit feedback ('like', 'dislike', 'not_interested') for a recommendation.
    Accepts JSON payload or form-encoded POST.
    """
    try:
        if request.content_type == "application/json" and request.body:
            body = json.loads(request.body)
        else:
            body = request.POST
    except Exception:
        body = request.POST

    talk_id = body.get("talk_id")
    feedback_type = body.get("feedback_type", "like")
    reason = body.get("reason", "")

    if not talk_id:
        return JsonResponse({"error": "talk_id required"}, status=400)

    try:
        tid = int(talk_id)
    except (ValueError, TypeError):
        return JsonResponse({"error": "Invalid talk_id"}, status=400)

    user = request.user if request.user.is_authenticated else None
    s_key = request.session.session_key or ""
    if not user and not s_key:
        request.session.create()
        s_key = request.session.session_key

    fb = UserFeedback.objects.create(
        user=user,
        session_key=s_key,
        talk_id=tid,
        feedback_type=feedback_type,
        reason=reason,
    )
    return JsonResponse({
        "status": "success",
        "feedback_id": fb.id,
        "talk_id": tid,
        "feedback_type": feedback_type,
        "message": "Feedback recorded. Recommender profile updated."
    }, status=200)
