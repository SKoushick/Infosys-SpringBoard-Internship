"""
Views for generating and displaying TED Talk recommendations.
"""

import time
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages

from apps.recommendations.services import get_recommender_service
from apps.talks.models import SavedTalk, UserFeedback, RecommendationEvent


CURATED_INTEREST_TAGS = [
    "technology",
    "science",
    "artificial intelligence",
    "education",
    "business",
    "leadership",
    "psychology",
    "health",
    "innovation",
    "environment",
    "creativity",
    "design",
    "future",
    "biology",
    "culture",
    "economics",
]


def recommend_form_view(request):
    """
    Page 3: Recommendation Input Interface.
    Allows user to select a seed talk, choose interests, or enter free-text interest.
    """
    service = get_recommender_service()

    # Pre-select seed talk if passed in query string (e.g. from talk detail "Recommend Similar")
    preselected_id = request.GET.get("talk_id")
    preselected_talk = None
    if preselected_id:
        try:
            preselected_talk = service.get_talk_by_id(int(preselected_id))
        except (ValueError, TypeError):
            pass

    # Check saved talks for user or session personalization
    saved_talks = []
    if request.user.is_authenticated:
        saved_records = SavedTalk.objects.filter(user=request.user)
    elif request.session.session_key:
        saved_records = SavedTalk.objects.filter(session_key=request.session.session_key)
    else:
        saved_records = []

    for item in saved_records:
        t_data = service.get_talk_by_id(item.talk_id)
        if t_data:
            saved_talks.append(t_data)

    context = {
        "curated_tags": CURATED_INTEREST_TAGS,
        "preselected_talk": preselected_talk,
        "preselected_id": preselected_id,
        "saved_talks": saved_talks,
        "saved_count": len(saved_talks),
    }
    return render(request, "recommendations/recommend_form.html", context)


def recommend_results_view(request):
    """
    Page 4: Recommendation Results.
    Executes the real ML recommendation engine and displays ranked results with explanations.
    """
    service = get_recommender_service()

    # Extract inputs from GET or POST
    data = request.POST if request.method == "POST" else request.GET

    seed_talk_id_raw = data.get("seed_talk_id", "").strip()
    query_text = data.get("query_text", "").strip()
    raw_interests = data.getlist("interests") if hasattr(data, "getlist") else data.get("interests", [])
    if isinstance(raw_interests, str):
        selected_tags = [t.strip() for t in raw_interests.split(",") if t.strip()]
    elif isinstance(raw_interests, (list, tuple)):
        selected_tags = [t.strip() for t in raw_interests if isinstance(t, str) and t.strip()]
    else:
        selected_tags = []

    use_saved_profile = data.get("use_saved_profile") == "1"
    method = data.get("method", "hybrid").lower().strip()
    if method not in ("tfidf", "semantic", "hybrid"):
        method = "hybrid"

    try:
        top_k = int(data.get("top_k", 10))
    except (ValueError, TypeError):
        top_k = 10
    top_k = max(1, min(20, top_k))

    apply_mmr = data.get("apply_mmr") in ("1", "true", "on", True)

    seed_talk_id = None
    if seed_talk_id_raw:
        try:
            seed_talk_id = int(seed_talk_id_raw)
        except (ValueError, TypeError):
            seed_talk_id = None

    # Handle saved talks and preferences for personalization
    liked_talk_ids = []
    if use_saved_profile:
        if request.user.is_authenticated:
            saved_records = SavedTalk.objects.filter(user=request.user)
            liked_talk_ids = [r.talk_id for r in saved_records]
            if not selected_tags and hasattr(request.user, "profile"):
                selected_tags = request.user.profile.get_preferred_topics_list()
        elif request.session.session_key:
            saved_records = SavedTalk.objects.filter(session_key=request.session.session_key)
            liked_talk_ids = [r.talk_id for r in saved_records]

    # Handle negative feedback to exclude or down-weight talks
    disliked_talk_ids = []
    if request.user.is_authenticated:
        disliked_records = UserFeedback.objects.filter(
            user=request.user, feedback_type__in=["dislike", "not_interested"]
        )
        disliked_talk_ids = [r.talk_id for r in disliked_records]
    elif request.session.session_key:
        disliked_records = UserFeedback.objects.filter(
            session_key=request.session.session_key, feedback_type__in=["dislike", "not_interested"]
        )
        disliked_talk_ids = [r.talk_id for r in disliked_records]

    # Validate that at least one valid input signal is provided
    if not seed_talk_id and not query_text and not selected_tags and not liked_talk_ids:
        messages.warning(
            request,
            "Please provide at least one recommendation input: select a talk, choose interest tags, or describe what you want to learn.",
        )
        return redirect("recommendations:form")

    # Time execution for live latency display
    t0 = time.perf_counter()

    seed_talk = None
    if seed_talk_id:
        seed_talk = service.get_talk_by_id(seed_talk_id)
        if not seed_talk:
            messages.error(request, f"Selected talk ID {seed_talk_id} not found in catalog.")
            return redirect("recommendations:form")

        recommendations = service.recommend_for_talk(
            seed_talk_id=seed_talk_id,
            method=method,
            top_k=top_k,
            apply_mmr=apply_mmr,
            exclude_talk_ids=set(disliked_talk_ids) if disliked_talk_ids else None,
        )
    else:
        recommendations = service.recommend_by_interests(
            user_interests=selected_tags,
            query_text=query_text,
            liked_talk_ids=liked_talk_ids if liked_talk_ids else None,
            disliked_talk_ids=disliked_talk_ids if disliked_talk_ids else None,
            method=method,
            top_k=top_k,
            apply_mmr=apply_mmr,
        )

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    # Audit logging for recommendation event
    try:
        user_obj = request.user if request.user.is_authenticated else None
        s_key = request.session.session_key or ""
        rec_ids = [r["talk_id"] for r in recommendations]
        RecommendationEvent.objects.create(
            user=user_obj,
            session_key=s_key,
            seed_talk_id=seed_talk_id,
            query_text=query_text,
            method=method,
            recommended_talk_ids=rec_ids,
            latency_ms=latency_ms,
        )
    except Exception:
        pass

    method_labels = {
        "hybrid": "Multi-Signal Hybrid (TF-IDF + Semantic)",
        "tfidf": "TF-IDF Lexical Matcher",
        "semantic": "Sentence-BERT Semantic Embeddings",
    }

    context = {
        "recommendations": recommendations,
        "recommendations_count": len(recommendations),
        "seed_talk": seed_talk,
        "query_text": query_text,
        "selected_tags": selected_tags,
        "used_saved_profile": use_saved_profile,
        "liked_talk_ids_count": len(liked_talk_ids),
        "method": method,
        "method_display": method_labels.get(method, "Hybrid"),
        "top_k": top_k,
        "apply_mmr": apply_mmr,
        "latency_ms": latency_ms,
    }
    return render(request, "recommendations/results.html", context)


def api_search_titles_view(request):
    """JSON API endpoint for autocomplete title search in recommendation forms."""
    query = request.GET.get("q", "").strip().lower()
    if not query:
        return JsonResponse({"results": []})

    service = get_recommender_service()
    matches = []
    for tid, t in service.id_to_talk.items():
        if query in t["title"].lower() or query in t["main_speaker"].lower():
            matches.append({
                "id": tid,
                "title": t["title"],
                "speaker": t["main_speaker"],
                "views": t["views_formatted"],
            })
            if len(matches) >= 15:
                break

    return JsonResponse({"results": matches})
