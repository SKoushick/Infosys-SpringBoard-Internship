"""
Views for TEDMind Talks Application.
Manages dashboard, catalogue exploration, talk details, collections, and bookmarks.
"""

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_POST

from apps.talks.models import (
    SavedTalk,
    Collection,
    CollectionItem,
    ViewingHistory,
    UserFeedback,
)
from apps.recommendations.services import get_recommender_service


def home_view(request):
    """
    Page 1: Personalized Home Dashboard.
    Displays hero section, search entry, trending topics, and featured talks.
    """
    service = get_recommender_service()
    featured_talks = service.get_featured_talks(limit=6)
    popular_talks = service.get_popular_talks(limit=6)

    # If user is authenticated and has preferred topics, provide personalized teaser
    personalized_talks = []
    user_topics = []
    if request.user.is_authenticated and hasattr(request.user, "profile"):
        user_topics = request.user.profile.get_preferred_topics_list()
        if user_topics:
            personalized_talks = service.recommend_by_interests(
                user_interests=user_topics,
                method="hybrid",
                top_k=6,
                apply_mmr=True,
            )

    # Recently viewed talks
    recent_viewed = []
    history_qs = ViewingHistory.objects.none()
    if request.user.is_authenticated:
        history_qs = ViewingHistory.objects.filter(user=request.user)
    elif request.session.session_key:
        history_qs = ViewingHistory.objects.filter(session_key=request.session.session_key)

    seen_ids = set()
    for item in history_qs.order_by("-viewed_at")[:10]:
        if item.talk_id not in seen_ids:
            seen_ids.add(item.talk_id)
            t_obj = service.get_talk_by_id(item.talk_id)
            if t_obj:
                recent_viewed.append(t_obj)
            if len(recent_viewed) >= 4:
                break

    context = {
        "featured_talks": featured_talks,
        "popular_talks": popular_talks,
        "personalized_talks": personalized_talks,
        "recent_viewed": recent_viewed,
        "user_topics": user_topics,
        "total_talks_count": len(service.id_to_talk),
    }
    return render(request, "talks/home.html", context)


def explore_view(request):
    """
    Page 2: Explore TED Talks Catalog.
    Provides multi-parameter search, topic filtering, speaker search, and pagination.
    """
    service = get_recommender_service()

    query = request.GET.get("q", "").strip()
    tag = request.GET.get("tag", "").strip()
    speaker = request.GET.get("speaker", "").strip()

    try:
        page = int(request.GET.get("page", 1))
    except (ValueError, TypeError):
        page = 1

    per_page = 12
    talks_page, total_count = service.search_talks(
        query=query,
        tag=tag,
        speaker=speaker,
        page=page,
        per_page=per_page,
    )

    total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1

    context = {
        "talks": talks_page,
        "total_count": total_count,
        "page": page,
        "total_pages": total_pages,
        "has_previous": page > 1,
        "has_next": page < total_pages,
        "previous_page_number": page - 1,
        "next_page_number": page + 1,
        "query": query,
        "selected_tag": tag,
        "selected_speaker": speaker,
    }
    return render(request, "talks/explore.html", context)


def talk_detail_view(request, talk_id: int):
    """
    Page 5: Talk Details.
    Displays rich talk metadata, external link to TED, and 3 companion recommendations.
    """
    service = get_recommender_service()
    talk = service.get_talk_by_id(talk_id)

    if not talk:
        messages.error(request, f"TED Talk with ID #{talk_id} was not found.")
        return redirect("talks:explore")

    # Record viewing history
    try:
        user = request.user if request.user.is_authenticated else None
        s_key = request.session.session_key or ""
        ViewingHistory.objects.create(user=user, session_key=s_key, talk_id=talk_id)
    except Exception:
        pass

    # Instant companion recommendations
    similar_talks = service.recommend_for_talk(
        seed_talk_id=talk_id,
        method="hybrid",
        top_k=3,
        apply_mmr=True,
    )

    # Check if currently saved
    is_saved = False
    if request.user.is_authenticated:
        is_saved = SavedTalk.objects.filter(user=request.user, talk_id=talk_id).exists()
    elif request.session.session_key:
        is_saved = SavedTalk.objects.filter(
            session_key=request.session.session_key, talk_id=talk_id
        ).exists()

    # User collections for "Add to Collection" dropdown
    user_collections = []
    if request.user.is_authenticated:
        user_collections = Collection.objects.filter(user=request.user)

    # Transcript
    transcript = service.get_talk_transcript(talk_id)

    context = {
        "talk": talk,
        "transcript": transcript,
        "similar_talks": similar_talks,
        "is_saved": is_saved,
        "user_collections": user_collections,
    }
    return render(request, "talks/talk_detail.html", context)


def toggle_save_talk_view(request, talk_id: int):
    """
    Bookmark / save talk for authenticated user or anonymous browser session.
    Accepts both POST (recommended, CSRF-safe) and GET (legacy fallback).
    """
    user = request.user if request.user.is_authenticated else None
    if not user and not request.session.session_key:
        request.session.create()
    session_key = request.session.session_key

    service = get_recommender_service()
    talk = service.get_talk_by_id(talk_id)

    if not talk:
        messages.error(request, "Talk not found.")
        return redirect("talks:explore")

    if user:
        saved_obj, created = SavedTalk.objects.get_or_create(
            user=user,
            talk_id=talk_id,
            defaults={
                "title": talk["title"],
                "speaker": talk["main_speaker"],
                "tags": ", ".join(talk["tags"][:3]),
                "is_liked": True,
            },
        )
    else:
        saved_obj, created = SavedTalk.objects.get_or_create(
            session_key=session_key,
            talk_id=talk_id,
            defaults={
                "title": talk["title"],
                "speaker": talk["main_speaker"],
                "tags": ", ".join(talk["tags"][:3]),
                "is_liked": True,
            },
        )

    if not created:
        saved_obj.delete()
        action = "removed"
        messages.info(request, f'Removed "{talk["title"]}" from your saved talks.')
    else:
        action = "added"
        messages.success(request, f'Saved "{talk["title"]}" to your profile.')

    if request.headers.get("x-requested-with") == "XMLHttpRequest":
        return JsonResponse({"status": "ok", "action": action, "talk_id": talk_id})

    # Return to previous page or talk detail
    next_url = request.META.get("HTTP_REFERER") or f"/talks/{talk_id}/"
    return redirect(next_url)


def saved_talks_view(request):
    """View saved talks and trigger multi-talk personalized recommendations."""
    service = get_recommender_service()

    if request.user.is_authenticated:
        saved_records = SavedTalk.objects.filter(user=request.user)
    else:
        if not request.session.session_key:
            request.session.create()
        s_key = request.session.session_key
        saved_records = SavedTalk.objects.filter(session_key=s_key)

    saved_talks = []
    for item in saved_records:
        t_data = service.get_talk_by_id(item.talk_id)
        if t_data:
            saved_talks.append(t_data)

    collections = []
    if request.user.is_authenticated:
        collections = Collection.objects.filter(user=request.user)

    context = {
        "saved_talks": saved_talks,
        "saved_count": len(saved_talks),
        "collections": collections,
    }
    return render(request, "talks/saved_talks.html", context)


@login_required
def create_collection_view(request):
    """Create a new custom collection / playlist."""
    if request.method == "POST":
        title = request.POST.get("title", "").strip()
        description = request.POST.get("description", "").strip()
        is_public = request.POST.get("is_public") == "1"

        if not title:
            messages.error(request, "Please provide a collection title.")
            return redirect("talks:saved")

        collection, created = Collection.objects.get_or_create(
            user=request.user,
            title=title,
            defaults={"description": description, "is_public": is_public},
        )
        if created:
            messages.success(request, f'Collection "{title}" created successfully.')
        else:
            messages.warning(request, f'A collection named "{title}" already exists.')

    return redirect("talks:saved")


@login_required
def collection_detail_view(request, collection_id: int):
    """View a single collection with its saved talk items."""
    collection = get_object_or_404(Collection, id=collection_id, user=request.user)
    service = get_recommender_service()

    items_data = []
    for item in collection.items.all():
        t_data = service.get_talk_by_id(item.talk_id)
        if t_data:
            items_data.append({"item": item, "talk": t_data})

    return render(
        request,
        "talks/collection_detail.html",
        {"collection": collection, "items_data": items_data},
    )


@login_required
@require_POST
def add_to_collection_ajax(request):
    """AJAX endpoint to add a talk to a collection."""
    collection_id = request.POST.get("collection_id")
    talk_id = request.POST.get("talk_id")

    if not collection_id or not talk_id:
        return JsonResponse({"error": "Missing collection_id or talk_id"}, status=400)

    collection = get_object_or_404(Collection, id=collection_id, user=request.user)
    item, created = CollectionItem.objects.get_or_create(
        collection=collection,
        talk_id=int(talk_id),
    )
    if created:
        msg = f"Added talk to '{collection.title}'."
    else:
        msg = f"Talk already in '{collection.title}'."

    return JsonResponse({"status": "ok", "message": msg, "created": created})


@require_POST
def record_feedback_ajax(request):
    """AJAX endpoint to record user feedback on recommended talks."""
    talk_id = request.POST.get("talk_id")
    feedback_type = request.POST.get("feedback_type", "like")
    reason = request.POST.get("reason", "").strip()

    if not talk_id:
        return JsonResponse({"error": "talk_id required"}, status=400)

    user = request.user if request.user.is_authenticated else None
    s_key = request.session.session_key or ""

    UserFeedback.objects.create(
        user=user,
        session_key=s_key,
        talk_id=int(talk_id),
        feedback_type=feedback_type,
        reason=reason,
    )

    return JsonResponse({"status": "ok", "message": "Feedback recorded."})


def about_view(request):
    """About page detailing problem, architecture, metrics, and technologies."""
    return render(request, "talks/about.html")
