"""
Views for TED Talks exploration, details, and saved talks.
"""

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, Http404
from django.contrib import messages
from django.core.paginator import Paginator

from apps.recommendations.services import get_recommender_service
from apps.talks.models import SavedTalk


def home_view(request):
    """Landing dashboard page."""
    service = get_recommender_service()
    featured_talks = service.get_featured_talks(limit=6)
    popular_talks = service.get_popular_talks(limit=6)
    top_tags = service.top_tags[:16]

    context = {
        "featured_talks": featured_talks,
        "popular_talks": popular_talks,
        "top_tags": top_tags,
    }
    return render(request, "talks/home.html", context)


def explore_view(request):
    """Explore and search TED talks with filtering and pagination."""
    service = get_recommender_service()

    query = request.GET.get("q", "").strip()
    tag = request.GET.get("tag", "").strip()
    speaker = request.GET.get("speaker", "").strip()

    try:
        page = max(1, int(request.GET.get("page", 1)))
    except (ValueError, TypeError):
        page = 1

    per_page = 12
    talks, total_count = service.search_talks(
        query=query, tag=tag, speaker=speaker, page=page, per_page=per_page
    )

    # Calculate pagination metadata
    total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1
    has_prev = page > 1
    has_next = page < total_pages

    context = {
        "talks": talks,
        "total_count": total_count,
        "current_page": page,
        "total_pages": total_pages,
        "has_prev": has_prev,
        "has_next": has_next,
        "prev_page": page - 1,
        "next_page": page + 1,
        "query": query,
        "selected_tag": tag,
        "selected_speaker": speaker,
        "top_tags": service.top_tags[:20],
    }
    return render(request, "talks/explore.html", context)


def talk_detail_view(request, talk_id: int):
    """Display talk metadata and quick similar recommendations."""
    service = get_recommender_service()
    talk = service.get_talk_by_id(talk_id)
    if not talk:
        raise Http404(f"TED Talk with ID {talk_id} not found.")

    # Generate 3 similar companion talks for instant preview
    similar_talks = service.recommend_for_talk(
        seed_talk_id=talk_id,
        method="hybrid",
        top_k=3,
        apply_mmr=True,
    )

    # Check if currently saved in session
    is_saved = False
    if request.session.session_key:
        is_saved = SavedTalk.objects.filter(
            session_key=request.session.session_key, talk_id=talk_id
        ).exists()

    context = {
        "talk": talk,
        "similar_talks": similar_talks,
        "is_saved": is_saved,
    }
    return render(request, "talks/talk_detail.html", context)


def toggle_save_talk_view(request, talk_id: int):
    """Bookmark / save talk for current anonymous browser session."""
    if not request.session.session_key:
        request.session.create()

    session_key = request.session.session_key
    service = get_recommender_service()
    talk = service.get_talk_by_id(talk_id)

    if not talk:
        messages.error(request, "Talk not found.")
        return redirect("talks:explore")

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
        messages.success(request, f'Saved "{talk["title"]}" to your session profile.')

    if request.headers.get("x-requested-with") == "XMLHttpRequest":
        return JsonResponse({"status": "ok", "action": action, "talk_id": talk_id})

    # Return to previous page or talk detail
    next_url = request.META.get("HTTP_REFERER") or f"/talks/{talk_id}/"
    return redirect(next_url)


def saved_talks_view(request):
    """View saved talks and trigger multi-talk personalized recommendations."""
    if not request.session.session_key:
        request.session.create()

    s_key = request.session.session_key
    saved_records = SavedTalk.objects.filter(session_key=s_key)

    service = get_recommender_service()
    saved_talks = []
    for item in saved_records:
        t_data = service.get_talk_by_id(item.talk_id)
        if t_data:
            saved_talks.append(t_data)

    context = {
        "saved_talks": saved_talks,
        "saved_count": len(saved_talks),
    }
    return render(request, "talks/saved_talks.html", context)


def about_view(request):
    """About page detailing problem, architecture, metrics, and technologies."""
    return render(request, "talks/about.html")
