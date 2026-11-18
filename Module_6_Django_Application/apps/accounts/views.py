"""
Authentication, Registration, and Profile Management Views.
"""

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse

from apps.accounts.forms import UserRegistrationForm, UserProfileUpdateForm
from apps.accounts.models import UserProfile
from apps.talks.models import SavedTalk, Collection, ViewingHistory
from apps.recommendations.services import get_recommender_service

CURATED_ONBOARDING_TOPICS = [
    {"slug": "technology", "name": "Technology", "icon": "bi-cpu", "description": "AI, computing, robotics, and digital frontiers"},
    {"slug": "science", "name": "Science", "icon": "bi-radioactive", "description": "Physics, biology, space, and nature"},
    {"slug": "education", "name": "Education", "icon": "bi-mortarboard", "description": "Learning, pedagogy, and creative classrooms"},
    {"slug": "business", "name": "Business", "icon": "bi-briefcase", "description": "Leadership, entrepreneurship, and economics"},
    {"slug": "psychology", "name": "Psychology", "icon": "bi-heart-pulse", "description": "Human behavior, cognitive science, and empathy"},
    {"slug": "design", "name": "Design", "icon": "bi-palette", "description": "Architecture, UX, aesthetics, and creative problem solving"},
    {"slug": "environment", "name": "Environment", "icon": "bi-tree", "description": "Climate change, conservation, and sustainability"},
    {"slug": "culture", "name": "Culture & Society", "icon": "bi-globe", "description": "Anthropology, stories, and global perspectives"},
    {"slug": "health", "name": "Health & Medicine", "icon": "bi-activity", "description": "Wellness, neuroscience, and medical breakthroughs"},
    {"slug": "future", "name": "Future & Innovation", "icon": "bi-rocket-takeoff", "description": "Emerging trends and planetary scenarios"},
]


def _migrate_session_bookmarks(session_key, user, request=None):
    """
    Seamlessly migrates any anonymous session bookmarks into the user's permanent account upon login.
    """
    if session_key:
        session_bookmarks = SavedTalk.objects.filter(session_key=session_key, user__isnull=True)
        migrated_count = 0
        for b in session_bookmarks:
            # Check if user already bookmarked this talk
            if not SavedTalk.objects.filter(user=user, talk_id=b.talk_id).exists():
                b.user = user
                b.save(update_fields=["user"])
                migrated_count += 1
            else:
                b.delete()  # Avoid duplicates
        if migrated_count > 0 and request:
            messages.info(request, f"Synced {migrated_count} talk(s) from your guest session into your account.")


def register_view(request):
    """User account registration."""
    if request.user.is_authenticated:
        return redirect("talks:home")

    if request.method == "POST":
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            old_session_key = request.session.session_key
            user = form.save()
            login(request, user)
            _migrate_session_bookmarks(old_session_key, user, request)
            messages.success(request, f"Welcome to TEDMind, {user.username}! Choose your learning interests.")
            return redirect("accounts:onboarding")
    else:
        form = UserRegistrationForm()

    return render(request, "accounts/register.html", {"form": form})


def login_view(request):
    """User account login."""
    if request.user.is_authenticated:
        return redirect("talks:home")

    next_url = request.GET.get("next") or request.POST.get("next") or "talks:home"

    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            old_session_key = request.session.session_key
            user = form.get_user()
            login(request, user)
            _migrate_session_bookmarks(old_session_key, user, request)
            messages.success(request, f"Welcome back, {user.username}!")
            return redirect(next_url)
    else:
        form = AuthenticationForm(request)

    return render(request, "accounts/login.html", {"form": form, "next": next_url})


def logout_view(request):
    """User logout view."""
    if request.method == "POST" or request.GET:
        logout(request)
        messages.info(request, "You have been logged out.")
    return redirect("talks:home")


@login_required
def profile_view(request):
    """View and update profile, topic interests, and personalization settings."""
    profile = request.user.profile
    if request.method == "POST":
        form = UserProfileUpdateForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, "Your profile preferences have been updated.")
            return redirect("accounts:profile")
    else:
        form = UserProfileUpdateForm(instance=profile)

    # User activity statistics
    saved_count = SavedTalk.objects.filter(user=request.user).count()
    collections = Collection.objects.filter(user=request.user)
    viewed_count = ViewingHistory.objects.filter(user=request.user).count()

    context = {
        "profile": profile,
        "form": form,
        "saved_count": saved_count,
        "collections": collections,
        "viewed_count": viewed_count,
        "curated_topics": CURATED_ONBOARDING_TOPICS,
    }
    return render(request, "accounts/profile.html", context)


@login_required
def onboarding_view(request):
    """First-time or settings onboarding to select core learning interests."""
    profile = request.user.profile

    if request.method == "POST":
        selected = request.POST.getlist("topics")
        profile.set_preferred_topics_list(selected)
        profile.onboarding_completed = True
        profile.save()
        messages.success(request, "Your personalized learning profile is ready!")
        return redirect("talks:home")

    current_topics = set(profile.get_preferred_topics_list())
    context = {
        "curated_topics": CURATED_ONBOARDING_TOPICS,
        "current_topics": current_topics,
    }
    return render(request, "accounts/onboarding.html", context)


@login_required
def toggle_topic_interest_ajax(request):
    """AJAX endpoint to quickly toggle a topic interest in the user's profile."""
    if request.method != "POST":
        return JsonResponse({"error": "POST method required"}, status=405)

    topic = request.POST.get("topic", "").strip().lower()
    if not topic:
        return JsonResponse({"error": "Topic required"}, status=400)

    profile = request.user.profile
    current = profile.get_preferred_topics_list()
    if topic in current:
        current.remove(topic)
        action = "removed"
    else:
        current.append(topic)
        action = "added"

    profile.set_preferred_topics_list(current)
    profile.save(update_fields=["preferred_topics"])

    return JsonResponse({
        "status": "ok",
        "action": action,
        "topic": topic,
        "topics_count": len(profile.preferred_topics),
    })
