"""
Context processor providing global TEDMind metadata across all templates.
"""
from apps.talks.models import SavedTalk
from apps.recommendations.services import get_recommender_service


def tedmind_global_context(request):
    """Adds saved talks count and top categories to all templates."""
    saved_count = 0
    saved_talk_ids = set()

    if request.session.session_key:
        s_key = request.session.session_key
        saved_qs = SavedTalk.objects.filter(session_key=s_key)
        saved_count = saved_qs.count()
        saved_talk_ids = set(saved_qs.values_list("talk_id", flat=True))

    service = get_recommender_service()
    popular_tags = [t[0] for t in service.top_tags[:12]] if service.is_initialized else []

    return {
        "saved_talks_count": saved_count,
        "saved_talk_ids": saved_talk_ids,
        "global_popular_tags": popular_tags,
        "app_name": "TEDMind",
        "app_tagline": "Discover ideas worth spreading, personalized around your interests.",
    }
