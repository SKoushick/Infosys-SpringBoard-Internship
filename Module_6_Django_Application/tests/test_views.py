"""
Automated unit and integration tests for Module 6 Django Frontend.
Uses Django's standard TestCase test runner for zero-dependency execution.
"""

from django.test import TestCase, Client
from django.urls import reverse

from apps.talks.models import SavedTalk
from apps.recommendations.services import get_recommender_service


class TEDMindViewTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.service = get_recommender_service()
        assert cls.service.is_initialized is True

    def setUp(self):
        self.client = Client()

    def test_home_page_loads(self):
        """Verify home page loads with 200, TEDMind branding, and featured talks."""
        url = reverse("talks:home")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")
        self.assertIn("TEDMind", content)
        self.assertIn("Featured TED Presentations", content)
        self.assertIn("Multi-Field TF-IDF", content)
        self.assertIn("Sentence Transformers", content)

    def test_explore_page_and_search(self):
        """Verify explore page lists talks and search filtering works."""
        url = reverse("talks:explore")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")
        self.assertIn("Explore TED Talks Catalog", content)
        self.assertIn("talks found", content)

        # Test search for specific title
        response_search = self.client.get(url, {"q": "schools kill creativity"})
        self.assertEqual(response_search.status_code, 200)
        search_content = response_search.content.decode("utf-8")
        self.assertIn("Ken Robinson", search_content)

    def test_talk_detail_page(self):
        """Verify talk details page displays metadata, tags, and similar recommendations."""
        url = reverse("talks:detail", kwargs={"talk_id": 1})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")
        self.assertIn("Ken Robinson", content)
        self.assertIn("Total Views", content)
        self.assertIn("Recommend Similar Talks", content)
        self.assertIn("Recommended Companion Talks", content)

    def test_recommendation_form_loads(self):
        """Verify recommendation input page loads with topic tags and model selection."""
        url = reverse("recommendations:form")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")
        self.assertIn("Generate Your Personalized TED Recommendations", content)
        self.assertIn("Method A", content)
        self.assertIn("Method B", content)
        self.assertIn("Method C", content)
        self.assertIn("Multi-Signal Hybrid", content)

    def test_recommendation_results_seed_talk(self):
        """Verify real recommendations generated for a seed talk with explanations."""
        url = reverse("recommendations:results")
        response = self.client.post(url, {
            "seed_talk_id": 1,
            "method": "hybrid",
            "top_k": 5,
            "apply_mmr": "1",
        })
        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")
        self.assertIn("Top 5 Recommended Talks", content)
        self.assertIn("Recommendation Rationale", content)
        self.assertIn("Match", content)
        self.assertIn("Watch on TED.com", content)

    def test_recommendation_results_free_text(self):
        """Verify real recommendations generated for a free-text prompt."""
        url = reverse("recommendations:results")
        response = self.client.post(url, {
            "query_text": "artificial intelligence and machine learning robotics",
            "method": "hybrid",
            "top_k": 5,
            "apply_mmr": "1",
        })
        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")
        self.assertIn("Top 5 Recommended Talks", content)
        self.assertIn("Recommendation Rationale", content)

    def test_recommendation_empty_input_redirects(self):
        """Verify submitting an empty form redirects back with warning message."""
        url = reverse("recommendations:results")
        response = self.client.post(url, {
            "seed_talk_id": "",
            "query_text": "",
            "interests": "",
        })
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("recommendations:form"))

    def test_toggle_save_talk(self):
        """Verify saving/bookmarking a talk in session works."""
        url = reverse("talks:toggle_save", kwargs={"talk_id": 1})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(SavedTalk.objects.filter(talk_id=1).count(), 1)

        # Toggle again should remove
        response_remove = self.client.get(url)
        self.assertEqual(response_remove.status_code, 302)
        self.assertEqual(SavedTalk.objects.filter(talk_id=1).count(), 0)

    def test_about_page_loads(self):
        """Verify about page loads with problem statement and evaluation scorecard."""
        url = reverse("talks:about")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")
        self.assertIn("Personalized TED Talks Recommendation System", content)
        self.assertIn("Problem Statement", content)
        self.assertIn("Module 5: Empirical Benchmark Scorecard", content)
        self.assertIn("Research Integrity & Collaborative Filtering", content)
