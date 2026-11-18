"""
Automated unit and integration tests for TEDMind REST API endpoints.
"""

import json
from django.contrib.auth import get_user_model
from django.test import TestCase, Client
from django.urls import reverse

from apps.talks.models import SavedTalk
from apps.recommendations.services import get_recommender_service

User = get_user_model()


class TEDMindApiTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.service = get_recommender_service()
        assert cls.service.is_initialized is True

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username="api_tester",
            email="api@example.com",
            password="SecurePassword2026!",
        )

    def test_api_health_check(self):
        """Verify GET /api/health/ returns 200 with catalog and model status."""
        url = reverse("api:health")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertGreater(data["catalog_size"], 2000)
        self.assertTrue(data["models_loaded"]["tfidf_recommender"])
        self.assertTrue(data["models_loaded"]["semantic_recommender"])
        self.assertTrue(data["models_loaded"]["hybrid_recommender"])

    def test_api_talks_list(self):
        """Verify GET /api/talks/ returns paginated talk catalog."""
        url = reverse("api:talks_list")
        response = self.client.get(url, {"per_page": 5})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["results"]), 5)
        self.assertGreater(data["total_count"], 2500)

    def test_api_talks_search(self):
        """Verify GET /api/talks/?q=creativity filters talks."""
        url = reverse("api:talks_list")
        response = self.client.get(url, {"q": "creativity", "per_page": 5})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreater(data["total_count"], 0)

    def test_api_talk_detail(self):
        """Verify GET /api/talks/<id>/ returns talk details."""
        url = reverse("api:talk_detail", kwargs={"talk_id": 1})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["talk_id"], 1)
        self.assertEqual(data["main_speaker"], "Ken Robinson")

    def test_api_talk_similar(self):
        """Verify GET /api/talks/<id>/similar/ returns recommendations."""
        url = reverse("api:talk_similar", kwargs={"talk_id": 1})
        response = self.client.get(url, {"top_k": 3})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["recommendations"]), 3)

    def test_api_generate_recommendations(self):
        """Verify POST /api/recommendations/generate/ with JSON payload."""
        url = reverse("api:generate_recommendations")
        payload = {
            "seed_talk_id": 1,
            "method": "hybrid",
            "top_k": 5,
            "apply_mmr": True,
        }
        response = self.client.post(
            url,
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(len(data["recommendations"]), 5)
        first = data["recommendations"][0]
        self.assertIn("score", first)
        self.assertIn("explanation", first)

    def test_api_user_saved_flow(self):
        """Verify saving and deleting talks via API."""
        self.client.login(username="api_tester", password="SecurePassword2026!")

        # Save talk
        save_url = reverse("api:user_saved")
        resp_post = self.client.post(
            save_url,
            data=json.dumps({"talk_id": 1}),
            content_type="application/json",
        )
        self.assertEqual(resp_post.status_code, 201)
        self.assertTrue(SavedTalk.objects.filter(user=self.user, talk_id=1).exists())

        # List saved
        resp_get = self.client.get(save_url)
        self.assertEqual(resp_get.status_code, 200)
        self.assertEqual(resp_get.json()["count"], 1)

        # Delete saved
        del_url = reverse("api:user_saved_delete", kwargs={"talk_id": 1})
        resp_del = self.client.delete(del_url)
        self.assertEqual(resp_del.status_code, 200)
        self.assertFalse(SavedTalk.objects.filter(user=self.user, talk_id=1).exists())

    def test_api_feedback(self):
        """Verify POST /api/feedback/ records user feedback."""
        url = reverse("api:feedback")
        payload = {
            "talk_id": 1,
            "feedback_type": "like",
            "reason": "Inspiring perspective on creativity"
        }
        response = self.client.post(
            url,
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["talk_id"], 1)
        self.assertEqual(data["feedback_type"], "like")
