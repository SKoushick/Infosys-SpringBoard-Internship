"""
Automated unit & integration tests for Accounts, User Profiles, Collections, and Personalization.
"""

from django.contrib.auth import get_user_model
from django.test import TestCase, Client
from django.urls import reverse

from apps.accounts.models import UserProfile
from apps.talks.models import SavedTalk, Collection, CollectionItem, UserFeedback

User = get_user_model()


class AccountsAndPersonalizationTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user_password = "SecurePassword123!"
        self.user = User.objects.create_user(
            username="alex_learner",
            email="alex@example.com",
            password=self.user_password,
        )

    def test_user_profile_automatically_created(self):
        """Verify UserProfile signal automatically creates profile upon User creation."""
        self.assertTrue(hasattr(self.user, "profile"))
        self.assertEqual(self.user.profile.preferred_topics, [])
        self.assertTrue(self.user.profile.recommendations_enabled)

    def test_user_registration_success(self):
        """Verify user registration creates account and profile."""
        url = reverse("accounts:register")
        response = self.client.post(url, {
            "username": "sarah_explorer",
            "email": "sarah@example.com",
            "password": "StrongPassword2026!",
            "confirm_password": "StrongPassword2026!",
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(User.objects.filter(username="sarah_explorer").exists())
        sarah = User.objects.get(username="sarah_explorer")
        self.assertTrue(hasattr(sarah, "profile"))

    def test_registration_password_mismatch(self):
        """Verify registration rejects mismatched passwords."""
        url = reverse("accounts:register")
        response = self.client.post(url, {
            "username": "mismatch_user",
            "email": "mismatch@example.com",
            "password": "Password1!",
            "confirm_password": "Password2!",
        })
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username="mismatch_user").exists())

    def test_login_and_session_bookmark_migration(self):
        """Verify anonymous session bookmarks migrate into user account upon login."""
        # 1. Save talk as anonymous guest session
        self.client.get(reverse("talks:home"))  # Establish session
        session_key = self.client.session.session_key
        SavedTalk.objects.create(
            session_key=session_key,
            talk_id=42,
            title="Guest Saved Talk",
            speaker="Guest Speaker",
        )
        self.assertEqual(SavedTalk.objects.filter(user__isnull=True, talk_id=42).count(), 1)

        # 2. Login as existing user
        login_url = reverse("accounts:login")
        response = self.client.post(login_url, {
            "username": "alex_learner",
            "password": self.user_password,
        })
        self.assertEqual(response.status_code, 302)

        # 3. Verify talk is now linked to user
        user_bookmarks = SavedTalk.objects.filter(user=self.user, talk_id=42)
        self.assertEqual(user_bookmarks.count(), 1)

    def test_onboarding_topic_selection(self):
        """Verify user can select onboarding interests and preferences persist."""
        self.client.login(username="alex_learner", password=self.user_password)
        url = reverse("accounts:onboarding")
        response = self.client.post(url, {
            "topics": ["technology", "education", "design"],
        })
        self.assertEqual(response.status_code, 302)

        self.user.refresh_from_db()
        self.assertEqual(
            sorted(self.user.profile.preferred_topics),
            ["design", "education", "technology"],
        )
        self.assertTrue(self.user.profile.onboarding_completed)

    def test_collection_creation_and_item_addition(self):
        """Verify authenticated user can create a collection and add talk items."""
        self.client.login(username="alex_learner", password=self.user_password)

        # Create collection
        create_url = reverse("talks:create_collection")
        resp_create = self.client.post(create_url, {
            "title": "Inspiring AI Breakthroughs",
            "description": "My curated collection of talks on intelligence and ethics.",
        })
        self.assertEqual(resp_create.status_code, 302)

        col = Collection.objects.get(user=self.user, title="Inspiring AI Breakthroughs")
        self.assertEqual(col.item_count, 0)

        # Add talk item via AJAX
        add_url = reverse("talks:add_to_collection_ajax")
        resp_add = self.client.post(add_url, {
            "collection_id": col.id,
            "talk_id": 1,
        })
        self.assertEqual(resp_add.status_code, 200)
        self.assertEqual(col.item_count, 1)

    def test_record_feedback_ajax(self):
        """Verify recording like/dislike user feedback."""
        self.client.login(username="alex_learner", password=self.user_password)
        url = reverse("talks:record_feedback_ajax")
        response = self.client.post(url, {
            "talk_id": 7,
            "feedback_type": "dislike",
            "reason": "Not relevant to my current focus",
        })
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            UserFeedback.objects.filter(
                user=self.user, talk_id=7, feedback_type="dislike"
            ).exists()
        )
