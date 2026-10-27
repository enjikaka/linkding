from unittest.mock import Mock, patch

from django.test import TestCase, override_settings

from bookmarks.services import jev


class JevEncryptionTestCase(TestCase):
    def test_encrypt_decrypt_roundtrip(self):
        api_key = "sk-or-v1-1234567890"
        encrypted = jev.encrypt_api_key(api_key)

        self.assertNotEqual(encrypted, api_key)
        self.assertEqual(jev.decrypt_api_key(encrypted), api_key)

    def test_encrypt_empty_key(self):
        self.assertEqual(jev.encrypt_api_key(""), "")

    def test_decrypt_invalid_value_returns_empty_string(self):
        self.assertEqual(jev.decrypt_api_key("not-an-encrypted-key"), "")

    def test_decrypt_empty_string(self):
        self.assertEqual(jev.decrypt_api_key(""), "")


class JevBuildQuestionsTestCase(TestCase):
    def test_builds_noul_question_per_tag(self):
        questions = jev.build_questions(["kaffe", "kök"])

        self.assertEqual(
            questions,
            {
                "kaffe": {
                    "type": "noul",
                    "instructions": 'Is the bookmarked page about "kaffe"?',
                },
                "kök": {
                    "type": "noul",
                    "instructions": 'Is the bookmarked page about "kök"?',
                },
            },
        )


def mock_response(answers):
    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {"answers": answers}
    return response


class JevGetRecommendationsTestCase(TestCase):
    def setUp(self):
        self.url = "https://example.com/awesome-post"
        self.title = "Awesome Post"
        self.description = "A post about coffee and kitchens"
        self.tag_names = ["kaffe", "kök", "böcker"]
        self.api_key = "sk-or-v1-test-key"

    @override_settings(LD_OPENROUTER_MODEL="typesafe/jev-1.13")
    def test_returns_tags_sorted_by_score_filtered_by_threshold(self):
        with patch("bookmarks.services.jev.requests.post") as mock_post:
            mock_post.return_value = mock_response(
                {
                    "kaffe": {"type": "noul", "noul": 0.95},
                    "kök": {"type": "noul", "noul": 0.4},
                    "böcker": {"type": "noul", "noul": 0.72},
                }
            )
            recommendations = jev.get_recommendations(
                self.url, self.title, self.description, self.tag_names, self.api_key
            )

        self.assertEqual(
            recommendations,
            [
                {"name": "kaffe", "score": 0.95},
                {"name": "böcker", "score": 0.72},
            ],
        )

    def test_sends_decisions_request_with_questions(self):
        with patch("bookmarks.services.jev.requests.post") as mock_post:
            mock_post.return_value = mock_response({})
            jev.get_recommendations(
                self.url, self.title, self.description, self.tag_names, self.api_key
            )

        mock_post.assert_called_once()
        request = mock_post.call_args
        self.assertEqual(request.args[0], jev.OPENROUTER_DECISIONS_URL)
        self.assertEqual(
            request.kwargs["headers"],
            {"Authorization": f"Bearer {self.api_key}"},
        )
        payload = request.kwargs["json"]
        self.assertEqual(payload["model"], "typesafe/jev-1.13")
        self.assertEqual(
            payload["state"],
            {
                "url": self.url,
                "title": self.title,
                "description": self.description,
            },
        )
        self.assertEqual(
            payload["questions"],
            jev.build_questions(self.tag_names),
        )

    def test_returns_empty_list_without_tags(self):
        recommendations = jev.get_recommendations(
            self.url, self.title, self.description, [], self.api_key
        )

        self.assertEqual(recommendations, [])

    def test_returns_empty_list_without_api_key(self):
        recommendations = jev.get_recommendations(
            self.url, self.title, self.description, self.tag_names, ""
        )

        self.assertEqual(recommendations, [])

    def test_dedupes_tag_names_case_insensitively(self):
        with patch("bookmarks.services.jev.requests.post") as mock_post:
            mock_post.return_value = mock_response(
                {
                    "kaffe": {"type": "noul", "noul": 0.9},
                }
            )
            recommendations = jev.get_recommendations(
                self.url,
                self.title,
                self.description,
                ["kaffe", "Kaffe", "KAFFE"],
                self.api_key,
            )

        self.assertEqual(recommendations, [{"name": "kaffe", "score": 0.9}])

    def test_skips_tags_without_noul_answer(self):
        with patch("bookmarks.services.jev.requests.post") as mock_post:
            mock_post.return_value = mock_response(
                {
                    "kaffe": {"type": "noul", "noul": 0.9},
                    "kök": {},
                }
            )
            recommendations = jev.get_recommendations(
                self.url,
                self.title,
                self.description,
                ["kaffe", "kök"],
                self.api_key,
            )

        self.assertEqual(recommendations, [{"name": "kaffe", "score": 0.9}])


class JevCachedRecommendationsTestCase(TestCase):
    def test_second_call_uses_cache(self):
        self.url = "https://example.com/one"
        tag_names = ["kaffe"]
        api_key = "sk-or-v1-test-key"

        with patch("bookmarks.services.jev.requests.post") as mock_post:
            mock_post.return_value = mock_response(
                {"kaffe": {"type": "noul", "noul": 0.9}}
            )
            first = jev.get_recommendations_cached(
                self.url, "Title", "Desc", tag_names, api_key
            )
            second = jev.get_recommendations_cached(
                self.url, "Title", "Desc", tag_names, api_key
            )

        self.assertEqual(first, [{"name": "kaffe", "score": 0.9}])
        self.assertEqual(second, [{"name": "kaffe", "score": 0.9}])
        # Only one API call should have been made
        mock_post.assert_called_once()

    def test_returns_empty_list_without_api_key(self):
        recommendations = jev.get_recommendations_cached(
            "https://example.com", "Title", "Desc", ["kaffe"], ""
        )

        self.assertEqual(recommendations, [])
