"""Tests for intent_detection module."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_client():
    with patch("intent_detection.client") as mock:
        yield mock


def make_response(intent: str, confidence: float, is_ambiguous: float) -> MagicMock:
    resp = MagicMock()
    resp.answers["intent"].choice = intent
    resp.answers["confidence"].score = confidence
    resp.answers["is_ambiguous"].noul = is_ambiguous
    return resp


class TestDetectIntent:
    def test_returns_expected_keys(self, mock_client):
        mock_client.system_one.return_value = make_response("account_balance", 1.8, 0.1)
        from intent_detection import detect_intent

        result = detect_intent("What is my account balance?")
        assert set(result.keys()) == {"intent", "confidence", "is_ambiguous"}

    def test_returns_detected_intent(self, mock_client):
        mock_client.system_one.return_value = make_response("credit_card_application", 1.9, 0.2)
        from intent_detection import detect_intent

        result = detect_intent("I want to apply for a credit card")
        assert result["intent"] == "credit_card_application"

    def test_returns_confidence_score(self, mock_client):
        mock_client.system_one.return_value = make_response("atm_location", 1.7, 0.3)
        from intent_detection import detect_intent

        result = detect_intent("Where is the nearest ATM?")
        assert result["confidence"] == pytest.approx(1.7)

    def test_returns_ambiguity_score(self, mock_client):
        mock_client.system_one.return_value = make_response("customer_support", 0.3, 0.85)
        from intent_detection import detect_intent

        result = detect_intent("Hello")
        assert result["is_ambiguous"] == pytest.approx(0.85)

    def test_passes_message_as_state(self, mock_client):
        mock_client.system_one.return_value = make_response("password_reset", 1.8, 0.1)
        from intent_detection import detect_intent

        detect_intent("I forgot my password")
        call_kwargs = mock_client.system_one.call_args.kwargs
        assert call_kwargs["state"] == "I forgot my password"

    def test_questions_include_all_three_types(self, mock_client):
        mock_client.system_one.return_value = make_response("loan_application", 1.5, 0.2)
        from intent_detection import detect_intent

        detect_intent("Apply for a loan")
        questions = mock_client.system_one.call_args.kwargs["questions"]
        assert "intent" in questions
        assert "confidence" in questions
        assert "is_ambiguous" in questions


class TestCriteria:
    def test_criteria_loads_49_intents(self):
        from intent_detection import criteria

        assert len(criteria) == 49

    def test_criteria_keys_are_strings(self):
        from intent_detection import criteria

        assert all(isinstance(k, str) for k in criteria)

    def test_known_intents_present(self):
        from intent_detection import criteria

        expected = [
            "credit_card_application",
            "account_balance",
            "fund_transfer",
            "atm_location",
            "password_reset",
            "loan_application",
        ]
        for intent in expected:
            assert intent in criteria, f"Missing intent: {intent}"

    def test_definitions_are_non_empty_strings(self):
        from intent_detection import criteria

        for intent, definition in criteria.items():
            assert isinstance(definition, str) and definition.strip(), (
                f"Empty definition for: {intent}"
            )
