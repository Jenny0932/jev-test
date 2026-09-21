"""Tests for chat_demo helper functions."""

import pytest

from chat_demo import INTENT_CATEGORIES, confidence_bar, get_intent_color


class TestGetIntentColor:
    def test_credit_card_prefix(self):
        assert get_intent_color("credit_card_application") == INTENT_CATEGORIES["credit_card"]

    def test_debit_card_prefix(self):
        assert get_intent_color("debit_card_block") == INTENT_CATEGORIES["debit_card"]

    def test_account_prefix(self):
        assert get_intent_color("account_balance") == INTENT_CATEGORIES["account"]

    def test_loan_prefix(self):
        assert get_intent_color("loan_application") == INTENT_CATEGORIES["loan"]

    def test_atm_prefix(self):
        assert get_intent_color("atm_location") == INTENT_CATEGORIES["atm"]

    def test_unknown_intent_returns_default(self):
        assert get_intent_color("unknown_intent") == "#6B7280"

    def test_returns_string(self):
        assert isinstance(get_intent_color("account_balance"), str)


class TestConfidenceBar:
    def test_returns_string(self):
        assert isinstance(confidence_bar(1.5), str)

    def test_high_confidence_green(self):
        html = confidence_bar(2.0)
        assert "#10B981" in html

    def test_medium_confidence_amber(self):
        html = confidence_bar(1.0)
        assert "#F59E0B" in html

    def test_low_confidence_red(self):
        html = confidence_bar(0.2)
        assert "#EF4444" in html

    def test_percentage_displayed(self):
        html = confidence_bar(2.0)
        assert "100%" in html

    def test_score_capped_at_100_percent(self):
        html = confidence_bar(999.0)
        assert "100%" in html

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0, 1.5, 2.0])
    def test_valid_scores_produce_output(self, score):
        result = confidence_bar(score)
        assert result and len(result) > 0


class TestDetectAndRespond:
    def test_appends_to_history(self, mocker):
        mock_resp = mocker.MagicMock()
        mock_resp.answers["intent"].choice = "account_balance"
        mock_resp.answers["confidence"].score = 1.8
        mock_resp.answers["is_ambiguous"].noul = 0.1
        mocker.patch("chat_demo.client").system_one.return_value = mock_resp

        from chat_demo import detect_and_respond

        history, _ = detect_and_respond("What's my balance?", [])
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_ambiguous_message_triggers_clarification(self, mocker):
        mock_resp = mocker.MagicMock()
        mock_resp.answers["intent"].choice = "customer_support"
        mock_resp.answers["confidence"].score = 0.3
        mock_resp.answers["is_ambiguous"].noul = 0.9
        mocker.patch("chat_demo.client").system_one.return_value = mock_resp

        from chat_demo import detect_and_respond

        history, _ = detect_and_respond("Hello", [])
        assert "clarify" in history[1]["content"].lower()

    def test_clear_message_returns_intent_label(self, mocker):
        mock_resp = mocker.MagicMock()
        mock_resp.answers["intent"].choice = "credit_card_application"
        mock_resp.answers["confidence"].score = 1.9
        mock_resp.answers["is_ambiguous"].noul = 0.1
        mocker.patch("chat_demo.client").system_one.return_value = mock_resp

        from chat_demo import detect_and_respond

        history, _ = detect_and_respond("I want to apply for a credit card", [])
        assert "Credit Card Application" in history[1]["content"]

    def test_panel_html_contains_intent_label(self, mocker):
        mock_resp = mocker.MagicMock()
        mock_resp.answers["intent"].choice = "fund_transfer"
        mock_resp.answers["confidence"].score = 1.7
        mock_resp.answers["is_ambiguous"].noul = 0.2
        mocker.patch("chat_demo.client").system_one.return_value = mock_resp

        from chat_demo import detect_and_respond

        _, panel = detect_and_respond("Send money to my friend", [])
        assert "Fund Transfer" in panel

    def test_history_accumulates_across_turns(self, mocker):
        mock_resp = mocker.MagicMock()
        mock_resp.answers["intent"].choice = "account_balance"
        mock_resp.answers["confidence"].score = 1.8
        mock_resp.answers["is_ambiguous"].noul = 0.1
        mocker.patch("chat_demo.client").system_one.return_value = mock_resp

        from chat_demo import detect_and_respond

        history, _ = detect_and_respond("First message", [])
        history, _ = detect_and_respond("Second message", history)
        assert len(history) == 4
