import json
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

load_dotenv()

client = TypeSafeClient()

intents_path = Path(__file__).parent.parent / "intent-detection" / "data" / "sample_intents.json"
intents = json.loads(intents_path.read_text())
criteria = {item["intent"]: item["definition"] for item in intents}


def detect_intent(user_message: str) -> dict:
    response = client.system_one(
        state=user_message,
        questions={
            "intent": Choice(
                instructions="Which banking intent best matches the user's message",
                criteria=criteria,
            ),
            "confidence": Score(
                instructions="How confident are you in this intent classification",
                criteria=[
                    "Very uncertain — message is vague or could match multiple intents",
                    "Somewhat confident — likely match but some ambiguity remains",
                    "Very confident — message clearly and unambiguously matches the intent",
                ],
            ),
            "is_ambiguous": Noul(
                instructions="The user's message is too vague or ambiguous to classify reliably",
            ),
        },
    )
    return {
        "intent": response.answers["intent"].choice,
        "confidence": response.answers["confidence"].score,
        "is_ambiguous": response.answers["is_ambiguous"].noul,
    }


if __name__ == "__main__":
    test_messages = [
        "I want to apply for a credit card",
        "Where's the nearest ATM?",
        "I forgot my banking password",
        "I lost my debit card",
        "What are your current interest rates?",
        "Can I send money to someone overseas?",
        "Hello",  # ambiguous
    ]

    for msg in test_messages:
        result = detect_intent(msg)
        print(f"Input:      {msg}")
        print(f"Intent:     {result['intent']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Ambiguous:  {result['is_ambiguous']:.2f}")
        print()
