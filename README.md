# jev-test

A banking chatbot **intent detection** demo using [jev](https://jev.ai) via the `typesafe-sdk`.

Given a user's natural-language message, the system classifies it into one of 49 banking intents (credit card, loans, transfers, ATM, etc.) and returns a structured result with confidence and ambiguity scores — no prompt engineering, no parsing.

## Files

| File | Description |
|---|---|
| `intent_detection.py` | Core classification logic — `detect_intent(message)` |
| `chat_demo.py` | Gradio chat UI with real-time intent analysis panel |
| `test.py` | Basic `typesafe-sdk` usage example (department/frustration/urgency) |

## How it works

Each message is sent to `client.system_one()` with three structured questions:

- **`intent`** — `Choice` over 49 banking intents (label → definition)
- **`confidence`** — `Score` across three levels (uncertain / likely / certain)
- **`is_ambiguous`** — `Noul` (continuous yes/no) for vague messages

```python
result = detect_intent("I lost my debit card")
# result["intent"]       → "debit_card_block"
# result["confidence"]   → 1.91
# result["is_ambiguous"] → 0.22
```

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file with your API key:

```
TYPESAFE_API_KEY=your_key_here
```

## Usage

**Run the CLI demo:**

```bash
uv run intent_detection.py
```

**Run the Gradio chat UI:**

```bash
uv run chat_demo.py
```

Opens at `http://localhost:7860`. Type any banking query to see the detected intent, confidence bar, and ambiguity score update in real time.

## Development

**Lint & format:**

```bash
uv run ruff check .
uv run ruff format .
```

**Tests:**

```bash
uv run pytest
```

33 tests covering intent classification, criteria loading, color mapping, confidence bar rendering, chat history accumulation, and response content.
