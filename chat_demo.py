from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path
import gradio as gr
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient()

intents_path = Path(__file__).parent.parent / "intent-detection" / "data" / "sample_intents.json"
intents = json.loads(intents_path.read_text())
criteria = {item["intent"]: item["definition"] for item in intents}

INTENT_CATEGORIES = {
    "credit_card": "#3B82F6",
    "debit_card": "#8B5CF6",
    "account": "#10B981",
    "loan": "#F59E0B",
    "fund_transfer": "#EF4444",
    "international_transfer": "#EF4444",
    "atm": "#06B6D4",
    "branch": "#06B6D4",
    "cheque": "#84CC16",
    "fixed_deposit": "#F97316",
    "password": "#EC4899",
    "interest_rate": "#F59E0B",
    "service_charge": "#F59E0B",
    "customer_support": "#6B7280",
    "complaint": "#6B7280",
    "feedback": "#6B7280",
}

def get_intent_color(intent: str) -> str:
    for prefix, color in INTENT_CATEGORIES.items():
        if intent.startswith(prefix):
            return color
    return "#6B7280"

def confidence_bar(score: float) -> str:
    # score is 0–2 (Score with 3 criteria), normalize to 0–1
    pct = min(score / 2.0, 1.0) * 100
    if pct >= 80:
        bar_color = "#10B981"
    elif pct >= 50:
        bar_color = "#F59E0B"
    else:
        bar_color = "#EF4444"
    return f"""
    <div style="background:#1e293b;border-radius:4px;height:8px;width:100%;margin-top:4px">
      <div style="background:{bar_color};border-radius:4px;height:8px;width:{pct:.0f}%"></div>
    </div>
    <span style="font-size:11px;color:#94a3b8">{pct:.0f}%</span>
    """

def detect_and_respond(user_message: str, history: list) -> tuple[list, str]:
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

    intent = response.answers["intent"].choice
    confidence = response.answers["confidence"].score
    is_ambiguous = response.answers["is_ambiguous"].noul

    color = get_intent_color(intent)
    intent_label = intent.replace("_", " ").title()
    definition = criteria[intent]

    if is_ambiguous > 0.6:
        bot_reply = f"I'm not entirely sure what you're looking for. Could you clarify your request? It might be related to **{intent_label}**, but your message is a bit ambiguous."
    else:
        bot_reply = f"Got it — this looks like a **{intent_label}** request. {definition} How can I assist you further?"

    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": bot_reply},
    ]

    # Build the intent analysis panel
    ambiguous_label = "Yes" if is_ambiguous > 0.5 else "No"
    ambiguous_color = "#EF4444" if is_ambiguous > 0.5 else "#10B981"

    panel_html = f"""
    <div style="font-family: system-ui, sans-serif; padding: 16px; background: #0f172a; border-radius: 12px; color: #e2e8f0;">
      <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-bottom: 12px;">
        Intent Analysis
      </div>

      <div style="background: {color}22; border: 1px solid {color}55; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
        <div style="font-size: 11px; color: {color}; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;">Detected Intent</div>
        <div style="font-size: 18px; font-weight: 600; color: #f1f5f9;">{intent_label}</div>
        <div style="font-size: 12px; color: #94a3b8; margin-top: 6px; line-height: 1.5;">{definition}</div>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
        <div style="background: #1e293b; border-radius: 8px; padding: 10px;">
          <div style="font-size: 11px; color: #64748b; margin-bottom: 4px;">Confidence</div>
          {confidence_bar(confidence)}
        </div>
        <div style="background: #1e293b; border-radius: 8px; padding: 10px;">
          <div style="font-size: 11px; color: #64748b; margin-bottom: 4px;">Ambiguous</div>
          <div style="margin-top: 4px;">
            <span style="background: {ambiguous_color}22; color: {ambiguous_color}; border-radius: 4px; padding: 2px 8px; font-size: 12px; font-weight: 500;">{ambiguous_label}</span>
          </div>
          <span style="font-size: 11px; color: #94a3b8">score: {is_ambiguous:.2f}</span>
        </div>
      </div>

      <div style="background: #1e293b; border-radius: 8px; padding: 10px;">
        <div style="font-size: 11px; color: #64748b; margin-bottom: 6px;">Raw Scores</div>
        <div style="font-size: 12px; color: #94a3b8; font-family: monospace; line-height: 1.8;">
          intent: <span style="color: {color}">{intent}</span><br>
          confidence: <span style="color: #e2e8f0">{confidence:.3f}</span><br>
          is_ambiguous: <span style="color: #e2e8f0">{is_ambiguous:.3f}</span>
        </div>
      </div>
    </div>
    """

    return history, panel_html


EXAMPLE_MESSAGES = [
    "I want to apply for a credit card",
    "Where's the nearest ATM?",
    "I forgot my banking password",
    "I lost my debit card",
    "Can I send money overseas?",
    "What are your interest rates?",
    "Hello",
]

with gr.Blocks(title="Banking Intent Detection") as demo:
    gr.Markdown(
        """
        # Banking Intent Detection
        *Powered by [jev](https://jev.ai) via typesafe-sdk — type any banking query to see the detected intent in real time.*
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=3, elem_classes="chat-col"):
            chatbot = gr.Chatbot(
                label="Chat",
                height=500,
                show_label=False,
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask something about your account, cards, loans...",
                    show_label=False,
                    scale=5,
                    submit_btn=True,
                )

            gr.Examples(
                examples=EXAMPLE_MESSAGES,
                inputs=msg_input,
                label="Try these",
            )

        with gr.Column(scale=2):
            intent_panel = gr.HTML(
                value="""
                <div style="font-family:system-ui,sans-serif;padding:16px;background:#0f172a;border-radius:12px;color:#475569;text-align:center;height:200px;display:flex;align-items:center;justify-content:center;">
                  <div>
                    <div style="font-size:32px;margin-bottom:8px;">💬</div>
                    <div style="font-size:14px;">Send a message to see<br>intent analysis here</div>
                  </div>
                </div>
                """,
                label="Intent Analysis",
            )

    state = gr.State([])

    msg_input.submit(
        fn=detect_and_respond,
        inputs=[msg_input, state],
        outputs=[state, intent_panel],
    ).then(
        fn=lambda h: (h, ""),
        inputs=[state],
        outputs=[chatbot, msg_input],
    )


if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        .gradio-container { max-width: 1100px !important; }
        footer { display: none !important; }
        """,
    )
