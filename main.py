import json
import logging
import os
import threading

import uvicorn
from anthropic import Anthropic
from fastapi import FastAPI
from pydantic import BaseModel, Field

import tools

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("customer-support-agent")

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
MAX_TURNS = 5
MAX_HISTORY = 20
MAX_SESSIONS = 1000

SYSTEM_PROMPT = """You are a Customer Support Agent for Acme Store.
You help customers with orders, returns and general questions.
- Use the tools to look up order status and FAQs. Never invent order details.
- If you cannot resolve the issue, or the customer asks for a human, create a support ticket
  and share the ticket ID.
- Keep answers short, friendly and factual."""

# Agent Manager injects these from the LLM provider attached to the agent, so all model calls
# go through the AI gateway. Without them the SDK falls back to its defaults (ANTHROPIC_API_KEY).
url = os.environ.get("ANTHROPIC_URL")
apikey = os.environ.get("ANTHROPIC_API_KEY")

client = Anthropic(
    base_url=url,
    api_key=apikey,
    default_headers={"API-Key": apikey} if apikey else None,
)
app = FastAPI(title="Customer Support Agent")

# In-memory conversation history per session: user messages and final answers only. Lost on restart.
sessions: dict[str, list[dict]] = {}
sessions_lock = threading.Lock()


class ChatRequest(BaseModel):
    message: str
    session_id: str = Field(min_length=8, max_length=128)
    context: dict | None = None


class ChatResponse(BaseModel):
    response: str


def run_agent(history: list[dict]) -> str:
    messages = list(history)
    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=tools.SCHEMAS,
            output_config={"effort": "low"},
            messages=messages,
        )
        if response.stop_reason == "refusal":
            return "Sorry, I can't help with that request."
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")
        # Pass the reply back unchanged (including thinking blocks), then all tool results in one message.
        messages.append({"role": "assistant", "content": response.content})
        results = [
            {"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(tools.call(b.name, b.input))}
            for b in response.content
            if b.type == "tool_use"
        ]
        messages.append({"role": "user", "content": results})
    return "Sorry, I couldn't complete that request. Please try rephrasing."


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    with sessions_lock:
        history = list(sessions.get(request.session_id, []))
    history.append({"role": "user", "content": request.message})
    try:
        answer = run_agent(history)
    except Exception:
        log.exception("Agent run failed for session %s", request.session_id)
        return ChatResponse(response="Sorry, something went wrong. Please try again shortly.")
    history.append({"role": "assistant", "content": answer})
    with sessions_lock:
        sessions.pop(request.session_id, None)
        if len(sessions) >= MAX_SESSIONS:
            sessions.pop(next(iter(sessions)))  # Evict the least recently used session.
        sessions[request.session_id] = history[-MAX_HISTORY:]
    return ChatResponse(response=answer)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
