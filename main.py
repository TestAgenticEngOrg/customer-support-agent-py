import json
import logging
import os
import threading

import uvicorn
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel, Field

import tools

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("customer-support-agent")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_TURNS = 5
MAX_HISTORY = 20
MAX_SESSIONS = 1000

SYSTEM_PROMPT = """You are a Customer Support Agent for Acme Store.
You help customers with orders, returns and general questions.
- Use the tools to look up order status and FAQs. Never invent order details.
- If you cannot resolve the issue, or the customer asks for a human, create a support ticket
  and share the ticket ID.
- Keep answers short, friendly and factual."""

client = OpenAI()  # Reads OPENAI_API_KEY from the environment.
app = FastAPI(title="Customer Support Agent")

# In-memory conversation history per session. Lost on restart.
sessions: dict[str, list[dict]] = {}
sessions_lock = threading.Lock()


class ChatRequest(BaseModel):
    message: str
    session_id: str = Field(min_length=8, max_length=128)
    context: dict | None = None


class ChatResponse(BaseModel):
    response: str


def run_agent(history: list[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
    for _ in range(MAX_TURNS):
        completion = client.chat.completions.create(model=MODEL, messages=messages, tools=tools.SCHEMAS)
        reply = completion.choices[0].message
        if not reply.tool_calls:
            return reply.content or ""
        messages.append(reply.model_dump(exclude_none=True))
        for call in reply.tool_calls:
            result = tools.call(call.function.name, json.loads(call.function.arguments or "{}"))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
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
