from os import getenv
from dotenv import load_dotenv
import json
import re
from typing import Optional, TypedDict, Annotated, List, Dict
from operator import add as list_append
from pathlib import Path
from django.conf import settings

from .checkpointers.json_folder_saver import JsonFolderSaver

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# If Django settings are available, we can base paths from BASE_DIR. Optional.
try:
    BASE_DIR = Path(settings.BASE_DIR)
except Exception:
    # Fallback: project root ~= 2 levels up from this file (chat/services/llm_service.py)
    BASE_DIR = Path(__file__).resolve().parents[2]

print("\n✅ Imports loaded successfully\n")

load_dotenv()
print("\n✅ Environment variables loaded\n")


# ──────────────────────────────────────────────────────────────────────────────
# State: only your pair list + minimal routing fields
# ──────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    messages: Annotated[List[Dict[str, str]], list_append]   # [{ "user_query": str, "answer": str }]
    current_user_query: Optional[str]                         # set per turn
    is_math_task: Optional[bool]
    refromed_user_query: Optional[str]                        # keep misspelling as requested

print("\n✅ AgentState defined\n")


# ──────────────────────────────────────────────────────────────────────────────
# LLM client
# ──────────────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url=getenv("OPENROUTER_BASE_URL"),
    model="deepseek/deepseek-chat-v3.1:free",
    default_headers={
        "HTTP-Referer": getenv("YOUR_SITE_URL"),
        "X-Title": getenv("YOUR_SITE_NAME"),
    },
    timeout=30,
    max_retries=2,
)
print("\n✅ LLM initialized\n")


# ──────────────────────────────────────────────────────────────────────────────
# Prompts (simple & focused)
# ──────────────────────────────────────────────────────────────────────────────
CLASSIFY_SYSTEM = (
    "You are a routing and query-refinement module for an AI assistant named Jarvis.\n"
    "Decide if the user's query is MATHEMATICAL (calculation, algebra, geometry, calculus, probability, statistics),\n"
    "or GENERAL chit-chat/factual. Then rewrite it clearly for the next node.\n\n"
    "Return ONLY a JSON object (no markdown, no extra text) with exactly these keys:\n"
    "{\n"
    "  \"is_math_task\": true|false,\n"
    "  \"refromed_user_query\": \"<elaborated version of the user's query>\"\n"
    "}\n"
)

CLASSIFY_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("human", "User Query: {user_query}")
])

CHAT_SYSTEM = (
    "You are Jarvis, an advanced AI assistant inspired by Iron Man’s AI. "
    "You are formal, intelligent, and slightly witty. Always address the user as 'sir'. "
    "If the user asks for a summary of past queries, read the conversation_history and summarize accordingly."
)

CHAT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("human",
     "User Query: {refined_query}\n\n"
     "conversation_history: {history_json}")
])


MATH_SYSTEM = (
    "You are Jarvis in math mode. Address the user as 'sir'. "
    "Solve the mathematical query precisely and concisely. "
    "Provide brief working when helpful, then the final result."
)
MATH_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("human", "{refined_query}")
])

print("\n✅ Prompts & templates ready\n")


# ──────────────────────────────────────────────────────────────────────────────
# Persistence helpers (per-conversation folders + per-turn JSON files)
# ──────────────────────────────────────────────────────────────────────────────
CONVERSATIONS_ROOT = BASE_DIR / "conversations"
CONVERSATIONS_ROOT.mkdir(parents=True, exist_ok=True)  # ensure root exists

def _ensure_conv_dir(thread_id: str) -> Path:
    """
    Ensure conversations/<thread_id>/ exists; return that folder path.
    """
    conv_dir = CONVERSATIONS_ROOT / thread_id
    conv_dir.mkdir(parents=True, exist_ok=True)
    return conv_dir

def _persist_turn(thread_id: str, turn_index: int, user_query: str, answer: str) -> None:
    """
    Write a per-turn JSON file inside conversations/<thread_id>/ as 0001.json, 0002.json, ...
    """
    try:
        conv_dir = _ensure_conv_dir(thread_id)
        filename = f"{turn_index:04d}.json"
        path = conv_dir / filename
        payload = {"user_query": user_query, "answer": answer}
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Persisted turn to: {path}\n")
    except Exception as e:
        print(f"\n❌ Failed to persist turn: {e}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Other helpers
# ──────────────────────────────────────────────────────────────────────────────
def _last5_history_pairs(state: AgentState) -> List[Dict[str, str]]:
    pairs = state.get("messages") or []
    return pairs[-5:] if len(pairs) > 5 else pairs

def _parse_json_safely(text: str) -> dict:
    if not text:
        return {}
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return {}
    return {}

def _route_after_classify(state: AgentState) -> str:
    is_math = state.get("is_math_task")
    next_node = "math" if is_math else "chat"
    print(f"\n🔀 Routing decision: is_math_task={is_math} → Next node = {next_node}\n")
    return next_node

print("\n✅ Helper functions defined\n")


# ──────────────────────────────────────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────────────────────────────────────
def classify(state: AgentState) -> dict:
    print("\n🟦 Node 1: classify entered\n")
    user_q = (state.get("current_user_query") or "").strip()
    history5 = _last5_history_pairs(state)
    print(f"\n\n history5==================> {history5}")
    history_json = json.dumps(history5, ensure_ascii=False)
    print(f"\n\n history_json==================> {history_json}")
    prompt_value = CLASSIFY_TEMPLATE.invoke({
        "system_prompt": CLASSIFY_SYSTEM,
        "user_query": user_q,
        "history_json": history_json
    })
    print(f"\n➡️ Classify prompt human content:\nUser Query: {user_q}\nHistory: {history_json}\n")

    ai = llm.invoke(prompt_value.to_messages())
    print(f"\n🔎 Raw classifier output: {ai.content}\n")

    parsed = _parse_json_safely(ai.content or "")
    refined = parsed.get("refromed_user_query") or parsed.get("reformed_user_query") or user_q
    is_math = parsed.get("is_math_task")
    if not isinstance(is_math, bool):
        is_math = False

    print(f"\n✅ Parsed classifier JSON: is_math_task={is_math}, refromed_user_query={refined}\n")
    return {"is_math_task": is_math, "refromed_user_query": refined}


def chat_node(state: AgentState) -> dict:
    print("\n🟩 Node 2: chat entered\n")
    user_q = (state.get("current_user_query") or "").strip()
    refined = state.get("refromed_user_query") or user_q
    history5 = _last5_history_pairs(state)
    history_json = json.dumps(history5, ensure_ascii=False)

    prompt_value = CHAT_TEMPLATE.invoke({
        "system_prompt": CHAT_SYSTEM,
        "refined_query": refined,
        "history_json": history_json
    })
    print(f"\n➡️ Chat prompt refined_query: {refined}\nHistory: {history_json}\n")

    ai = llm.invoke(prompt_value.to_messages())
    print(f"\n🤖 Chat reply: {ai.content}\n")

    # Append new pair
    return {"messages": [{"user_query": user_q, "answer": ai.content}]}

    # Append new pair
    return {"messages": [{"user_query": user_q, "answer": ai.content}]}


def math_node(state: AgentState) -> dict:
    print("\n🟧 Node 3: math entered\n")
    user_q = (state.get("current_user_query") or "").strip()
    refined = state.get("refromed_user_query") or user_q

    prompt_value = MATH_TEMPLATE.invoke({
        "system_prompt": MATH_SYSTEM,
        "refined_query": refined
    })
    print(f"\n➡️ Math prompt refined_query: {refined}\n")

    ai = llm.invoke(prompt_value.to_messages())
    print(f"\n🧮 Math reply: {ai.content}\n")

    return {"messages": [{"user_query": user_q, "answer": ai.content}]}


def debug_state_node(state: AgentState) -> dict:
    print("\n🟪 Node 4: debug_state entered\n")
    try:
        snapshot = {
            "messages": state.get("messages", []),
            "is_math_task": state.get("is_math_task"),
            "refromed_user_query": state.get("refromed_user_query"),
            "current_user_query": state.get("current_user_query"),
        }
        print("\n📊 Full state snapshot:\n", json.dumps(snapshot, indent=2, ensure_ascii=False), "\n")
    except Exception as e:
        print(f"\n❌ Error while printing state: {e}\n")
    return {}

print("\n✅ Nodes defined\n")


# ──────────────────────────────────────────────────────────────────────────────
# Build graph (classify → chat/math → debug_state → END)
# ──────────────────────────────────────────────────────────────────────────────
builder = StateGraph(AgentState)
builder.add_node("classify", classify)          # Node 1
builder.add_node("chat", chat_node)             # Node 2
builder.add_node("math", math_node)             # Node 3
builder.add_node("debug_state", debug_state_node)  # Node 4

builder.set_entry_point("classify")
builder.add_conditional_edges("classify", _route_after_classify, {"chat": "chat", "math": "math"})
builder.add_edge("chat", "debug_state")
builder.add_edge("math", "debug_state")
builder.add_edge("debug_state", END)

# Use the custom filesystem checkpointer (stores under conversations/<thread_id>/_checkpoints)
checkpointer = JsonFolderSaver(CONVERSATIONS_ROOT)
graph = builder.compile(checkpointer=checkpointer)

print("\n✅ LangGraph compiled with 4 nodes + JsonFolderSaver (simple flow)\n")


# ──────────────────────────────────────────────────────────────────────────────
# Public API (Django calls this)
# ──────────────────────────────────────────────────────────────────────────────
def get_llm_response(user_message: str, thread_id: str = "default") -> str:
    """
    Flow:
      - current_user_query=user_message
      - classify uses last 5 pairs + query
      - route to chat/math
      - append {user_query, answer} to messages
      - persist this turn to conversations/<thread_id>/<NNNN>.json
    """
    print(f"\n🚀 get_llm_response called: {user_message}, thread_id={thread_id}\n")
    config = {"configurable": {"thread_id": thread_id}}

    result_state: AgentState = graph.invoke(
        {"current_user_query": user_message},
        config=config,
    )

    pairs = result_state.get("messages") or []
    final_text = pairs[-1]["answer"] if pairs else ""
    print(f"\n✅ Returning to Django: {final_text}\n")

    # Persist the last turn
    try:
        turn_index = len(pairs)  # 1-based index for file naming
        _persist_turn(
            thread_id=thread_id,
            turn_index=turn_index,
            user_query=pairs[-1]["user_query"] if pairs else user_message,
            answer=final_text
        )
    except Exception as e:
        print(f"\n❌ Persistence error (non-fatal): {e}\n")

    return final_text or ""
