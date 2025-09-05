# from os import getenv
# from dotenv import load_dotenv
# import json
# import re
# from typing import Optional, TypedDict, Annotated, List, Dict, Any

# from langchain_openai import ChatOpenAI
# from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
# from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph.message import add_messages

# print("\n✅ Imports loaded successfully\n")

# load_dotenv()
# print("\n✅ Environment variables loaded\n")

# # ──────────────────────────────────────────────────────────────────────────────
# # State definition
# # We keep TWO histories:
# #   1) lc_messages: internal raw turn-by-turn chat used by the model context
# #      (managed with add_messages so LLM replies append automatically)
# #   2) messages: your requested list of dicts: [{user_query, answer}, ...]
# # ──────────────────────────────────────────────────────────────────────────────
# class AgentState(TypedDict, total=False):
#     lc_messages: Annotated[List[BaseMessage], add_messages]
#     messages: List[Dict[str, Any]]          # [{ "user_query": str, "answer": str }, ...]
#     is_math_task: Optional[bool]
#     refromed_user_query: Optional[str]

# print("\n✅ AgentState defined\n")

# # ──────────────────────────────────────────────────────────────────────────────
# # Prompts
# # ──────────────────────────────────────────────────────────────────────────────
# SYSTEM_PROMPT_CLASSIFIER = (
#     "You are a routing and query-refinement module for an AI assistant named Jarvis.\n"
#     "Decide if the user's query is MATHEMATICAL (calculation, algebra, geometry, calculus, probability, stats), "
#     "or GENERAL chit-chat/factual. Then rewrite the query clearly.\n\n"
#     "Return ONLY a JSON object (no markdown, no extra text) with exactly these keys:\n"
#     "{\n"
#     "  \"is_math_task\": true|false,\n"
#     "  \"refromed_user_query\": \"<elaborated version of the user's query>\"\n"
#     "}\n"
# )

# SYSTEM_PROMPT_CHAT = (
#     "You are Jarvis, an advanced AI assistant inspired by Iron Man’s AI. "
#     "Your personality is formal, intelligent, and slightly witty. "
#     "You must always address the user as 'sir'. "
#     "Never break character."
# )

# SYSTEM_PROMPT_MATH = (
#     "You are Jarvis in math mode. Address the user as 'sir'. "
#     "Solve the mathematical query precisely and concisely. "
#     "Provide brief working when helpful, then the final result."
# )

# print("\n✅ System prompts defined\n")

# # ──────────────────────────────────────────────────────────────────────────────
# # LLM client (OpenRouter via OpenAI-compatible ChatOpenAI)
# # ──────────────────────────────────────────────────────────────────────────────
# llm = ChatOpenAI(
#     api_key=getenv("OPENROUTER_API_KEY"),
#     base_url=getenv("OPENROUTER_BASE_URL"),
#     model="mistralai/mistral-small-3.2-24b-instruct:free",
#     default_headers={
#         "HTTP-Referer": getenv("YOUR_SITE_URL"),
#         "X-Title": getenv("YOUR_SITE_NAME"),
#     },
#     timeout=30,
#     max_retries=2,
# )
# print("\n✅ LLM initialized\n")

# # ──────────────────────────────────────────────────────────────────────────────
# # Helpers
# # ──────────────────────────────────────────────────────────────────────────────
# def _last_user_text(state: AgentState) -> str:
#     """
#     Pull the latest user text from lc_messages (internal raw history).
#     We assume invoke() is called with {'lc_messages': [{'role': 'user', 'content': ...}]}
#     """
#     for msg in reversed(state.get("lc_messages", [])):
#         content = getattr(msg, "content", None) or (isinstance(msg, dict) and msg.get("content"))
#         role = getattr(msg, "type", None) or (isinstance(msg, dict) and msg.get("role"))
#         if role in ("human", "user") and isinstance(content, str):
#             return content
#     return ""

# def _parse_json_safely(text: str) -> dict:
#     if not text:
#         return {}
#     cleaned = text.strip()
#     # strip ```json ... ``` fences if model adds them
#     cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
#     try:
#         return json.loads(cleaned)
#     except json.JSONDecodeError:
#         m = re.search(r"\{.*\}", cleaned, flags=re.S)
#         if m:
#             try:
#                 return json.loads(m.group(0))
#             except json.JSONDecodeError:
#                 return {}
#     return {}

# def _route_after_classify(state: AgentState) -> str:
#     is_math = state.get("is_math_task")
#     next_node = "math" if is_math else "chat"
#     print(f"\n🔀 Routing decision: is_math_task={is_math} → Next node = {next_node}\n")
#     return next_node

# print("\n✅ Helper functions defined\n")

# # ──────────────────────────────────────────────────────────────────────────────
# # Nodes
# # ──────────────────────────────────────────────────────────────────────────────
# def classify(state: AgentState) -> dict:
#     print("\n🟦 Node 1: classify entered\n")
#     user_text = _last_user_text(state)
#     print(f"\n➡️ Last user message: {user_text}\n")

#     messages = [
#         SystemMessage(content=SYSTEM_PROMPT_CLASSIFIER),
#         HumanMessage(content=f"User query: {user_text}")
#     ]
#     ai = llm.invoke(messages)
#     print(f"\n🔎 Raw classifier output: {ai.content}\n")

#     parsed = _parse_json_safely(ai.content or "")
#     refined = parsed.get("refromed_user_query") or parsed.get("reformed_user_query")
#     is_math = parsed.get("is_math_task")

#     # Defensive coercion
#     if not isinstance(is_math, bool):
#         is_math = False
#     if not isinstance(refined, str) or not refined.strip():
#         refined = user_text

#     print(f"\n✅ Parsed classifier JSON: is_math_task={is_math}, refromed_user_query={refined}\n")
#     return {"is_math_task": is_math, "refromed_user_query": refined}

# def chat_node(state: AgentState) -> dict:
#     print("\n🟩 Node 2: chat entered\n")
#     refined = state.get("refromed_user_query") or _last_user_text(state)
#     user_text = _last_user_text(state)
#     print(f"\n➡️ Refined query to chat LLM: {refined}\n")

#     ai = llm.invoke([SystemMessage(content=SYSTEM_PROMPT_CHAT), HumanMessage(content=refined)])
#     print(f"\n🤖 Chat reply: {ai.content}\n")

#     # Append to your 'messages' list as {user_query, answer}
#     new_entry = {"user_query": user_text, "answer": ai.content}
#     new_messages = (state.get("messages") or []) + [new_entry]

#     # ALSO append the AI reply to lc_messages so LLM keeps context next turn
#     return {
#         "messages": new_messages,
#         "lc_messages": [ai],  # add_messages will append this to internal history
#     }

# def math_node(state: AgentState) -> dict:
#     print("\n🟧 Node 3: math entered\n")
#     refined = state.get("refromed_user_query") or _last_user_text(state)
#     user_text = _last_user_text(state)
#     print(f"\n➡️ Refined query to math LLM: {refined}\n")

#     ai = llm.invoke([SystemMessage(content=SYSTEM_PROMPT_MATH), HumanMessage(content=refined)])
#     print(f"\n🧮 Math reply: {ai.content}\n")

#     new_entry = {"user_query": user_text, "answer": ai.content}
#     new_messages = (state.get("messages") or []) + [new_entry]

#     return {
#         "messages": new_messages,
#         "lc_messages": [ai],
#     }

# def debug_state_node(state: AgentState) -> dict:
#     print("\n🟪 Node 4: debug_state entered\n")
#     try:
#         # Build a readable snapshot
#         snapshot = {
#             "messages": state.get("messages", []),               # your desired pairs
#             "is_math_task": state.get("is_math_task"),
#             "refromed_user_query": state.get("refromed_user_query"),
#             "_lc_messages_count": len(state.get("lc_messages", [])),
#         }
#         print("\n📊 Full state snapshot:\n", json.dumps(snapshot, indent=2, ensure_ascii=False), "\n")
#     except Exception as e:
#         print(f"\n❌ Error while printing state: {e}\n")
#     return {}

# print("\n✅ Nodes defined\n")

# # ──────────────────────────────────────────────────────────────────────────────
# # Build Graph
# # ──────────────────────────────────────────────────────────────────────────────
# builder = StateGraph(AgentState)

# builder.add_node("classify", classify)         # Node 1
# builder.add_node("chat", chat_node)            # Node 2
# builder.add_node("math", math_node)            # Node 3
# builder.add_node("debug_state", debug_state_node)  # Node 4

# builder.set_entry_point("classify")
# builder.add_conditional_edges("classify", _route_after_classify, {"chat": "chat", "math": "math"})
# builder.add_edge("chat", "debug_state")
# builder.add_edge("math", "debug_state")
# builder.add_edge("debug_state", END)

# memory = MemorySaver()
# graph = builder.compile(checkpointer=memory)

# print("\n✅ LangGraph compiled with 4 nodes + MemorySaver\n")

# # ──────────────────────────────────────────────────────────────────────────────
# # Public API (Django calls this)
# # ──────────────────────────────────────────────────────────────────────────────
# def get_llm_response(user_message: str, thread_id: str = "default") -> str:
#     print(f"\n🚀 get_llm_response called: {user_message}, thread_id={thread_id}\n")
#     config = {"configurable": {"thread_id": thread_id}}

#     # IMPORTANT: we now pass the user message under 'lc_messages' (internal history),
#     # not under 'messages', to avoid clashing with your pair list.
#     result_state: AgentState = graph.invoke(
#         {"lc_messages": [{"role": "user", "content": user_message}]},
#         config=config,
#     )

#     # The latest AI reply is the last item appended to lc_messages
#     ai_msg = result_state.get("lc_messages", [])[-1]
#     final_text = getattr(ai_msg, "content", "") if ai_msg else ""
#     print(f"\n✅ Returning to Django: {final_text}\n")
#     return final_text or ""
# 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
