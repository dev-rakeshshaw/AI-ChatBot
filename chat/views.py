import uuid
from typing import Optional
from django.http import JsonResponse, HttpResponse, HttpRequest
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from .services.llm_service import get_llm_response


@require_http_methods(["GET", "POST"])
def chat(request: HttpRequest) -> HttpResponse:
    """
    Single view to:
    - GET: generate a new conversation_id (UUID), store in session, render chat UI with conversation_id in template.
    - POST: accept 'query' and 'conversation_id', validate conversation_id against session,
            call LangGraph (get_llm_response) with thread_id=conversation_id, return JSON.
    """
    print("➡️ Entered chat view with method:", request.method)

    if request.method == "GET":
        # Generate a new conversation id and store in session
        conversation_id = str(uuid.uuid4())
        request.session["conversation_id"] = conversation_id
        print(f"[GET] Generated new conversation_id: {conversation_id}")
        print("[GET] Stored conversation_id in session")

        # Render the chat template, injecting the conversation id for the hidden field
        return render(request, "chat/chat.html", {"conversation_id": conversation_id})

    # POST handling
    print("➡️ Handling POST request")

    # Read posted values
    user_message = request.POST.get("query", "").strip()
    conversation_id = request.POST.get("conversation_id", "")
    print(f"[POST] Received user_message: '{user_message}'")
    print(f"[POST] Received conversation_id from UI: {conversation_id}")

    # Helper to try to get an existing conversation id from session
    session_conv_id: Optional[str] = request.session.get("conversation_id")
    print(f"[POST] Session conversation_id: {session_conv_id}")

    # NEW: strict UUID validation
    try:
        uuid.UUID(conversation_id)
    except (ValueError, TypeError):
        print("❌ Invalid conversation_id format.")
        return JsonResponse({
            "user": user_message,
            "bot": "⚠️ Invalid conversation id, sir. Please refresh to start a new conversation.",
            "conversation_id": None,
        }, status=400)

    # Validation 1: conversation id missing in session
    if not session_conv_id:
        print("❌ No conversation_id in session. Something went wrong.")
        return JsonResponse(
            {
                "user": user_message,
                "bot": "⚠️ Session expired or invalid, sir. Please refresh to start a new conversation.",
                "conversation_id": None,
            }
        )

    # Validation 2: mismatch between session and UI
    if conversation_id != session_conv_id:
        print("❌ Mismatch between session and UI conversation_id.")
        return JsonResponse(
            {
                "user": user_message,
                "bot": "⚠️ Conversation mismatch detected, sir. Please refresh the page to reset.",
                "conversation_id": session_conv_id,
            }
        )

    # Validation 3: empty user message
    if not user_message:
        print("⚠️ User submitted empty message.")
        return JsonResponse(
            {
                "user": user_message,
                "bot": "⚠️ Please enter a message, sir.",
                "conversation_id": conversation_id,
            }
        )

    # Call LangGraph-backed LLM service using conversation_id as thread_id
    try:
        print(f"[POST] Calling get_llm_response with thread_id={conversation_id}")
        bot_message = get_llm_response(user_message, thread_id=conversation_id)
        print(f"[POST] Received bot_message: {bot_message}")
        if not bot_message:
            print("⚠️ Bot returned empty response.")
            bot_message = "⚠️ Sorry sir, I couldn't generate a response."
    except Exception as e:
        print(f"❌ Exception while calling LLM: {e}")
        bot_message = f"❌ Error: {str(e)}"

    # Return JSON including the conversation_id so UI and future calls remain in sync
    print("[POST] Returning response JSON to client")
    return JsonResponse(
        {"user": user_message, "bot": bot_message, "conversation_id": conversation_id}
    )
