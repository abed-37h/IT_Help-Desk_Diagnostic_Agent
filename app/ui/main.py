"""
app/ui/main.py — Streamlit interface for the IT Help-Desk Diagnostic Agent.

Current purpose:
- Connect Streamlit UI to MemoryManager.
- Store user messages in short-term memory.
- Store assistant messages in short-term memory.
- Store intent, collected info, missing fields, confirmation state,
  latest tool result, and workflow state in working memory.
- Show memory in the sidebar for debugging/observability.

Placeholder logic will later be replaced by app/agent/orchestrator.py.
"""
from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.memory.memory_manager import MemoryManager


st.set_page_config(
    page_title="IT Help Desk Assistant",
    page_icon="🛠️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------
def init_state() -> None:
    """
    Create one MemoryManager per Streamlit session.
    """
    if "memory" not in st.session_state:
        st.session_state.memory = MemoryManager()


# ---------------------------------------------------------------------------
# PLACEHOLDER LOGIC — replace with orchestrator calls later
# ---------------------------------------------------------------------------
def classify_intent(text: str) -> str:
    """
    Temporary intent classifier.

    Later, this will move to DiagnosticTool / orchestrator.
    """
    text = text.lower()

    if "ticket" in text or "create a ticket" in text or "log this" in text:
        return "create_ticket"

    if (
        "vpn" in text
        or "password" in text
        or "printer" in text
        or "wifi" in text
        or "wi-fi" in text
        or "network" in text
        or "internet" in text
    ):
        return "troubleshoot_issue"

    return "unsupported_request"


def update_memory_for_troubleshooting(user_message: str) -> tuple[str, str]:
    """
    Placeholder troubleshooting logic.

    Returns:
        tool_name, reply_text
    """
    memory = st.session_state.memory
    lowered = user_message.lower()

    memory.set_intent("troubleshoot_issue")
    memory.set_workflow_state("information_gathering")

    if "vpn" in lowered:
        memory.collect_info("category", "network")
        memory.collect_info("symptoms", ["VPN not connecting"])
        memory.require_fields(["device_os", "vpn_error_message"])

        return (
            "InfoTool (KnowledgeBaseTool)",
            "I understand your VPN is not connecting. "
            "Please tell me your device OS and the exact VPN error message you see.",
        )

    if "wifi" in lowered or "wi-fi" in lowered or "internet" in lowered or "network" in lowered:
        memory.collect_info("category", "network")
        memory.collect_info("symptoms", ["network connectivity issue"])
        memory.require_fields(["connection_type", "affected_devices"])

        return (
            "InfoTool (KnowledgeBaseTool)",
            "This sounds like a network issue. "
            "Are you using Wi-Fi or Ethernet, and is the issue only on your device?",
        )

    if "password" in lowered or "account" in lowered:
        memory.collect_info("category", "account")
        memory.collect_info("symptoms", ["account access issue"])
        memory.require_fields(["account_type", "error_message"])

        return (
            "InfoTool (KnowledgeBaseTool)",
            "This sounds like an account issue. "
            "Which account are you trying to access, and what error message appears?",
        )

    if "printer" in lowered:
        memory.collect_info("category", "hardware")
        memory.collect_info("symptoms", ["printer issue"])
        memory.require_fields(["printer_name", "error_message"])

        return (
            "InfoTool (KnowledgeBaseTool)",
            "This sounds like a printer issue. "
            "Please tell me the printer name and the error message shown.",
        )

    memory.collect_info("category", "unknown")
    memory.collect_info("symptoms", [user_message])
    memory.require_fields(["issue_details"])

    return (
        "Fallback",
        "I can help troubleshoot this. "
        "Please describe the device, application, and any error message.",
    )


def request_ticket_confirmation(user_message: str) -> str:
    """
    Placeholder ticket confirmation logic.

    Does not create a ticket immediately.
    It only stores a pending confirmation in memory.
    """
    memory = st.session_state.memory

    memory.set_intent("create_ticket")
    memory.set_workflow_state("awaiting_confirmation")
    memory.collect_info("ticket_summary", user_message)
    memory.require_fields(["user_confirmation"])

    memory.request_confirmation(
        action="create_ticket",
        details={
            "title": user_message,
            "category": memory.get_context()["working"]["collected_info"].get("category", "unknown"),
            "priority": "medium",
        },
    )

    return "I can create a support ticket for this. Please confirm using the button below."


def run_placeholder_agent(user_message: str) -> tuple[str, str]:
    """
    Temporary agent logic until orchestrator.py is ready.

    Returns:
        tool_name, assistant_reply
    """
    intent = classify_intent(user_message)

    if intent == "create_ticket":
        reply = request_ticket_confirmation(user_message)
        return "ActionTool (TicketTool)", reply

    if intent == "troubleshoot_issue":
        return update_memory_for_troubleshooting(user_message)

    st.session_state.memory.set_intent("unsupported_request")
    st.session_state.memory.set_workflow_state("human_handoff")
    st.session_state.memory.collect_info("category", "unknown")
    st.session_state.memory.require_fields(["supported_issue_type"])

    return (
        "Fallback",
        "I'm not fully able to handle that yet. "
        "I can help with network, account, operating system, application, or hardware issues.",
    )


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    """
    Render memory/debug information.
    """
    memory = st.session_state.memory
    context = memory.get_context()

    with st.sidebar:
        st.subheader("Session")

        user_name = context["short_term"]["user_name"]

        if user_name:
            st.success(f"Current user: {user_name}")
        else:
            st.warning("No user set yet.")

        st.divider()

        st.subheader("Short-term Memory")
        st.json(context["short_term"])

        st.subheader("Working Memory")
        st.json(context["working"])

        st.divider()

        if st.button("Reset current task"):
            memory.reset_current_task()
            st.rerun()

        if st.button("Reset all memory"):
            memory.reset_all()
            st.rerun()


def render_chat_history() -> None:
    """
    Render chat messages stored inside MemoryManager.
    """
    context = st.session_state.memory.get_context()
    messages = context["short_term"]["recent_messages"]

    for message in messages:
        role = message["role"]
        content = message["content"]
        tool = message.get("tool")

        if role in {"user", "assistant"}:
            with st.chat_message(role):
                st.markdown(content)
                if tool:
                    st.caption(f"🔧 Tool used: {tool}")


def handle_name_capture() -> bool:
    """
    Capture the user's name before starting the chat.

    Returns:
        True if user is already known.
        False if still waiting for name.
    """
    memory = st.session_state.memory
    context = memory.get_context()

    if context["short_term"]["user_name"] is not None:
        return True

    name = st.chat_input("Before we start, what's your name?")

    if name:
        memory.set_user(name)
        memory.set_workflow_state("awaiting_user")

        greeting = f"Nice to meet you, {name}! What IT issue can I help with today?"
        memory.add_assistant_message(greeting)

        st.rerun()

    return False


def handle_pending_confirmation() -> bool:
    """
    Handle confirmation gate before state-changing actions.

    Returns:
        True if confirmation is pending and input should be blocked.
        False if no confirmation is pending.
    """
    memory = st.session_state.memory
    context = memory.get_context()
    pending = context["working"]["pending_confirmation"]

    if not pending:
        return False

    st.warning(f"Confirm action: **{pending['action']}**")

    col1, col2 = st.columns(2)

    if col1.button("✅ Confirm"):
        confirmed_action = memory.confirm_action()

        # Placeholder ticket creation.
        # Later this will call TicketTool.
        ticket_id = "TCK-0003"

        memory.store_tool_result(
            tool_name="TicketTool",
            result={
                "ticket_id": ticket_id,
                "status": "open",
                "confirmed_action": confirmed_action,
            },
        )

        memory.set_workflow_state("reporting")

        reply = f"Ticket **{ticket_id}** has been created. You'll get updates by email."
        memory.add_assistant_message(reply, tool="ActionTool (TicketTool)")

        st.rerun()

    if col2.button("❌ Cancel"):
        memory.cancel_action()

        reply = "Okay, I won't create a ticket."
        memory.add_assistant_message(reply)

        st.rerun()

    return True


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
def main() -> None:
    init_state()

    st.title("IT Help Desk Assistant 🛠️")
    st.caption(
        "This assistant provides decision support for common IT issues. "
        "It does not replace your IT department for urgent or critical problems."
    )

    render_sidebar()

    if not handle_name_capture():
        return

    render_chat_history()

    if handle_pending_confirmation():
        return

    user_input = st.chat_input("Describe your issue...")

    if user_input:
        memory = st.session_state.memory

        memory.add_user_message(user_input)

        tool_name, reply = run_placeholder_agent(user_input)

        memory.add_assistant_message(reply, tool=tool_name)

        st.rerun()


if __name__ == "__main__":
    main()