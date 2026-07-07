"""
app/ui/main.py — Streamlit interface for the IT Help-Desk Diagnostic Agent.

Design choices tied directly to the project proposal:
- Short-term memory: captures the user's name once, reuses it all session.
- Working memory: explicit dict tracking intent, collected info, missing
  fields, pending confirmation, last tool result, and workflow state —
  rendered in the sidebar for observability/grading purposes.
- Confirmation gate: ticket creation (a state-changing action) requires an
  explicit button click before it "executes".
- Tool visibility: each reply is tagged with which placeholder tool ran,
  so it's obvious which of the four tools (info/analysis/action/report)
  handled the turn.

Everything under "PLACEHOLDER LOGIC" gets replaced by real calls into
app/agent/orchestrator.py once it exists. The UI structure itself
(session state, sidebar, confirmation flow) should not need to change.
"""

import streamlit as st

st.set_page_config(page_title="IT Help Desk Assistant", page_icon="", layout="wide")


# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------
def init_state():
    if "user_name" not in st.session_state:
        st.session_state.user_name = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "working_memory" not in st.session_state:
        st.session_state.working_memory = {
            "current_intent": None,
            "collected_info": {},
            "missing_fields": [],
            "pending_confirmation": None,   # e.g. {"action": "create_ticket", "details": {...}}
            "last_tool_result": None,
            "workflow_state": "awaiting_name",
        }


# ---------------------------------------------------------------------------
# PLACEHOLDER LOGIC — replace with orchestrator calls later
# ---------------------------------------------------------------------------
def classify_intent(text: str) -> str:
    text = text.lower()
    if "ticket" in text or "create a ticket" in text or "log this" in text:
        return "create_ticket"
    if "vpn" in text or "password" in text or "printer" in text or "wifi" in text or "network" in text:
        return "troubleshoot"
    return "unknown"


def run_placeholder_tool(intent: str, user_message: str):
    """Returns (tool_name, reply_text). Stands in for real tool calls."""
    if intent == "troubleshoot":
        return (
            "InfoTool (KnowledgeBaseTool)",
            "Based on the knowledge base: check your connection, restart the "
            "affected client/app, and confirm your credentials are current. "
            "(Placeholder — will be replaced with a real KB lookup.)",
        )
    elif intent == "create_ticket":
        return ("ActionTool (TicketTool)", "__PENDING_CONFIRMATION__")
    else:
        return (
            "Fallback",
            "I'm not fully able to handle that yet. I can help with network, "
            "account, OS, or application issues, or open a support ticket for you.",
        )


# ---------------------------------------------------------------------------
# UI: sidebar (working memory / observability panel)
# ---------------------------------------------------------------------------
def render_sidebar():
    wm = st.session_state.working_memory
    with st.sidebar:
        st.subheader("Working Memory")
        st.write(f"**Workflow state:** `{wm['workflow_state']}`")
        st.write(f"**Current intent:** `{wm['current_intent']}`")
        st.write("**Collected info:**")
        st.json(wm["collected_info"] or {})
        st.write(f"**Missing fields:** {wm['missing_fields'] or 'none'}")
        st.write(f"**Pending confirmation:** {wm['pending_confirmation'] or 'none'}")
        st.write("**Last tool result:**")
        st.json(wm["last_tool_result"] or {})


# ---------------------------------------------------------------------------
# UI: main chat flow
# ---------------------------------------------------------------------------
def main():
    init_state()
    render_sidebar()

    st.title("IT Help Desk Assistant")
    st.caption(
        "This assistant provides decision support for common IT issues. "
        "It does not replace your IT department for urgent or critical problems."
    )

    # --- Step 1: capture the user's name (short-term memory requirement) ---
    if st.session_state.user_name is None:
        name = st.chat_input("Before we start, what's your name?")
        if name:
            st.session_state.user_name = name
            st.session_state.working_memory["workflow_state"] = "chatting"
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Nice to meet you, {name}! What IT issue can I help with today?"}
            )
            st.rerun()
        return

    # --- Render existing conversation ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("tool"):
                st.caption(f"🔧 Tool used: {message['tool']}")

    # --- Handle a pending confirmation gate first, if one exists ---
    wm = st.session_state.working_memory
    if wm["pending_confirmation"]:
        st.warning(f"Confirm action: **{wm['pending_confirmation']['action']}**")
        col1, col2 = st.columns(2)
        if col1.button("✅ Confirm"):
            ticket_id = "TCK-0003"  # placeholder — real version calls TicketTool
            reply = f"Ticket **{ticket_id}** has been created. You'll get updates by email."
            st.session_state.messages.append({"role": "assistant", "content": reply, "tool": "ActionTool (TicketTool)"})
            wm["last_tool_result"] = {"ticket_id": ticket_id, "status": "open"}
            wm["pending_confirmation"] = None
            wm["workflow_state"] = "chatting"
            st.rerun()
        if col2.button("❌ Cancel"):
            st.session_state.messages.append({"role": "assistant", "content": "Okay, I won't create a ticket."})
            wm["pending_confirmation"] = None
            wm["workflow_state"] = "chatting"
            st.rerun()
        return  # block new input until confirmation is resolved

    # --- Normal chat input ---
    user_input = st.chat_input("Describe your issue...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        intent = classify_intent(user_input)
        wm["current_intent"] = intent
        wm["workflow_state"] = f"handling_{intent}"

        tool_name, reply = run_placeholder_tool(intent, user_input)

        if reply == "__PENDING_CONFIRMATION__":
            wm["pending_confirmation"] = {"action": "create_ticket", "details": {"summary": user_input}}
            wm["workflow_state"] = "awaiting_confirmation"
        else:
            wm["last_tool_result"] = {"tool": tool_name, "summary": reply[:60]}
            st.session_state.messages.append({"role": "assistant", "content": reply, "tool": tool_name})

        st.rerun()


if __name__ == "__main__":
    main()