"""
Claude Code-inspired Streamlit UI for the IT Help-Desk Diagnostic Agent.

Only this UI file is changed. The agent, tools, database schema, prompts,
and ticket workflow remain untouched.

Demo behavior:
- DEMO_MODE defaults to enabled.
- Existing tickets are cleared once when the Streamlit server starts.
- Tickets created during the current demo remain visible across reruns.
- Set DEMO_MODE=0 before starting Streamlit to preserve tickets across restarts.
"""

from __future__ import annotations

import html
import os
import re
import sys
import uuid
from pathlib import Path
from textwrap import dedent
from typing import Any

# ---------------------------------------------------------------------------
# Project import path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

import app.agent.orchestrator as orchestrator
from app.data.init_db import connect


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="IT Support Assistant",
    page_icon="🛠️",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Runtime compatibility layer
# ---------------------------------------------------------------------------
#
# These adjustments affect only the running UI process. They do not modify
# orchestrator.py, the tools, the database schema, or any other backend file.


def safe_log_llm_error(
    function_name: str,
    error_code: str,
    message: str,
) -> None:
    """Prevent the logger signature mismatch from hiding the real LLM error."""
    print(
        "[LLM ERROR] "
        f"function={function_name} "
        f"code={error_code} "
        f"message={message}"
    )


def safe_log_tool_error(
    tool_name: str,
    error_payload: Any,
) -> None:
    """Log a tool error without interrupting the UI."""
    print(
        "[TOOL ERROR] "
        f"tool={tool_name} "
        f"details={error_payload}"
    )


orchestrator.logger.log_llm_error = safe_log_llm_error
orchestrator.logger.log_tool_error = safe_log_tool_error


# Keep the model configuration that already worked with this integration.
orchestrator.llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
)


# Expose only the teammate's required support workflow functions.
orchestrator.tooled_llm = orchestrator.llm.bind_tools(
    [
        orchestrator.classify_and_validate,
        orchestrator.fetch_issue_knowledge,
        orchestrator.open_support_ticket,
        orchestrator.update_support_ticket,
    ]
)


# ---------------------------------------------------------------------------
# LangGraph compatibility functions
# ---------------------------------------------------------------------------


def route_for_ui(
    state: orchestrator.AgentState,
) -> str:
    """Route assistant tool requests to the tool-execution node."""
    messages = state.get("messages", [])

    if not messages:
        return "end"

    last_message = messages[-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return "end"


def execute_tools_for_ui(
    state: orchestrator.AgentState,
) -> orchestrator.AgentState:
    """
    Execute tools using the orchestrator's existing handlers.

    GraphInterrupt is re-raised so Streamlit can render the human approval
    controls for ticket creation and ticket-status updates.
    """
    messages = state.get("messages", [])

    if not messages:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "The workflow did not provide a tool request "
                        "to execute."
                    )
                )
            ]
        }

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []

    update: dict[str, Any] = {}
    tool_messages: list[ToolMessage] = []

    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "unknown_tool")
        tool_args = tool_call.get("args", {}) or {}
        tool_call_id = (
            tool_call.get("id")
            or f"ui-tool-{uuid.uuid4()}"
        )

        result: Any = None

        try:
            orchestrator.logger.log_tool_execution(tool_name)

            match tool_name:
                case "classify_and_validate":
                    result = orchestrator.handle_classification(
                        state,
                        tool_args,
                    )

                    update["classification_result"] = result.model_dump(
                        mode="json"
                    )

                    if not isinstance(result, orchestrator.Error):
                        update["valid_user_info"] = result.is_valid
                        update["issue_id"] = result.issue_id or None

                case "fetch_issue_knowledge":
                    effective_state = {
                        **state,
                        **update,
                    }

                    result = orchestrator.handle_fetch_kb(
                        effective_state,
                        tool_args,
                    )

                    update["knowledge_result"] = result.model_dump(
                        mode="json"
                    )

                    if not isinstance(result, orchestrator.Error):
                        category = result.category

                        if hasattr(category, "value"):
                            category = category.value

                        update["category"] = category
                        update["severity"] = result.severity
                        update["steps"] = result.steps
                        update["escalate"] = result.escalate

                case "open_support_ticket":
                    effective_state = {
                        **state,
                        **update,
                    }

                    result = orchestrator.handle_open_ticket(
                        effective_state,
                        tool_args,
                    )

                    update["open_ticket_result"] = result.model_dump(
                        mode="json"
                    )

                    if not isinstance(result, orchestrator.Error):
                        update["ticket_id"] = result.ticket_id
                        update["ticket_status"] = result.status

                case "update_support_ticket":
                    effective_state = {
                        **state,
                        **update,
                    }

                    result = orchestrator.handle_update_ticket(
                        effective_state,
                        tool_args,
                    )

                    update["update_ticket_result"] = result.model_dump(
                        mode="json"
                    )

                    if not isinstance(result, orchestrator.Error):
                        update["ticket_id"] = result.ticket_id
                        update["ticket_status"] = result.new_status

                case _:
                    result = orchestrator.Error(
                        error="unsupported_tool",
                        message=(
                            f"The current UI does not execute "
                            f"the tool '{tool_name}'."
                        ),
                    )

        except GraphInterrupt:
            # Approval requests must reach the Streamlit layer.
            raise

        except Exception as error:
            result = orchestrator.Error(
                error="tool_execution_error",
                message=str(error),
            )

        if result is None:
            result = orchestrator.Error(
                error="empty_tool_result",
                message=(
                    f"Tool '{tool_name}' returned no result."
                ),
            )

        if isinstance(result, orchestrator.Error):
            if result.error in {
                "open_ticket_rejected",
                "update_ticket_rejected",
            }:
                tool_message_content = (
                    "The user intentionally cancelled the requested action. "
                    "No ticket was created or updated. "
                    "This was not a technical failure."
                )
            else:
                error_payload = result.model_dump(mode="json")
                tool_message_content = str(error_payload)

                orchestrator.logger.log_tool_error(
                    tool_name,
                    error_payload,
                )

        else:
            result_payload = result.model_dump(mode="json")
            tool_message_content = str(result_payload)

            orchestrator.logger.log_tool_result(
                tool_name,
                result_payload,
            )

        tool_messages.append(
            ToolMessage(
                content=tool_message_content,
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )

    orchestrator.logger.log_state_update(
        "ui_execute_tools",
        update,
    )

    return {
        "messages": tool_messages,
        **update,
    }


def build_ui_graph() -> CompiledStateGraph:
    """Build the working UI graph around the existing orchestrator agent."""
    builder = StateGraph(orchestrator.AgentState)

    builder.add_node(
        "agent",
        orchestrator.agent,
    )

    builder.add_node(
        "tools",
        execute_tools_for_ui,
    )

    builder.add_edge(
        START,
        "agent",
    )

    builder.add_conditional_edges(
        "agent",
        route_for_ui,
        {
            "tools": "tools",
            "end": END,
        },
    )

    builder.add_edge(
        "tools",
        "agent",
    )

    return builder.compile(
        checkpointer=MemorySaver(),
    )


# ---------------------------------------------------------------------------
# Demo database preparation
# ---------------------------------------------------------------------------


@st.cache_resource
def prepare_demo_database() -> bool:
    """
    Clear tickets once when the Streamlit server starts in demo mode.

    The orchestrator initializes the database before this function runs.
    Clearing here removes both seeded records and tickets from older demo
    sessions, while tickets created during the current server process remain.

    Set DEMO_MODE=0 before launching Streamlit to keep database contents.
    """
    demo_mode_enabled = (
        os.getenv("DEMO_MODE", "1").strip() == "1"
    )

    if not demo_mode_enabled:
        return False

    with connect() as connection:
        # Remove dependent history records before deleting tickets.
        connection.execute(
            "DELETE FROM ticket_history"
        )

        connection.execute(
            "DELETE FROM tickets"
        )

        connection.commit()

    print(
        "[DEMO MODE] Ticket database cleared for a fresh demonstration."
    )

    return True


prepare_demo_database()


# ---------------------------------------------------------------------------
# Session-state management
# ---------------------------------------------------------------------------


def initialize_state() -> None:
    """Create all Streamlit session values used by the UI."""
    defaults: dict[str, Any] = {
        "graph": None,
        "session_id": str(uuid.uuid4()),
        "user_info": {},
        "profile_complete": False,
        "profile_sent": False,
        "chat_messages": [],
        "pending_interrupt": None,
        "last_error": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.graph is None:
        st.session_state.graph = build_ui_graph()


def start_new_conversation(
    keep_profile: bool = True,
) -> None:
    """
    Start a new LangGraph thread.

    Employee information remains available when keep_profile is True.
    """
    saved_profile = (
        dict(st.session_state.user_info)
        if keep_profile
        else {}
    )

    st.session_state.graph = build_ui_graph()
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_messages = []
    st.session_state.pending_interrupt = None
    st.session_state.profile_sent = False
    st.session_state.last_error = None
    st.session_state.user_info = saved_profile
    st.session_state.profile_complete = bool(saved_profile)

    if saved_profile:
        employee_name = saved_profile.get(
            "user_name",
            "Employee",
        )

        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    f"Hello {employee_name}. "
                    "Describe the IT issue you are experiencing."
                ),
            }
        ]


initialize_state()


# ---------------------------------------------------------------------------
# Read-only ticket helpers
# ---------------------------------------------------------------------------


def get_ticket_status_counts() -> dict[str, int]:
    """Return ticket totals grouped by status."""
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT status, COUNT(*) AS total
            FROM tickets
            GROUP BY status
            """
        ).fetchall()

    return {
        str(row["status"]): int(row["total"])
        for row in rows
    }


def get_recent_tickets(
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return recent tickets for a compact sidebar display."""
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT
                ticket_id,
                title,
                category,
                priority,
                status
            FROM tickets
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def message_to_text(content: Any) -> str:
    """Convert LangChain message content into displayable Markdown."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []

        for part in content:
            if isinstance(part, str):
                text_parts.append(part)

            elif isinstance(part, dict):
                text = part.get("text")

                if text:
                    text_parts.append(str(text))

        return "\n".join(text_parts)

    return str(content)


def get_last_assistant_text(
    response: dict[str, Any],
) -> str:
    """Return the most recent assistant message from a graph response."""
    messages = response.get("messages", []) or []

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = message_to_text(
                message.content
            ).strip()

            if text:
                return text

    return ""


def process_graph_response(
    response: dict[str, Any],
) -> None:
    """Store either a pending approval or the assistant's final reply."""
    if orchestrator.is_interrupt(response):
        st.session_state.pending_interrupt = (
            orchestrator.get_interrupt_metadata(response)
        )
        return

    st.session_state.pending_interrupt = None

    assistant_text = get_last_assistant_text(response)

    if assistant_text:
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": assistant_text,
            }
        )


def build_orchestrator_message(
    user_message: str,
) -> str:
    """
    Add employee context only to the first hidden orchestrator message.

    The visible chat still shows exactly what the employee typed.
    """
    if st.session_state.profile_sent:
        return user_message

    profile = st.session_state.user_info
    st.session_state.profile_sent = True

    return (
        "Known employee information for this conversation:\n"
        f"- user_name: {profile['user_name']}\n"
        f"- user_id: {profile['user_id']}\n"
        f"- device_type: {profile['device_type']}\n"
        f"- os: {profile['os']}\n\n"
        f"Employee issue:\n{user_message}"
    )


def extract_ticket_id(text: str) -> str | None:
    """Extract a ticket ID already returned by the orchestrator."""
    match = re.search(
        r"\bTCK-\d{4,}\b",
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return match.group(0).upper()


def render_ticket_success_card(
    assistant_text: str,
) -> None:
    """Show a compact success card only for a confirmed created ticket."""
    ticket_id = extract_ticket_id(
        assistant_text
    )

    normalized = assistant_text.casefold()

    creation_language = any(
        phrase in normalized
        for phrase in (
            "created a support ticket",
            "opened a support ticket",
            "ticket has been created",
            "ticket was created",
            "ticket created",
        )
    )

    if not ticket_id or not creation_language:
        return

    ticket_success_html = dedent(
        f"""
        <div class="ticket-success-card">
            <div class="ticket-success-icon">✓</div>
            <div>
                <div class="ticket-success-label">ticket created</div>
                <div class="ticket-success-id">
                    {html.escape(ticket_id)}
                </div>
                <div class="ticket-success-note">
                    status: open · queued for technician review
                </div>
            </div>
        </div>
        """
    ).strip()

    st.markdown(
        ticket_success_html,
        unsafe_allow_html=True,
    )


def render_error_panel() -> None:
    """Display a friendly error with collapsed technical details."""
    last_error = st.session_state.last_error

    if not last_error:
        return

    st.error(
        "The support workflow encountered an error. "
        "Please try again or start a new thread."
    )

    with st.expander(
        "Technical details",
        expanded=False,
    ):
        st.code(
            str(last_error)
        )


def parse_approval_metadata(
    content: str,
) -> dict[str, str]:
    """Extract display-only ticket metadata from the approval sentence."""
    title_match = re.search(
        r"title '([^']+)'",
        content,
        flags=re.IGNORECASE,
    )

    category_match = re.search(
        r"category '([^']+)'",
        content,
        flags=re.IGNORECASE,
    )

    priority_match = re.search(
        r"priority '([^']+)'",
        content,
        flags=re.IGNORECASE,
    )

    return {
        "title": (
            title_match.group(1)
            if title_match
            else "Support request"
        ),
        "category": (
            category_match.group(1)
            if category_match
            else "Not specified"
        ),
        "priority": (
            priority_match.group(1)
            if priority_match
            else "Not specified"
        ),
    }


def parse_update_approval_metadata(
    content: str,
) -> dict[str, str]:
    """
    Extract display-only values from the orchestrator's ticket-update
    approval message.
    """
    ticket_match = re.search(
        r"update ticket\s+([^\s]+)",
        content,
        flags=re.IGNORECASE,
    )

    status_match = re.search(
        r"to status\s+'([^']+)'",
        content,
        flags=re.IGNORECASE,
    )

    note_match = re.search(
        r"with this note:\s*'([^']*)'",
        content,
        flags=re.IGNORECASE,
    )

    return {
        "ticket_id": (
            ticket_match.group(1)
            if ticket_match
            else "Current ticket"
        ),
        "new_status": (
            status_match.group(1)
            if status_match
            else "Not specified"
        ),
        "note": (
            note_match.group(1)
            if note_match
            else "No note provided"
        ),
    }


def is_ticket_update_approval(
    content: str,
) -> bool:
    """Return True when the interrupt asks to update a ticket status."""
    normalized = content.casefold()

    return (
        "update ticket" in normalized
        and "to status" in normalized
    )


def get_latest_user_message() -> str:
    """Return the last visible user message."""
    for message in reversed(
        st.session_state.chat_messages
    ):
        if message.get("role") == "user":
            return str(
                message.get("content", "")
            )

    return "Not available"


def status_dot_class(status: str) -> str:
    """Return the CSS class used for a ticket status dot."""
    return {
        "open": "status-open",
        "in_progress": "status-progress",
        "pending_user": "status-pending",
        "resolved": "status-resolved",
        "closed": "status-closed",
    }.get(
        status,
        "status-closed",
    )


def humanize_status(status: str) -> str:
    """Convert a stored status into a user-facing label."""
    labels = {
        "open": "Open",
        "in_progress": "In progress",
        "pending_user": "Pending",
        "resolved": "Resolved",
        "closed": "Closed",
    }

    return labels.get(
        status,
        status.replace("_", " ").title(),
    )


# ---------------------------------------------------------------------------
# Claude Code-inspired visual theme
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        :root {
            --bg-main: #151411;
            --bg-sidebar: #10100e;
            --bg-panel: #1c1a17;
            --bg-panel-soft: #201e1a;
            --bg-input: #1b1916;
            --border: #3b352d;
            --border-strong: #554b40;
            --text-main: #f4eee6;
            --text-soft: #c3b8aa;
            --text-muted: #8e857a;
            --accent: #e9784a;
            --accent-soft: rgba(233, 120, 74, 0.12);
            --success: #8fbd9a;
            --warning: #e4ad59;
            --pending: #998abf;
            --closed: #716d66;
        }

        html,
        body,
        [class*="css"] {
            font-family:
                "SFMono-Regular",
                "Cascadia Code",
                "Liberation Mono",
                Consolas,
                monospace;
        }

        .stApp,
        [data-testid="stAppViewContainer"],
        section[data-testid="stMain"] {
            background:
                radial-gradient(
                    circle at 20% 0%,
                    rgba(233, 120, 74, 0.07),
                    transparent 28rem
                ),
                var(--bg-main) !important;
            color: var(--text-main);
        }

        .block-container {
            max-width: 1020px;
            padding-top: 1.55rem;
            padding-bottom: 9rem;
        }

        header[data-testid="stHeader"] {
            background: transparent !important;
        }

        [data-testid="stDeployButton"],
        #MainMenu,
        footer {
            display: none !important;
        }

        [data-testid="stSidebar"] {
            background: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.35rem;
        }

        h1,
        h2,
        h3,
        h4,
        p,
        label,
        span,
        div {
            color: inherit;
        }

        /* Keep form field titles readable on the dark employee page. */
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stTextInput"] label,
        [data-testid="stTextInput"] label p,
        [data-testid="stSelectbox"] label,
        [data-testid="stSelectbox"] label p {
            color: var(--text-main) !important;
            opacity: 1 !important;
        }

        .app-shell {
            padding: 1.45rem 1.55rem 1.55rem;
            margin-bottom: 1.3rem;
            border: 1px solid var(--border-strong);
            border-radius: 16px;
            background:
                linear-gradient(
                    145deg,
                    rgba(233, 120, 74, 0.06),
                    transparent 42%
                ),
                var(--bg-panel);
            box-shadow: 0 18px 48px rgba(0, 0, 0, 0.18);
        }

        .terminal-line {
            color: var(--text-muted);
            font-size: 0.78rem;
            font-weight: 700;
        }

        .terminal-prompt {
            color: var(--accent);
        }

        .app-title {
            margin: 2rem 0 0;
            color: var(--text-main);
            font-family:
                Inter,
                system-ui,
                sans-serif;
            font-size: clamp(2rem, 5vw, 3rem);
            line-height: 1.05;
            letter-spacing: -0.045em;
        }

        .app-description {
            max-width: 760px;
            margin-top: 1.15rem;
            color: var(--text-soft);
            font-family:
                Inter,
                system-ui,
                sans-serif;
            font-size: 0.98rem;
            line-height: 1.7;
        }

        .session-line {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem 0.75rem;
            margin-top: 1rem;
            color: var(--text-muted);
            font-size: 0.76rem;
        }

        .session-online {
            color: var(--success);
        }

        .welcome-panel {
            margin: 0.75rem 0 1rem;
            padding: 1rem 1.05rem;
            border: 1px dashed var(--border-strong);
            border-radius: 13px;
            background: rgba(233, 120, 74, 0.035);
        }

        .welcome-title {
            color: var(--text-main);
            font-size: 0.9rem;
            font-weight: 700;
        }

        .welcome-examples {
            display: grid;
            gap: 0.35rem;
            margin-top: 0.65rem;
            color: var(--text-muted);
            font-size: 0.78rem;
            line-height: 1.55;
        }

        .profile-card {
            padding: 0.95rem 1rem;
            border: 1px solid var(--border);
            border-radius: 13px;
            background: var(--bg-panel);
        }

        .profile-label {
            margin-top: 0.8rem;
            color: var(--text-muted);
            font-size: 0.67rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .profile-label:first-child {
            margin-top: 0;
        }

        .profile-value {
            margin-top: 0.15rem;
            color: var(--text-main);
            font-size: 0.86rem;
            font-weight: 700;
            overflow-wrap: anywhere;
        }

        .sidebar-heading {
            margin: 0.25rem 0 0.7rem;
            color: var(--text-soft);
            font-size: 0.77rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .sidebar-heading::first-letter {
            color: var(--accent);
        }

        .status-list {
            display: grid;
            gap: 0.5rem;
        }

        .status-row {
            display: grid;
            grid-template-columns: auto 1fr auto;
            align-items: center;
            gap: 0.65rem;
            padding: 0.78rem 0.85rem;
            border: 1px solid var(--border);
            border-radius: 11px;
            background: var(--bg-panel);
        }

        .status-dot {
            width: 0.55rem;
            height: 0.55rem;
            border-radius: 999px;
        }

        .status-open {
            background: var(--accent);
        }

        .status-progress {
            background: var(--warning);
        }

        .status-pending {
            background: var(--pending);
        }

        .status-resolved {
            background: var(--success);
        }

        .status-closed {
            background: var(--closed);
        }

        .status-name {
            color: var(--text-soft);
            font-size: 0.75rem;
        }

        .status-value {
            color: var(--text-main);
            font-size: 0.84rem;
            font-weight: 800;
        }

        div[data-testid="stExpander"] {
            margin-top: 0.75rem;
            overflow: hidden;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            background: var(--bg-panel) !important;
        }

        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary:hover {
            background: var(--bg-panel) !important;
            color: var(--text-soft) !important;
        }

        div[data-testid="stExpander"] summary {
            padding-top: 0.15rem;
            padding-bottom: 0.15rem;
        }

        .recent-ticket-list {
            display: grid;
            gap: 0.65rem;
            padding: 0.15rem 0 0.35rem;
        }

        .recent-ticket-card {
            padding: 0.8rem 0.85rem;
            border: 1px solid var(--border);
            border-radius: 10px;
            background: var(--bg-sidebar);
        }

        .recent-ticket-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.65rem;
        }

        .recent-ticket-id {
            color: var(--accent);
            font-size: 0.72rem;
            font-weight: 800;
        }

        .recent-ticket-status {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            color: var(--text-muted);
            font-size: 0.67rem;
        }

        .recent-ticket-title {
            margin-top: 0.5rem;
            color: var(--text-main);
            font-family:
                Inter,
                system-ui,
                sans-serif;
            font-size: 0.84rem;
            font-weight: 650;
            line-height: 1.35;
            overflow-wrap: anywhere;
        }

        .recent-ticket-meta {
            margin-top: 0.45rem;
            color: var(--text-muted);
            font-size: 0.67rem;
            line-height: 1.45;
        }

        [data-testid="stChatMessage"] {
            margin-bottom: 0.75rem;
            padding: 0.95rem 1rem;
            border: 1px solid var(--border);
            border-radius: 14px;
            background: rgba(28, 26, 23, 0.78);
        }

        [data-testid="stChatMessageContent"] {
            color: var(--text-main);
            font-family:
                Inter,
                system-ui,
                sans-serif;
            font-size: 0.96rem;
            line-height: 1.68;
        }

        [data-testid="stChatMessageContent"] li {
            margin-bottom: 0.45rem;
        }

        .approval-card {
            margin-top: 1rem;
            padding: 1.05rem;
            border: 1px solid rgba(228, 173, 89, 0.35);
            border-left: 4px solid var(--warning);
            border-radius: 13px;
            background: rgba(228, 173, 89, 0.06);
        }

        .approval-kicker {
            color: var(--warning);
            font-size: 0.69rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .approval-title {
            margin-top: 0.3rem;
            color: var(--text-main);
            font-family:
                Inter,
                system-ui,
                sans-serif;
            font-size: 1.05rem;
            font-weight: 750;
        }

        .approval-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.65rem;
            margin-top: 0.85rem;
        }

        .approval-field {
            padding: 0.7rem 0.75rem;
            border: 1px solid var(--border);
            border-radius: 10px;
            background: rgba(16, 16, 14, 0.5);
        }

        .approval-field-wide {
            grid-column: 1 / -1;
        }

        .approval-label {
            color: var(--text-muted);
            font-size: 0.65rem;
            font-weight: 750;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .approval-value {
            margin-top: 0.25rem;
            color: var(--text-main);
            font-family:
                Inter,
                system-ui,
                sans-serif;
            font-size: 0.82rem;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }

        .approval-note {
            margin-top: 0.8rem;
            color: var(--text-muted);
            font-size: 0.72rem;
        }

        .ticket-success-card {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 0.25rem 0 1rem 3.2rem;
            padding: 0.8rem 0.9rem;
            border: 1px solid rgba(143, 189, 154, 0.35);
            border-radius: 12px;
            background: rgba(143, 189, 154, 0.07);
        }

        .ticket-success-icon {
            display: grid;
            place-items: center;
            width: 1.9rem;
            height: 1.9rem;
            border-radius: 999px;
            background: var(--success);
            color: #11130f;
            font-weight: 900;
        }

        .ticket-success-label {
            color: var(--success);
            font-size: 0.66rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .ticket-success-id {
            margin-top: 0.08rem;
            color: var(--text-main);
            font-size: 0.96rem;
            font-weight: 800;
        }

        .ticket-success-note {
            margin-top: 0.1rem;
            color: var(--text-muted);
            font-size: 0.7rem;
        }

        .privacy-note {
            margin-top: 0.75rem;
            color: var(--text-muted);
            font-size: 0.73rem;
            line-height: 1.55;
        }

        .support-footer {
            margin-top: 5rem;
            padding: 1.2rem 0 0.25rem;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
            text-align: center;
            font-size: 0.68rem;
        }

        .footer-dot {
            color: var(--accent);
        }

        div[data-testid="stButton"] > button {
            border: 1px solid var(--border-strong) !important;
            border-radius: 10px !important;
            background: var(--bg-panel-soft) !important;
            color: var(--text-main) !important;
            font-family:
                Inter,
                system-ui,
                sans-serif !important;
            font-weight: 650 !important;
        }

        div[data-testid="stButton"] > button:hover {
            border-color: var(--accent) !important;
            color: var(--accent) !important;
        }

        div[data-testid="stButton"] > button[kind="primary"] {
            border-color: var(--accent) !important;
            background: var(--accent) !important;
            color: #17120f !important;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        textarea {
            border-color: var(--border-strong) !important;
            background: var(--bg-input) !important;
            color: var(--text-main) !important;
        }

        input,
        textarea {
            color: var(--text-main) !important;
            caret-color: var(--accent) !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: var(--text-muted) !important;
            opacity: 1 !important;
        }

        /* Fixed chat-input area */
        [data-testid="stBottom"] {
            background: var(--bg-main) !important;
            border-top: 1px solid var(--border) !important;
        }

        [data-testid="stBottom"] > div,
        [data-testid="stBottomBlockContainer"] {
            background: var(--bg-main) !important;
        }

        [data-testid="stBottomBlockContainer"] {
            max-width: 1020px !important;
            padding-top: 0.8rem !important;
            padding-bottom: 1rem !important;
        }

        [data-testid="stChatInput"],
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] div[data-baseweb="textarea"],
        [data-testid="stChatInput"] div[data-baseweb="textarea"] > div,
        [data-testid="stChatInput"] textarea {
            background: var(--bg-input) !important;
        }

        [data-testid="stChatInput"] {
            overflow: hidden;
            border: 1px solid var(--accent) !important;
            border-radius: 13px !important;
            box-shadow: 0 0 0 1px rgba(233, 120, 74, 0.18) !important;
        }

        [data-testid="stChatInput"] textarea {
            color: var(--text-main) !important;
            caret-color: var(--accent) !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--text-muted) !important;
        }

        [data-testid="stChatInput"] button {
            background: transparent !important;
            color: var(--accent) !important;
        }

        @media (max-width: 720px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .app-shell {
                padding: 1.15rem;
            }

            .approval-grid {
                grid-template-columns: 1fr;
            }

            .approval-field-wide {
                grid-column: auto;
            }

            .ticket-success-card {
                margin-left: 0;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "## IT Support"
    )

    if st.session_state.profile_complete:
        profile = st.session_state.user_info

        safe_name = html.escape(
            str(profile.get("user_name", "—"))
        )
        safe_user_id = html.escape(
            str(profile.get("user_id", "—"))
        )
        safe_device = html.escape(
            str(profile.get("device_type", "—")).title()
        )
        safe_os = html.escape(
            str(profile.get("os", "—"))
        )

        profile_card_html = dedent(
            f"""
            <div class="profile-card">
                <div class="profile-label">employee</div>
                <div class="profile-value">{safe_name}</div>
                <div class="profile-label">employee id</div>
                <div class="profile-value">{safe_user_id}</div>
                <div class="profile-label">device</div>
                <div class="profile-value">{safe_device}</div>
                <div class="profile-label">operating system</div>
                <div class="profile-value">{safe_os}</div>
            </div>
            """
        ).strip()

        st.markdown(
            profile_card_html,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='height: 0.75rem'></div>",
            unsafe_allow_html=True,
        )

        if st.button(
            "New thread",
            use_container_width=True,
        ):
            start_new_conversation(
                keep_profile=True
            )
            st.rerun()

    if st.button(
        "Reset workspace",
        use_container_width=True,
    ):
        start_new_conversation(
            keep_profile=False
        )
        st.rerun()

    st.divider()

    st.markdown(
        '<div class="sidebar-heading">// ticket overview</div>',
        unsafe_allow_html=True,
    )

    try:
        status_counts = get_ticket_status_counts()

        all_statuses = [
            ("open", "Open"),
            ("in_progress", "In progress"),
            ("pending_user", "Pending"),
            ("resolved", "Resolved"),
            ("closed", "Closed"),
        ]

        always_visible_statuses = {
            "open",
            "resolved",
        }

        statuses = [
            (status, label)
            for status, label in all_statuses
            if (
                status in always_visible_statuses
                or status_counts.get(status, 0) > 0
            )
        ]

        status_rows = "".join(
            (
                '<div class="status-row">'
                f'<span class="status-dot {status_dot_class(status)}"></span>'
                f'<span class="status-name">{label}</span>'
                f'<span class="status-value">{status_counts.get(status, 0)}</span>'
                "</div>"
            )
            for status, label in statuses
        )

        status_list_html = dedent(
            f"""
            <div class="status-list">
                {status_rows}
            </div>
            """
        ).strip()

        st.markdown(
            status_list_html,
            unsafe_allow_html=True,
        )

        recent_tickets = get_recent_tickets()

        # Hide the recent-ticket expander until at least one real ticket exists.
        # This avoids showing "last 5" in a fresh demo with zero tickets.
        if recent_tickets:
            visible_ticket_count = min(
                len(recent_tickets),
                5,
            )

            with st.expander(
                f"Recent tickets · {visible_ticket_count}",
                expanded=False,
            ):
                ticket_cards: list[str] = []

                for ticket in recent_tickets:
                    ticket_id = html.escape(
                        str(ticket.get("ticket_id", "—"))
                    )
                    title = html.escape(
                        str(ticket.get("title", "Untitled ticket"))
                    )
                    category = html.escape(
                        str(ticket.get("category", "Unknown"))
                    )
                    priority = html.escape(
                        str(ticket.get("priority", "Unknown")).title()
                    )
                    status = str(
                        ticket.get("status", "unknown")
                    )
                    safe_status = html.escape(
                        humanize_status(status)
                    )
                    dot_class = status_dot_class(
                        status
                    )

                    ticket_cards.append(
                        dedent(
                            f"""
                            <div class="recent-ticket-card">
                                <div class="recent-ticket-top">
                                    <span class="recent-ticket-id">
                                        {ticket_id}
                                    </span>
                                    <span class="recent-ticket-status">
                                        <span class="status-dot {dot_class}"></span>
                                        {safe_status}
                                    </span>
                                </div>
                                <div class="recent-ticket-title">
                                    {title}
                                </div>
                                <div class="recent-ticket-meta">
                                    {category} · {priority} priority
                                </div>
                            </div>
                            """
                        ).strip()
                    )

                st.markdown(
                    (
                        '<div class="recent-ticket-list">'
                        + "".join(ticket_cards)
                        + "</div>"
                    ),
                    unsafe_allow_html=True,
                )

        else:
            st.caption(
                "No tickets created in this demo yet."
            )

    except Exception as error:
        st.caption(
            "Ticket information is temporarily unavailable."
        )

        with st.expander(
            "Technical details",
            expanded=False,
        ):
            st.code(
                str(error)
            )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

profile = st.session_state.user_info

session_details = ""

if st.session_state.profile_complete:
    session_details = (
        '<div class="session-line">'
        '<span class="session-online">● session active</span>'
        f"<span>employee {html.escape(str(profile.get('user_id', '—')))}</span>"
        f"<span>{html.escape(str(profile.get('device_type', '—')).title())}</span>"
        f"<span>{html.escape(str(profile.get('os', '—')))}</span>"
        "</div>"
    )

header_html = dedent(
    f"""
    <div class="app-shell">
        <div class="terminal-line">
            <span class="terminal-prompt">$</span>
            it-support --interactive
        </div>
        <h1 class="app-title">IT Support Assistant</h1>
        <div class="app-description">
            Diagnose issues, follow approved troubleshooting steps,
            and escalate unresolved cases through a guided agent workflow.
        </div>
        {session_details}
    </div>
    """
).strip()

st.markdown(
    header_html,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Employee form
# ---------------------------------------------------------------------------

if not st.session_state.profile_complete:
    st.subheader(
        "Initialize support session"
    )

    st.caption(
        "Enter the employee and device context required by the diagnostic agent."
    )

    with st.container(
        border=True,
    ):
        with st.form(
            "employee_profile_form",
            clear_on_submit=False,
        ):
            first_column, second_column = st.columns(
                2
            )

            with first_column:
                user_name = st.text_input(
                    "Full name",
                    placeholder="Mohamad",
                )

            with second_column:
                user_id = st.text_input(
                    "Employee ID",
                    placeholder="EMP-001",
                )

            third_column, fourth_column = st.columns(
                2
            )

            with third_column:
                device_type = st.selectbox(
                    "Device type",
                    [
                        "Laptop",
                        "Desktop",
                        "Mobile",
                        "Tablet",
                        "Other",
                    ],
                )

            with fourth_column:
                operating_system = st.selectbox(
                    "Operating system",
                    [
                        "Windows 11",
                        "Windows 10",
                        "macOS",
                        "Linux",
                        "Android",
                        "iOS",
                        "Other",
                    ],
                )

            submitted = st.form_submit_button(
                "Start support session",
                type="primary",
                use_container_width=True,
            )

        privacy_note_html = dedent(
            """
            <div class="privacy-note">
                Employee information is used only for troubleshooting context
                and support-ticket creation.
            </div>
            """
        ).strip()

        st.markdown(
            privacy_note_html,
            unsafe_allow_html=True,
        )

    if submitted:
        cleaned_name = " ".join(
            user_name.split()
        )

        cleaned_user_id = (
            user_id.strip().upper()
        )

        if not cleaned_name:
            st.error(
                "Please enter your full name."
            )

        elif not cleaned_user_id:
            st.error(
                "Please enter your employee ID."
            )

        else:
            st.session_state.user_info = {
                "user_name": cleaned_name,
                "user_id": cleaned_user_id,
                "device_type": device_type.lower(),
                "os": operating_system,
            }

            st.session_state.profile_complete = True
            st.session_state.last_error = None

            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": (
                        f"Hello {cleaned_name}. "
                        "Describe the IT issue you are experiencing."
                    ),
                }
            ]

            st.rerun()

    st.stop()


# ---------------------------------------------------------------------------
# Welcome examples
# ---------------------------------------------------------------------------

if len(st.session_state.chat_messages) <= 1:
    welcome_panel_html = dedent(
        """
        <div class="welcome-panel">
            <div class="welcome-title">Start with a clear issue description</div>
            <div class="welcome-examples">
                <span>→ My VPN is not connecting.</span>
                <span>→ Microsoft Teams has no audio.</span>
                <span>→ I cannot sign in to my account.</span>
            </div>
        </div>
        """
    ).strip()

    st.markdown(
        welcome_panel_html,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for chat_message in st.session_state.chat_messages:
    role = chat_message.get(
        "role",
        "assistant",
    )

    content = str(
        chat_message.get(
            "content",
            "",
        )
    )

    avatar = "👤" if role == "user" else "🛠️"

    with st.chat_message(
        role,
        avatar=avatar,
    ):
        st.markdown(
            content
        )

    if role == "assistant":
        render_ticket_success_card(
            content
        )


render_error_panel()


# ---------------------------------------------------------------------------
# Approval card
# ---------------------------------------------------------------------------

pending_interrupt = (
    st.session_state.pending_interrupt
)

if pending_interrupt:
    approval_content = str(
        pending_interrupt.get(
            "content",
            "The support agent needs your approval.",
        )
    )

    update_approval = is_ticket_update_approval(
        approval_content
    )

    latest_issue = get_latest_user_message()

    if update_approval:
        update_metadata = parse_update_approval_metadata(
            approval_content
        )

        approval_heading = "Update ticket status"
        approval_action_label = "ticket status update"

        approval_details_html = dedent(
            f"""
            <div class="approval-grid">
                <div class="approval-field">
                    <div class="approval-label">ticket id</div>
                    <div class="approval-value">
                        {html.escape(update_metadata["ticket_id"])}
                    </div>
                </div>
                <div class="approval-field">
                    <div class="approval-label">new status</div>
                    <div class="approval-value">
                        {html.escape(update_metadata["new_status"])}
                    </div>
                </div>
                <div class="approval-field approval-field-wide">
                    <div class="approval-label">update note</div>
                    <div class="approval-value">
                        {html.escape(update_metadata["note"])}
                    </div>
                </div>
            </div>
            """
        ).strip()

        confirm_button_label = "Confirm status update"
        progress_message = "Updating the ticket status..."
        approval_note = (
            "The database status will not change until this action "
            "is confirmed."
        )

    else:
        metadata = parse_approval_metadata(
            approval_content
        )

        approval_heading = "Create support ticket"
        approval_action_label = "ticket creation"

        approval_details_html = dedent(
            f"""
            <div class="approval-grid">
                <div class="approval-field approval-field-wide">
                    <div class="approval-label">ticket title</div>
                    <div class="approval-value">
                        {html.escape(metadata["title"])}
                    </div>
                </div>
                <div class="approval-field">
                    <div class="approval-label">category</div>
                    <div class="approval-value">
                        {html.escape(metadata["category"])}
                    </div>
                </div>
                <div class="approval-field">
                    <div class="approval-label">priority</div>
                    <div class="approval-value">
                        {html.escape(metadata["priority"])}
                    </div>
                </div>
                <div class="approval-field approval-field-wide">
                    <div class="approval-label">issue description</div>
                    <div class="approval-value">
                        {html.escape(latest_issue)}
                    </div>
                </div>
            </div>
            """
        ).strip()

        confirm_button_label = "Confirm and create ticket"
        progress_message = "Creating the support ticket..."
        approval_note = (
            "No ticket will be created until this action is confirmed."
        )

    approval_card_html = dedent(
        f"""
        <div class="approval-card">
            <div class="approval-kicker">approval required</div>
            <div class="approval-title">
                {html.escape(approval_heading)}
            </div>
            {approval_details_html}
            <div class="approval-note">
                {html.escape(approval_note)}
            </div>
        </div>
        """
    ).strip()

    st.markdown(
        approval_card_html,
        unsafe_allow_html=True,
    )

    confirm_column, cancel_column = st.columns(
        2
    )

    with confirm_column:
        if st.button(
            confirm_button_label,
            type="primary",
            use_container_width=True,
        ):
            st.session_state.pending_interrupt = None
            st.session_state.last_error = None

            try:
                with st.spinner(
                    progress_message
                ):
                    response = orchestrator.resume_workflow(
                        graph=st.session_state.graph,
                        feedback=True,
                        session_id=st.session_state.session_id,
                    )

                process_graph_response(
                    response
                )

            except Exception as error:
                st.session_state.last_error = str(
                    error
                )

            # Rerunning makes the sidebar query SQLite again. After an approved
            # update, the count moves from Open to Resolved/In progress/etc.
            st.rerun()

    with cancel_column:
        if st.button(
            "Cancel",
            use_container_width=True,
        ):
            st.session_state.pending_interrupt = None
            st.session_state.last_error = None

            try:
                response = orchestrator.resume_workflow(
                    graph=st.session_state.graph,
                    feedback=False,
                    session_id=st.session_state.session_id,
                )

                process_graph_response(
                    response
                )

            except Exception as error:
                st.session_state.last_error = str(
                    error
                )

            st.rerun()

    # Disable normal chat input while an approval action is pending.
    st.stop()


# ---------------------------------------------------------------------------
# Chat input and orchestrator invocation
# ---------------------------------------------------------------------------

user_message = st.chat_input(
    "Describe the issue, any error message, and when it started..."
)

if user_message:
    cleaned_message = user_message.strip()

    if cleaned_message:
        st.session_state.last_error = None

        # The visible chat stores only the employee's original message.
        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": cleaned_message,
            }
        )

        # Employee context is added only to the hidden orchestrator request.
        orchestrator_message = build_orchestrator_message(
            cleaned_message
        )

        try:
            with st.spinner(
                "Analyzing the issue..."
            ):
                response = orchestrator.invoke_agent(
                    graph=st.session_state.graph,
                    user_message=orchestrator_message,
                    session_id=st.session_state.session_id,
                )

            process_graph_response(
                response
            )

        except Exception as error:
            st.session_state.last_error = str(
                error
            )

        st.rerun()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

footer_html = dedent(
    """
    <div class="support-footer">
        <span class="footer-dot">●</span>
        internal support agent · approved tools only
    </div>
    """
).strip()

st.markdown(
    footer_html,
    unsafe_allow_html=True,
)