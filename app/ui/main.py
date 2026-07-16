"""
Minimal Streamlit chat UI connected to the existing LangGraph orchestrator.

Only this UI file is changed. The tools and agent source files remain untouched.

The small compatibility layer in this file works around three issues in the
current orchestrator implementation:
1. build_graph() references a node named "summarize" that is not registered.
2. route() contains an invalid len(...) expression.
3. execute_tool() converts LangGraph approval interrupts into normal errors.

The UI still uses the orchestrator's agent, tool handlers, invoke_agent(),
resume_workflow(), is_interrupt(), and get_interrupt_metadata() functions.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

import app.agent.orchestrator as orchestrator


st.set_page_config(
    page_title="IT Support Assistant",
    page_icon="🛠️",
    layout="centered",
)


# Expose only the required deadline workflow tools at runtime.
# This does not change any source file under app/agent or app/tools.
orchestrator.tooled_llm = orchestrator.llm.bind_tools(
    [
        orchestrator.classify_and_validate,
        orchestrator.fetch_issue_knowledge,
        orchestrator.open_support_ticket,
    ]
)


def route_for_ui(state: orchestrator.AgentState) -> str:
    """Route assistant tool calls safely without changing orchestrator.py."""
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
    Execute tool requests through the orchestrator's existing handlers.

    GraphInterrupt is re-raised so Streamlit receives the approval request.
    """
    messages = state.get("messages", [])

    if not messages:
        return {
            "messages": [
                AIMessage(
                    content="No tool request was available to execute."
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
        tool_call_id = tool_call.get("id")
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

                case _:
                    result = orchestrator.Error(
                        error="unsupported_tool",
                        message=(
                            f"The deadline UI does not execute "
                            f"the tool '{tool_name}'."
                        ),
                    )

        except GraphInterrupt:
            raise

        except Exception as error:
            result = orchestrator.Error(
                error="tool_execution_error",
                message=str(error),
            )

        if result is None:
            result = orchestrator.Error(
                error="empty_tool_result",
                message=f"Tool '{tool_name}' returned no result.",
            )

        if isinstance(result, orchestrator.Error):
            if result.error in {
                "open_ticket_rejected",
                "update_ticket_rejected",
            }:
                content = (
                    "The user intentionally cancelled the action. "
                    "No ticket was created or updated. "
                    "This was not a technical failure."
                )
            else:
                payload = result.model_dump(mode="json")
                content = str(payload)

                orchestrator.logger.log_tool_error(
                    tool_name,
                    payload,
                )
        else:
            payload = result.model_dump(mode="json")
            content = str(payload)

            orchestrator.logger.log_tool_result(
                tool_name,
                payload,
            )

        tool_messages.append(
            ToolMessage(
                content=content,
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
    """Build a minimal working graph around the existing orchestrator agent."""
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


def initialize_state() -> None:
    """Initialize the Streamlit session."""
    defaults: dict[str, Any] = {
        "graph": None,
        "session_id": str(uuid.uuid4()),
        "user_info": {},
        "profile_complete": False,
        "profile_sent": False,
        "chat_messages": [],
        "pending_interrupt": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.graph is None:
        st.session_state.graph = build_ui_graph()


def start_new_conversation(
    keep_profile: bool = True,
) -> None:
    """Start a new LangGraph thread and optionally keep employee details."""
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
    st.session_state.user_info = saved_profile
    st.session_state.profile_complete = bool(saved_profile)


initialize_state()


def message_to_text(content: Any) -> str:
    """Convert LangChain message content into displayable text."""
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
    """Return the newest assistant message from a graph response."""
    messages = response.get("messages", []) or []

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = message_to_text(message.content).strip()

            if text:
                return text

    return ""


def process_graph_response(
    response: dict[str, Any],
) -> None:
    """Save either a pending approval or the assistant's final reply."""
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
    Add employee context to the first message because the current
    invoke_agent() function accepts only user_message and session_id.
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


st.markdown(
    """
    <style>
        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.2);
        }

        .support-header {
            padding: 1.1rem 1.25rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 14px;
            margin-bottom: 1.25rem;
        }

        .support-status {
            font-size: 0.9rem;
            opacity: 0.75;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("IT Support")

    if st.session_state.profile_complete:
        profile = st.session_state.user_info

        st.write(
            f"**Employee:** {profile.get('user_name', '-')}"
        )
        st.write(
            f"**Employee ID:** {profile.get('user_id', '-')}"
        )
        st.write(
            f"**Device:** {profile.get('device_type', '-')}"
        )
        st.write(
            f"**OS:** {profile.get('os', '-')}"
        )

        st.divider()

        if st.button(
            "New conversation",
            use_container_width=True,
        ):
            start_new_conversation(
                keep_profile=True
            )
            st.rerun()

    if st.button(
        "Reset everything",
        use_container_width=True,
    ):
        start_new_conversation(
            keep_profile=False
        )
        st.rerun()


st.markdown(
    """
    <div class="support-header">
        <h2 style="margin: 0;">IT Support Assistant</h2>
        <div class="support-status">
            ● Online · Guided troubleshooting and ticket escalation
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if not st.session_state.profile_complete:
    st.subheader("Employee details")
    st.caption(
        "Enter the information required by the diagnostic workflow."
    )

    with st.form(
        "employee_profile_form",
        clear_on_submit=False,
    ):
        user_name = st.text_input(
            "Name",
            placeholder="Mohamad",
        )

        user_id = st.text_input(
            "Employee ID",
            placeholder="EMP-001",
        )

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

    if submitted:
        cleaned_name = " ".join(
            user_name.split()
        )

        cleaned_user_id = (
            user_id.strip().upper()
        )

        if not cleaned_name:
            st.error(
                "Please enter your name."
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


for chat_message in st.session_state.chat_messages:
    with st.chat_message(
        chat_message["role"]
    ):
        st.markdown(
            chat_message["content"]
        )


pending_interrupt = (
    st.session_state.pending_interrupt
)

if pending_interrupt:
    st.warning(
        pending_interrupt.get(
            "content",
            "The agent needs your approval.",
        )
    )

    confirm_column, cancel_column = (
        st.columns(2)
    )

    with confirm_column:
        if st.button(
            "Confirm",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.pending_interrupt = None

            try:
                with st.spinner(
                    "Executing approved action..."
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
                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "The approved action could not be completed. "
                            f"Details: {error}"
                        ),
                    }
                )

            st.rerun()

    with cancel_column:
        if st.button(
            "Cancel",
            use_container_width=True,
        ):
            st.session_state.pending_interrupt = None

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
                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "The cancellation could not be processed. "
                            f"Details: {error}"
                        ),
                    }
                )

            st.rerun()

    st.stop()


user_message = st.chat_input(
    "Describe your IT issue..."
)

if user_message:
    cleaned_message = user_message.strip()

    if cleaned_message:
        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": cleaned_message,
            }
        )

        orchestrator_message = (
            build_orchestrator_message(
                cleaned_message
            )
        )

        try:
            with st.spinner(
                "Analyzing your issue..."
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
            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "The support workflow encountered an error. "
                        f"Details: {error}"
                    ),
                }
            )

        st.rerun()