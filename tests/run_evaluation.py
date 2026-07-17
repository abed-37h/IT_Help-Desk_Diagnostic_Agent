"""
Run the required 20 conversation evaluations for the IT Help-Desk agent.

This file belongs in:
    tests/run_evaluation.py

It does not modify the production agent, tools, state model, or database schema.

What it does:
- loads tests/cases/evaluation_cases.json;
- runs each conversation against the teammate's existing agent and handlers;
- supports LangGraph confirmation interrupts;
- records expected and actual results;
- calculates the metrics required by the project proposal;
- backs up and restores app/data/tickets.db so evaluation does not damage demo data.

The test graph contains only a compatibility wiring layer because the current
production build_graph() refers to an unknown node named "summarize". The
actual agent function and tool handlers still come from orchestrator.py.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
from operator import index
import os
import shutil
import sys
import uuid
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any
import time

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


# ---------------------------------------------------------------------------
# Paths and database protection
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CASES_PATH = (
    PROJECT_ROOT
    / "tests"
    / "cases"
    / "evaluation_cases.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "tests"
    / "results"
)

RESULTS_JSON_PATH = (
    RESULTS_DIR
    / "evaluation_results.json"
)

SUMMARY_CSV_PATH = (
    RESULTS_DIR
    / "evaluation_summary.csv"
)

TRACES_JSON_PATH = (
    RESULTS_DIR
    / "selected_traces.json"
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "tickets.db"
)

DATABASE_BACKUP_PATH = (
    RESULTS_DIR
    / "_tickets_before_evaluation.db"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DATABASE_EXISTED_BEFORE_TESTS = (
    DATABASE_PATH.exists()
)


def backup_database() -> None:
    """Back up the user's current ticket database before importing the agent."""
    if DATABASE_BACKUP_PATH.exists():
        DATABASE_BACKUP_PATH.unlink()

    if DATABASE_PATH.exists():
        shutil.copy2(
            DATABASE_PATH,
            DATABASE_BACKUP_PATH,
        )


def restore_database() -> None:
    """Restore the database that existed before the evaluation run."""
    try:
        if DATABASE_BACKUP_PATH.exists():
            shutil.copy2(
                DATABASE_BACKUP_PATH,
                DATABASE_PATH,
            )

            DATABASE_BACKUP_PATH.unlink()

        elif (
            not DATABASE_EXISTED_BEFORE_TESTS
            and DATABASE_PATH.exists()
        ):
            DATABASE_PATH.unlink()

    except Exception as error:
        print(
            "[WARNING] Could not restore the original database: "
            f"{error}"
        )


backup_database()
atexit.register(
    restore_database
)


# Importing the orchestrator initializes its SQLite database.
import app.agent.orchestrator as orchestrator  # noqa: E402
from app.agent.state import AgentState  # noqa: E402
from app.data.init_db import connect  # noqa: E402


# ---------------------------------------------------------------------------
# Evaluation runtime configuration
# ---------------------------------------------------------------------------

load_dotenv(
    PROJECT_ROOT / ".env"
)


def safe_log_llm_error(
    function_name: str,
    error_code: str,
    message: str,
) -> None:
    """Prevent a logger signature mismatch from hiding the real LLM error."""
    print(
        "[LLM ERROR] "
        f"function={function_name} "
        f"code={error_code} "
        f"message={message}"
    )


def safe_log_tool_error(
    tool_name: str,
    payload: Any,
) -> None:
    """Print tool errors without crashing the evaluation runner."""
    print(
        "[TOOL ERROR] "
        f"tool={tool_name} "
        f"payload={payload}"
    )


orchestrator.logger.log_llm_error = (
    safe_log_llm_error
)

orchestrator.logger.log_tool_error = (
    safe_log_tool_error
)


# Use the same default model as the working Streamlit integration.
# It can be overridden without editing code:
#     $env:EVAL_MODEL="another-model"
EVALUATION_MODEL = os.getenv(
    "EVAL_MODEL",
    "gemini-3.1-flash-lite",
)

orchestrator.llm = ChatGoogleGenerativeAI(
    model=EVALUATION_MODEL,
    temperature=0,
)

# Preserve the teammate's complete tool set.
orchestrator.tooled_llm = (
    orchestrator.llm.bind_tools(
        orchestrator.TOOLS
    )
)


# ---------------------------------------------------------------------------
# Per-case trace collector
# ---------------------------------------------------------------------------

ACTIVE_TRACE: dict[str, Any] = {}


def reset_trace() -> None:
    """Reset observable execution data for one evaluation case."""
    ACTIVE_TRACE.clear()

    ACTIVE_TRACE.update(
        {
            "requested_tools": [],
            "executed_tools": [],
            "interrupts": [],
            "turns": [],
            "errors": [],
            "approved_resumes": 0,
            "rejected_resumes": 0,
        }
    )


# ---------------------------------------------------------------------------
# Test graph
# ---------------------------------------------------------------------------


def route_for_evaluation(
    state: AgentState,
) -> str:
    """
    Continue to the tool node only when the assistant requested a tool.

    This mirrors the working UI loop and avoids changing orchestrator.py.
    """
    messages = state.get(
        "messages",
        [],
    )

    if not messages:
        return "end"

    last_message = messages[-1]

    if getattr(
        last_message,
        "tool_calls",
        None,
    ):
        return "tools"

    return "end"


def serialize_model(
    value: Any,
) -> dict[str, Any]:
    """Convert a Pydantic result into JSON-safe data."""
    if hasattr(
        value,
        "model_dump",
    ):
        return value.model_dump(
            mode="json"
        )

    return {
        "value": str(value)
    }


def execute_tools_for_evaluation(
    state: AgentState,
) -> AgentState:
    """
    Execute the teammate's tools through the teammate's existing handlers.

    GraphInterrupt must be re-raised so the runner can test confirmation.
    """
    messages = state.get(
        "messages",
        [],
    )

    if not messages:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "No tool request was available."
                    )
                )
            ]
        }

    last_message = messages[-1]
    tool_calls = (
        getattr(
            last_message,
            "tool_calls",
            [],
        )
        or []
    )

    update: dict[str, Any] = {}
    tool_messages: list[ToolMessage] = []

    for tool_call in tool_calls:
        tool_name = tool_call.get(
            "name",
            "unknown_tool",
        )

        tool_args = (
            tool_call.get(
                "args",
                {},
            )
            or {}
        )

        tool_call_id = (
            tool_call.get("id")
            or f"eval-{uuid.uuid4()}"
        )

        ACTIVE_TRACE[
            "requested_tools"
        ].append(
            tool_name
        )

        result: Any = None

        try:
            match tool_name:
                case "classify_and_validate":
                    result = (
                        orchestrator.handle_classification(
                            state,
                            tool_args,
                        )
                    )

                    update[
                        "classification_result"
                    ] = serialize_model(
                        result
                    )

                    if not isinstance(
                        result,
                        orchestrator.Error,
                    ):
                        update[
                            "valid_user_info"
                        ] = result.is_valid

                        update[
                            "issue_id"
                        ] = (
                            result.issue_id
                            or None
                        )

                case "fetch_issue_knowledge":
                    effective_state = {
                        **state,
                        **update,
                    }

                    result = (
                        orchestrator.handle_fetch_kb(
                            effective_state,
                            tool_args,
                        )
                    )

                    update[
                        "knowledge_result"
                    ] = serialize_model(
                        result
                    )

                    if not isinstance(
                        result,
                        orchestrator.Error,
                    ):
                        category = (
                            result.category
                        )

                        if hasattr(
                            category,
                            "value",
                        ):
                            category = (
                                category.value
                            )

                        update[
                            "category"
                        ] = category

                        update[
                            "severity"
                        ] = result.severity

                        update[
                            "steps"
                        ] = result.steps

                        update[
                            "escalate"
                        ] = result.escalate

                case "open_support_ticket":
                    effective_state = {
                        **state,
                        **update,
                    }

                    result = (
                        orchestrator.handle_open_ticket(
                            effective_state,
                            tool_args,
                        )
                    )

                    update[
                        "open_ticket_result"
                    ] = serialize_model(
                        result
                    )

                    if not isinstance(
                        result,
                        orchestrator.Error,
                    ):
                        update[
                            "ticket_id"
                        ] = result.ticket_id

                        update[
                            "ticket_status"
                        ] = result.status

                case "update_support_ticket":
                    effective_state = {
                        **state,
                        **update,
                    }

                    result = (
                        orchestrator.handle_update_ticket(
                            effective_state,
                            tool_args,
                        )
                    )

                    update[
                        "update_ticket_result"
                    ] = serialize_model(
                        result
                    )

                    if not isinstance(
                        result,
                        orchestrator.Error,
                    ):
                        update[
                            "ticket_id"
                        ] = result.ticket_id

                        update[
                            "ticket_status"
                        ] = result.new_status

                case "generate_report":
                    effective_state = {
                        **state,
                        **update,
                    }

                    result = (
                        orchestrator.handle_generate_report(
                            effective_state,
                            tool_args,
                        )
                    )

                    update[
                        "generate_report_result"
                    ] = serialize_model(
                        result
                    )

                    if not isinstance(
                        result,
                        orchestrator.Error,
                    ):
                        update[
                            "report"
                        ] = serialize_model(
                            result
                        )

                case _:
                    result = orchestrator.Error(
                        error="unknown_tool",
                        message=(
                            "Unsupported tool: "
                            f"{tool_name}"
                        ),
                    )

        except GraphInterrupt:
            # This is expected for open/update confirmation gates.
            raise

        except Exception as error:
            ACTIVE_TRACE[
                "errors"
            ].append(
                {
                    "tool": tool_name,
                    "error": str(error),
                }
            )

            result = orchestrator.Error(
                error="tool_execution_error",
                message=str(error),
            )

        if result is None:
            result = orchestrator.Error(
                error="empty_tool_result",
                message=(
                    f"{tool_name} returned no result."
                ),
            )

        result_payload = serialize_model(
            result
        )

        if not isinstance(
            result,
            orchestrator.Error,
        ):
            ACTIVE_TRACE[
                "executed_tools"
            ].append(
                tool_name
            )

        tool_messages.append(
            ToolMessage(
                content=json.dumps(
                    result_payload,
                    ensure_ascii=False,
                ),
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )

    return {
        "messages": tool_messages,
        **update,
    }


def build_evaluation_graph():
    """
    Build a working graph from the teammate's agent and handler functions.

    This corrects only graph wiring inside the test package.
    """
    builder = StateGraph(
        AgentState
    )

    builder.add_node(
        "agent",
        orchestrator.agent,
    )

    builder.add_node(
        "tools",
        execute_tools_for_evaluation,
    )

    builder.add_edge(
        START,
        "agent",
    )

    builder.add_conditional_edges(
        "agent",
        route_for_evaluation,
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
# Database isolation and inspection
# ---------------------------------------------------------------------------


def clear_test_database() -> None:
    """Start every conversation case with an empty ticket database."""
    with connect() as connection:
        connection.execute(
            "DELETE FROM ticket_history"
        )

        connection.execute(
            "DELETE FROM tickets"
        )

        connection.commit()


def database_snapshot() -> dict[str, Any]:
    """Return the ticket and history state used by evaluation assertions."""
    with connect() as connection:
        ticket_rows = connection.execute(
            """
            SELECT
                ticket_id,
                title,
                category,
                priority,
                status
            FROM tickets
            ORDER BY created_at, ticket_id
            """
        ).fetchall()

        history_rows = connection.execute(
            """
            SELECT
                ticket_id,
                old_status,
                new_status,
                changed_by,
                note
            FROM ticket_history
            ORDER BY changed_at
            """
        ).fetchall()

    tickets = [
        dict(row)
        for row in ticket_rows
    ]

    history = [
        dict(row)
        for row in history_rows
    ]

    return {
        "ticket_count": len(tickets),
        "history_count": len(history),
        "tickets": tickets,
        "history": history,
        "latest_status": (
            tickets[-1]["status"]
            if tickets
            else None
        ),
    }


# ---------------------------------------------------------------------------
# JSON and state utilities
# ---------------------------------------------------------------------------


def load_cases() -> list[dict[str, Any]]:
    """Load the required documented conversations."""
    if not CASES_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation cases not found: {CASES_PATH}"
        )

    with CASES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        cases = json.load(file)

    if not isinstance(
        cases,
        list,
    ):
        raise ValueError(
            "evaluation_cases.json must contain a list."
        )

    return cases


def normalize_value(
    value: Any,
) -> Any:
    """Normalize enums and strings for stable comparisons."""
    if isinstance(
        value,
        Enum,
    ):
        return normalize_value(
            value.value
        )

    if isinstance(
        value,
        str,
    ):
        return (
            value.strip()
            .casefold()
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): normalize_value(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            normalize_value(item)
            for item in value
        ]

    return value


def compare_subset(
    actual: Any,
    expected: Any,
    path: str = "state",
) -> list[str]:
    """
    Compare an expected subset against actual state.

    Extra actual fields do not fail a test.
    """
    failures: list[str] = []

    if isinstance(
        expected,
        dict,
    ):
        if not isinstance(
            actual,
            dict,
        ):
            return [
                (
                    f"{path}: expected an object, "
                    f"got {type(actual).__name__}"
                )
            ]

        for key, expected_value in expected.items():
            actual_value = actual.get(
                key
            )

            failures.extend(
                compare_subset(
                    actual_value,
                    expected_value,
                    f"{path}.{key}",
                )
            )

        return failures

    if normalize_value(
        actual
    ) != normalize_value(
        expected
    ):
        failures.append(
            (
                f"{path}: expected "
                f"{expected!r}, got {actual!r}"
            )
        )

    return failures


def message_content_to_text(
    content: Any,
) -> str:
    """Convert LangChain message content into plain text."""
    if isinstance(
        content,
        str,
    ):
        return content

    if isinstance(
        content,
        list,
    ):
        parts: list[str] = []

        for item in content:
            if isinstance(
                item,
                str,
            ):
                parts.append(
                    item
                )

            elif isinstance(
                item,
                dict,
            ):
                text = item.get(
                    "text"
                )

                if text:
                    parts.append(
                        str(text)
                    )

        return "\n".join(
            parts
        )

    return str(
        content
    )


def last_assistant_response(
    state: dict[str, Any],
) -> str:
    """Return the latest assistant text from graph state."""
    for message in reversed(
        state.get(
            "messages",
            [],
        )
    ):
        if isinstance(
            message,
            AIMessage,
        ):
            text = message_content_to_text(
                message.content
            ).strip()

            if text:
                return text

    return ""


def json_safe(
    value: Any,
) -> Any:
    """Convert graph state and trace values into JSON-safe structures."""
    if isinstance(
        value,
        Enum,
    ):
        return value.value

    if hasattr(
        value,
        "model_dump",
    ):
        return value.model_dump(
            mode="json"
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
            if key != "messages"
        }

    if isinstance(
        value,
        list,
    ):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ) or value is None:
        return value

    return str(
        value
    )


def selected_state(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Save only meaningful workflow fields in evaluation output."""
    keys = [
        "user_info",
        "valid_user_info",
        "issue_id",
        "category",
        "severity",
        "steps",
        "escalate",
        "ticket_id",
        "ticket_status",
        "workflow_stage",
        "remaining_iterations",
        "classification_result",
        "knowledge_result",
        "open_ticket_result",
        "update_ticket_result",
        "generate_report_result",
        "pending_confirmation",
        "pending_action",
        "fallback_triggered",
        "fallback_reason",
    ]

    return {
        key: json_safe(
            state.get(key)
        )
        for key in keys
        if key in state
    }


# ---------------------------------------------------------------------------
# Evaluation checks
# ---------------------------------------------------------------------------


def check_response_terms(
    response: str,
    expected_terms: list[str],
) -> bool:
    """Return True when at least one expected term appears."""
    if not expected_terms:
        return True

    normalized_response = (
        response.casefold()
    )

    return any(
        term.casefold()
        in normalized_response
        for term in expected_terms
    )


def detect_fallback(
    state: dict[str, Any],
    response: str,
    requested_tools: list[str],
) -> bool:
    """
    Detect either explicit fallback state or a safe unsupported-request reply.
    """
    if state.get(
        "fallback_triggered",
        False,
    ):
        return True

    limitation_terms = [
        "outside",
        "cannot",
        "can't",
        "not able",
        "it support",
        "help desk",
        "contact",
        "human",
        "technician",
        "hr",
    ]

    response_is_limitation = any(
        term in response.casefold()
        for term in limitation_terms
    )

    return (
        response_is_limitation
        and not requested_tools
    )


def evaluate_case_result(
    case: dict[str, Any],
    final_state: dict[str, Any],
    final_response: str,
    database: dict[str, Any],
) -> dict[str, Any]:
    """Compare expected behavior with the observed case result."""
    expected = case.get(
        "expected",
        {},
    )

    failures: list[str] = []

    requested_tools = list(
        ACTIVE_TRACE[
            "requested_tools"
        ]
    )

    executed_tools = list(
        ACTIVE_TRACE[
            "executed_tools"
        ]
    )

    requested_tool_set = set(
        requested_tools
    )

    executed_tool_set = set(
        executed_tools
    )

    required_tools = set(
        expected.get(
            "required_tools",
            [],
        )
    )

    forbidden_tools = set(
        expected.get(
            "forbidden_tools",
            [],
        )
    )

    forbidden_executed_tools = set(
        expected.get(
            "forbidden_executed_tools",
            [],
        )
    )

    missing_required_tools = (
        required_tools
        - requested_tool_set
    )

    requested_forbidden_tools = (
        forbidden_tools
        & requested_tool_set
    )

    executed_forbidden_tools = (
        forbidden_executed_tools
        & executed_tool_set
    )

    if missing_required_tools:
        failures.append(
            "Missing required tools: "
            + ", ".join(
                sorted(
                    missing_required_tools
                )
            )
        )

    if requested_forbidden_tools:
        failures.append(
            "Forbidden tools were requested: "
            + ", ".join(
                sorted(
                    requested_forbidden_tools
                )
            )
        )

    if executed_forbidden_tools:
        failures.append(
            "Forbidden tools were executed: "
            + ", ".join(
                sorted(
                    executed_forbidden_tools
                )
            )
        )

    expected_state = expected.get(
    "state_subset",
    {},
)

# Some optional boolean fields in AgentState are not written until
# their condition occurs. Treat an absent field as its normal default
# rather than incorrectly interpreting it as a failed behavior.
    state_with_defaults = {
    "valid_user_info": False,
    "pending_confirmation": False,
    "fallback_triggered": False,
    **final_state,
}

    failures.extend(
    compare_subset(
        state_with_defaults,
        expected_state,
    )
)

    interrupt_count = len(
        ACTIVE_TRACE[
            "interrupts"
        ]
    )

    if "expected_interrupt_count" in expected:
        expected_interrupt_count = int(
            expected[
                "expected_interrupt_count"
            ]
        )

        if interrupt_count != expected_interrupt_count:
            failures.append(
                (
                    "Interrupt count: expected "
                    f"{expected_interrupt_count}, "
                    f"got {interrupt_count}"
                )
            )

    elif "expected_interrupt" in expected:
        expected_interrupt = bool(
            expected[
                "expected_interrupt"
            ]
        )

        actual_interrupt = (
            interrupt_count > 0
        )

        if actual_interrupt != expected_interrupt:
            failures.append(
                (
                    "Confirmation interrupt: expected "
                    f"{expected_interrupt}, "
                    f"got {actual_interrupt}"
                )
            )

    response_terms = expected.get(
        "response_contains_any",
        [],
    )

    response_check_passed = (
        check_response_terms(
            final_response,
            response_terms,
        )
    )

    if not response_check_passed:
        failures.append(
            (
                "Final response did not contain any "
                "expected term: "
                + ", ".join(
                    response_terms
                )
            )
        )

    database_expected = expected.get(
        "database",
        {},
    )

    if "tickets_created" in database_expected:
        expected_created = int(
            database_expected[
                "tickets_created"
            ]
        )

        if (
            database[
                "ticket_count"
            ]
            != expected_created
        ):
            failures.append(
                (
                    "Tickets created: expected "
                    f"{expected_created}, got "
                    f"{database['ticket_count']}"
                )
            )

    if "tickets_updated" in database_expected:
        expected_updated = int(
            database_expected[
                "tickets_updated"
            ]
        )

        if (
            database[
                "history_count"
            ]
            != expected_updated
        ):
            failures.append(
                (
                    "Ticket updates: expected "
                    f"{expected_updated}, got "
                    f"{database['history_count']}"
                )
            )

    if "final_status" in database_expected:
        expected_status = (
            database_expected[
                "final_status"
            ]
        )

        actual_status = database.get(
            "latest_status"
        )

        if normalize_value(
            actual_status
        ) != normalize_value(
            expected_status
        ):
            failures.append(
                (
                    "Final database status: expected "
                    f"{expected_status!r}, got "
                    f"{actual_status!r}"
                )
            )

    approved_actions = int(
        ACTIVE_TRACE[
            "approved_resumes"
        ]
    )

    state_changing_executions = sum(
        1
        for tool_name in executed_tools
        if tool_name in {
            "open_support_ticket",
            "update_support_ticket",
        }
    )

    unsafe_actions = max(
        0,
        (
            state_changing_executions
            - approved_actions
        ),
    )

    expected_unsafe_actions = int(
        expected.get(
            "unsafe_actions",
            0,
        )
    )

    if unsafe_actions != expected_unsafe_actions:
        failures.append(
            (
                "Unsafe actions: expected "
                f"{expected_unsafe_actions}, "
                f"got {unsafe_actions}"
            )
        )

    fallback_expected = expected.get(
        "fallback_expected"
    )

    fallback_actual = detect_fallback(
        final_state,
        final_response,
        requested_tools,
    )

    fallback_correct: bool | None = None

    if fallback_expected is not None:
        fallback_correct = (
            bool(fallback_expected)
            == fallback_actual
        )

        if not fallback_correct:
            failures.append(
                (
                    "Fallback decision: expected "
                    f"{bool(fallback_expected)}, "
                    f"got {fallback_actual}"
                )
            )

    tool_selection_correct = not (
        missing_required_tools
        or requested_forbidden_tools
        or executed_forbidden_tools
    )

    return {
        "passed": not failures,
        "failures": failures,
        "tool_selection_correct": tool_selection_correct,
        "fallback_expected": fallback_expected,
        "fallback_actual": fallback_actual,
        "fallback_correct": fallback_correct,
        "unsafe_actions": unsafe_actions,
        "response_check_passed": response_check_passed,
    }


# ---------------------------------------------------------------------------
# Conversation execution
# ---------------------------------------------------------------------------


def invoke_user_turn(
    graph: Any,
    text: str,
    thread_id: str,
) -> dict[str, Any]:
    """Send one user message into the test graph."""
    return graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=text
                )
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id
            },
            "recursion_limit": 50,
        },
    )


def resume_turn(
    graph: Any,
    approved: bool,
    thread_id: str,
) -> dict[str, Any]:
    """Resume a confirmation interrupt."""
    if approved:
        ACTIVE_TRACE[
            "approved_resumes"
        ] += 1
    else:
        ACTIVE_TRACE[
            "rejected_resumes"
        ] += 1

    return graph.invoke(
        Command(
            resume=approved
        ),
        config={
            "configurable": {
                "thread_id": thread_id
            },
            "recursion_limit": 50,
        },
    )


def run_case(
    case: dict[str, Any],
) -> dict[str, Any]:
    """Execute one complete documented conversation."""
    clear_test_database()
    reset_trace()

    graph = build_evaluation_graph()
    thread_id = (
        "eval-"
        + case["id"].lower()
        + "-"
        + str(uuid.uuid4())
    )

    state: dict[str, Any] = {}
    pending_interrupt = False

    for turn_index, turn in enumerate(
        case.get(
            "turns",
            [],
        ),
        start=1,
    ):
        turn_record: dict[str, Any] = {
            "turn": turn_index,
        }

        try:
            if "user" in turn:
                user_text = str(
                    turn["user"]
                )

                turn_record[
                    "type"
                ] = "user"

                turn_record[
                    "input"
                ] = user_text

                state = invoke_user_turn(
                    graph,
                    user_text,
                    thread_id,
                )

            elif "resume_with" in turn:
                approved = bool(
                    turn[
                        "resume_with"
                    ]
                )

                turn_record[
                    "type"
                ] = "resume"

                turn_record[
                    "approved"
                ] = approved

                if not pending_interrupt:
                    ACTIVE_TRACE[
                        "errors"
                    ].append(
                        {
                            "turn": turn_index,
                            "error": (
                                "The case attempted to resume "
                                "without a pending interrupt."
                            ),
                        }
                    )

                state = resume_turn(
                    graph,
                    approved,
                    thread_id,
                )

            else:
                ACTIVE_TRACE[
                    "errors"
                ].append(
                    {
                        "turn": turn_index,
                        "error": (
                            "Turn has neither 'user' "
                            "nor 'resume_with'."
                        ),
                    }
                )

                continue

            pending_interrupt = (
                "__interrupt__"
                in state
            )

            if pending_interrupt:
                interrupt_value = (
                    state[
                        "__interrupt__"
                    ][0].value
                )

                ACTIVE_TRACE[
                    "interrupts"
                ].append(
                    json_safe(
                        interrupt_value
                    )
                )

                turn_record[
                    "interrupt"
                ] = json_safe(
                    interrupt_value
                )

            assistant_text = (
                last_assistant_response(
                    state
                )
            )

            if assistant_text:
                turn_record[
                    "assistant"
                ] = assistant_text

            turn_record[
                "state"
            ] = selected_state(
                state
            )

        except Exception as error:
            ACTIVE_TRACE[
                "errors"
            ].append(
                {
                    "turn": turn_index,
                    "error": str(error),
                }
            )

            turn_record[
                "error"
            ] = str(error)

        ACTIVE_TRACE[
            "turns"
        ].append(
            turn_record
        )

    final_response = (
        last_assistant_response(
            state
        )
    )

    final_database = (
        database_snapshot()
    )

    checks = evaluate_case_result(
        case,
        state,
        final_response,
        final_database,
    )

    return {
        "id": case["id"],
        "name": case["name"],
        "requirement": case.get(
            "requirement"
        ),
        "objective": case.get(
            "objective"
        ),
        "passed": checks[
            "passed"
        ],
        "failures": checks[
            "failures"
        ],
        "expected": case.get(
            "expected",
            {},
        ),
        "actual": {
            "requested_tools": (
                ACTIVE_TRACE[
                    "requested_tools"
                ]
            ),
            "executed_tools": (
                ACTIVE_TRACE[
                    "executed_tools"
                ]
            ),
            "interrupt_count": len(
                ACTIVE_TRACE[
                    "interrupts"
                ]
            ),
            "final_state": selected_state(
                state
            ),
            "final_response": final_response,
            "database": final_database,
            "unsafe_actions": checks[
                "unsafe_actions"
            ],
            "fallback_actual": checks[
                "fallback_actual"
            ],
            "errors": (
                ACTIVE_TRACE[
                    "errors"
                ]
            ),
        },
        "metrics": {
            "task_completed": checks[
                "passed"
            ],
            "tool_selection_correct": checks[
                "tool_selection_correct"
            ],
            "fallback_correct": checks[
                "fallback_correct"
            ],
            "unsafe_actions": checks[
                "unsafe_actions"
            ],
        },
        "trace": deepcopy(
            ACTIVE_TRACE[
                "turns"
            ]
        ),
    }


# ---------------------------------------------------------------------------
# Reports and required metrics
# ---------------------------------------------------------------------------


def build_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate the metrics required by the proposal."""
    total = len(
        results
    )

    completed = sum(
        1
        for result in results
        if result[
            "metrics"
        ][
            "task_completed"
        ]
    )

    tool_correct = sum(
        1
        for result in results
        if result[
            "metrics"
        ][
            "tool_selection_correct"
        ]
    )

    fallback_results = [
        result
        for result in results
        if result[
            "metrics"
        ][
            "fallback_correct"
        ]
        is not None
    ]

    fallback_correct = sum(
        1
        for result in fallback_results
        if result[
            "metrics"
        ][
            "fallback_correct"
        ]
    )

    unsafe_actions = sum(
        int(
            result[
                "metrics"
            ][
                "unsafe_actions"
            ]
        )
        for result in results
    )

    return {
        "model": EVALUATION_MODEL,
        "total_cases": total,
        "passed_cases": completed,
        "failed_cases": total - completed,
        "task_completion_rate": (
            completed / total
            if total
            else 0.0
        ),
        "correct_tool_selection_rate": (
            tool_correct / total
            if total
            else 0.0
        ),
        "fallback_cases": len(
            fallback_results
        ),
        "fallback_accuracy": (
            fallback_correct
            / len(fallback_results)
            if fallback_results
            else None
        ),
        "unsafe_or_invalid_actions_executed": (
            unsafe_actions
        ),
    }


def save_results(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """Save expected/actual results, summary CSV, and selected traces."""
    payload = {
        "summary": summary,
        "cases": results,
    }

    RESULTS_JSON_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with SUMMARY_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "name",
                "requirement",
                "passed",
                "tool_selection_correct",
                "fallback_correct",
                "unsafe_actions",
                "failure_count",
            ],
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "id": result["id"],
                    "name": result["name"],
                    "requirement": result[
                        "requirement"
                    ],
                    "passed": result[
                        "passed"
                    ],
                    "tool_selection_correct": result[
                        "metrics"
                    ][
                        "tool_selection_correct"
                    ],
                    "fallback_correct": result[
                        "metrics"
                    ][
                        "fallback_correct"
                    ],
                    "unsafe_actions": result[
                        "metrics"
                    ][
                        "unsafe_actions"
                    ],
                    "failure_count": len(
                        result[
                            "failures"
                        ]
                    ),
                }
            )

    selected_ids = {
        "EVAL-007",
        "EVAL-009",
        "EVAL-015",
        "EVAL-019",
    }

    selected = [
        result
        for result in results
        if (
            result["id"]
            in selected_ids
            or not result[
                "passed"
            ]
        )
    ]

    TRACES_JSON_PATH.write_text(
        json.dumps(
            selected,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def print_summary(
    summary: dict[str, Any],
) -> None:
    """Print the evaluation metrics clearly."""
    print()
    print("=" * 72)
    print("EVALUATION SUMMARY")
    print("=" * 72)

    print(
        f"Model: {summary['model']}"
    )

    print(
        "Cases: "
        f"{summary['passed_cases']} passed / "
        f"{summary['total_cases']} total"
    )

    print(
        "Task-completion rate: "
        f"{summary['task_completion_rate']:.2%}"
    )

    print(
        "Correct tool-selection rate: "
        f"{summary['correct_tool_selection_rate']:.2%}"
    )

    fallback_accuracy = summary[
        "fallback_accuracy"
    ]

    if fallback_accuracy is None:
        fallback_text = "N/A"
    else:
        fallback_text = (
            f"{fallback_accuracy:.2%}"
        )

    print(
        "Fallback accuracy: "
        f"{fallback_text}"
    )

    print(
        "Unsafe or invalid actions executed: "
        f"{summary['unsafe_or_invalid_actions_executed']}"
    )

    print()
    print(
        f"Detailed results: {RESULTS_JSON_PATH}"
    )

    print(
        f"Summary CSV: {SUMMARY_CSV_PATH}"
    )

    print(
        f"Selected traces: {TRACES_JSON_PATH}"
    )


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the IT Help-Desk conversation "
            "evaluation suite."
        )
    )

    parser.add_argument(
        "--case",
        dest="case_id",
        help=(
            "Run only one case, for example "
            "--case EVAL-001"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_cases = load_cases()

    if args.case_id:
        requested_id = (
            args.case_id.strip()
            .upper()
        )

        cases = [
            case
            for case in all_cases
            if case[
                "id"
            ].upper()
            == requested_id
        ]

        if not cases:
            raise SystemExit(
                f"Unknown case ID: {requested_id}"
            )

    else:
        cases = all_cases

    print(
        "Running "
        f"{len(cases)} evaluation conversation(s) "
        f"with {EVALUATION_MODEL}."
    )

    results: list[dict[str, Any]] = []

    try:
        for index, case in enumerate(
            cases,
            start=1,
        ):
            print()
            print(
                f"[{index}/{len(cases)}] "
                f"{case['id']} — {case['name']}"
            )

            result = run_case(
                case
            )

            results.append(
                result
            )
            # Avoid Gemini free-tier rate-limit errors between conversations.
            if index < len(cases):
                print("  Waiting 65 seconds before the next case...")
                time.sleep(65)

            if result[
                "passed"
            ]:
                print(
                    "  PASS"
                )

            else:
                print(
                    "  FAIL"
                )

                for failure in result[
                    "failures"
                ]:
                    print(
                        f"    - {failure}"
                    )

        summary = build_summary(
            results
        )

        save_results(
            results,
            summary,
        )

        print_summary(
            summary
        )

    finally:
        restore_database()


if __name__ == "__main__":
    main()