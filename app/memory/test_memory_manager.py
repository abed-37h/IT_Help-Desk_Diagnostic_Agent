from pathlib import Path
from tempfile import TemporaryDirectory

from app.memory.memory_manager import MemoryManager


LONG_TERM_VALUES = {
    "preferred_language": "English",
    "operating_system": "Windows 11",
    "department": "Engineering",
    "common_device": "Dell Latitude laptop",
    "previous_ticket_ids": ["TCK-0003", "TCK-0008"],
}


def test_short_term_and_working_memory(db_path: Path) -> None:
    memory = MemoryManager(db_path=db_path)

    memory.set_user(
        user_name="Mohamad",
        user_id="employee_1025",
    )

    # Test short-term conversation memory.
    memory.add_user_message("My VPN is not connecting.")

    memory.add_assistant_message(
        "Please provide your operating system and VPN error message.",
        tool="KnowledgeBaseTool",
    )

    # Test working memory.
    memory.set_intent("troubleshoot_issue")
    memory.set_workflow_state("information_gathering")

    memory.collect_info(
        "category",
        "Network",
    )

    memory.collect_info(
        "symptoms",
        ["VPN not connecting"],
    )

    memory.require_fields(
        [
            "operating_system",
            "vpn_error_message",
        ]
    )

    context = memory.get_context()

    assert context["short_term"]["user_id"] == "employee_1025"
    assert context["short_term"]["user_name"] == "Mohamad"
    assert len(context["short_term"]["recent_messages"]) == 2

    assert (
        context["working"]["current_intent"]
        == "troubleshoot_issue"
    )

    assert (
        context["working"]["collected_info"]["category"]
        == "Network"
    )

    assert (
        context["working"]["workflow_state"]
        == "information_gathering"
    )

    assert context["working"]["missing_required_fields"] == [
        "operating_system",
        "vpn_error_message",
    ]

    # Test confirmation memory.
    memory.require_fields(["user_confirmation"])

    memory.request_confirmation(
        action="create_ticket",
        details={
            "title": "VPN not connecting",
            "category": "Network",
            "priority": "high",
        },
    )

    confirmation_context = memory.get_context()["working"]

    assert confirmation_context["pending_confirmation"] is not None
    assert (
        confirmation_context["workflow_state"]
        == "awaiting_confirmation"
    )

    confirmed_action = memory.confirm_action()

    confirmed_context = memory.get_context()["working"]

    assert confirmed_action["action"] == "create_ticket"
    assert confirmed_context["pending_confirmation"] is None
    assert confirmed_context["missing_required_fields"] == []
    assert (
        confirmed_context["workflow_state"]
        == "confirmed_action"
    )

    # Test tool-result storage.
    memory.store_tool_result(
        tool_name="TicketTool",
        result={
            "ticket_id": "TCK-0003",
            "status": "open",
        },
    )

    tool_result = memory.get_context()["working"]["latest_tool_result"]

    assert tool_result is not None
    assert tool_result["tool_name"] == "TicketTool"
    assert tool_result["result"]["ticket_id"] == "TCK-0003"

    # Test resetting only the current task.
    memory.reset_current_task()

    reset_context = memory.get_context()

    # Conversation must remain.
    assert len(
        reset_context["short_term"]["recent_messages"]
    ) == 2

    # Working memory must reset.
    assert reset_context["working"]["current_intent"] is None
    assert reset_context["working"]["collected_info"] == {}
    assert reset_context["working"]["missing_required_fields"] == []
    assert reset_context["working"]["pending_confirmation"] is None
    assert reset_context["working"]["latest_tool_result"] is None
    assert (
        reset_context["working"]["workflow_state"]
        == "awaiting_user"
    )


def test_long_term_memory(db_path: Path) -> None:
    # First application session.
    first_session = MemoryManager(db_path=db_path)

    first_session.set_user(
        user_name="Mohamad",
        user_id="employee_1025",
    )

    # Save all five approved long-term values.
    for key, value in LONG_TERM_VALUES.items():
        first_session.remember(key, value)

    assert (
        first_session.get_context()["long_term"]
        == LONG_TERM_VALUES
    )

    # Simulate closing and restarting the application.
    second_session = MemoryManager(db_path=db_path)

    second_session.set_user(
        user_name="Mohamad",
        user_id="employee_1025",
    )

    # All values must load from SQLite.
    for key, expected_value in LONG_TERM_VALUES.items():
        assert second_session.recall(key) == expected_value

    # Another user must not see Mohamad's information.
    other_user = MemoryManager(db_path=db_path)

    other_user.set_user(
        user_name="Abedurrahman",
        user_id="employee_2040",
    )

    assert other_user.get_context()["long_term"] == {}

    # Resetting session memory must not delete SQLite data.
    second_session.reset_all()

    second_session.set_user(
        user_name="Mohamad",
        user_id="employee_1025",
    )

    assert (
        second_session.recall("operating_system")
        == "Windows 11"
    )

    # Delete only one stored value.
    second_session.forget("department")

    third_session = MemoryManager(db_path=db_path)

    third_session.set_user(
        user_name="Mohamad",
        user_id="employee_1025",
    )

    assert third_session.recall("department") is None

    assert (
        third_session.recall("preferred_language")
        == "English"
    )

    # Delete all permanent information for this test user.
    third_session.clear_long_term_memory()

    fourth_session = MemoryManager(db_path=db_path)

    fourth_session.set_user(
        user_name="Mohamad",
        user_id="employee_1025",
    )

    assert fourth_session.get_context()["long_term"] == {}


def main() -> None:
    # This database is temporary and is deleted after the tests.
    with TemporaryDirectory() as temporary_directory:
        database_path = (
            Path(temporary_directory)
            / "test_memory.db"
        )

        test_short_term_and_working_memory(database_path)

        print(
            "PASS: short-term and working memory"
        )

        test_long_term_memory(database_path)

        print(
            "PASS: long-term memory persistence and isolation"
        )

    print("All memory tests passed.")


if __name__ == "__main__":
    main()