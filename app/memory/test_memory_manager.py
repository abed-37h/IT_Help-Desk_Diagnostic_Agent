import json

from memory_manager import MemoryManager


def main():
    memory = MemoryManager()

    # 1. Set current session user
    memory.set_user("Mohamad")

    # 2. Store user message
    memory.add_user_message("My VPN is not connecting.")

    # 3. Store intent and workflow state
    memory.set_intent("troubleshoot_issue")
    memory.set_workflow_state("information_gathering")

    # 4. Store collected information
    memory.collect_info("symptoms", ["VPN not connecting"])
    memory.collect_info("category", "network")

    # 5. Store missing required fields
    memory.require_fields(["user_confirmation"])

    # 6. Request confirmation before creating a ticket
    memory.request_confirmation(
        action="create_ticket",
        details={
            "title": "VPN not connecting",
            "category": "network",
            "priority": "high",
        },
    )

    print("\n--- Context before confirmation ---")
    print(json.dumps(memory.get_context(), indent=4))

    # 7. Confirm pending action
    pending_action = memory.confirm_action()
    memory.require_fields([])

    print("\n--- Confirmed action ---")
    print(json.dumps(pending_action, indent=4))

    # 8. Store tool result after ticket creation
    memory.store_tool_result(
        tool_name="TicketTool",
        result={
            "ticket_id": "TCK-0003",
            "status": "open",
        },
    )

    # 9. Move workflow to reporting stage
    memory.set_workflow_state("reporting")

    print("\n--- Context after tool result ---")
    print(json.dumps(memory.get_context(), indent=4))


if __name__ == "__main__":
    main()