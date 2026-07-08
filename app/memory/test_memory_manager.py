from memory_manager import MemoryManager


def main():
    memory = MemoryManager()

    memory.set_user("Mohamad")
    memory.add_user_message("My VPN is not connecting.")
    memory.set_intent("troubleshoot")
    memory.collect_info("symptoms", ["VPN not connecting"])
    memory.collect_info("category", "Network")
    memory.require_fields(["confirmation"])
    memory.request_confirmation(
        action="create_ticket",
        details={
            "title": "VPN not connecting",
            "category": "Network",
            "priority": "high",
        },
    )

    print(memory.get_context())

    pending_action = memory.confirm_action()
    print("Confirmed action:")
    print(pending_action)

    memory.store_tool_result(
        tool_name="TicketTool",
        result={
            "ticket_id": "TCK-0003",
            "status": "open",
        },
    )

    print(memory.get_context())


if __name__ == "__main__":
    main()