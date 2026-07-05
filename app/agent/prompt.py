from app.agent.state import WorkflowStage

SYSTEM_PROMPT = """
You are an IT Help-Desk Diagnostic Agent. You help users troubleshoot technical issues through a structured workflow.

You must follow these stages in order:
1. Information Gathering: Ask the user to describe their issue and collect symptoms.
2. Analysis: Classify the issue and retrieve relevant troubleshooting steps.
3. Action: Create a support ticket only after explicit user confirmation.
4. Reporting: Generate a final resolution report.

Rules:
- Only use information returned by your tools, never guess or assume.
- Always ask for user confirmation before creating or updating a ticket.
- If the issue is outside your scope, acknowledge it and offer to hand off to a human technician.
- Keep responses concise and professional.
"""

TOOLS = [
    {
        "name": "KnowledgeBaseTool",
        "description": "Retrieves IT troubleshooting articles and steps from the knowledge base that match the user's issue.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The issue or symptom to search for in the knowledge base."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "DiagnosticTool",
        "description": "Analyzes the collected symptoms and classifies the issue into one of four categories: Network, Account, OS, or Application.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symptoms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of symptoms described by the user."
                }
            },
            "required": ["symptoms"]
        }
    },
    {
        "name": "TicketTool",
        "description": "Creates or updates a support ticket in the system. Must only be called after explicit user confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "issue_summary": {
                    "type": "string",
                    "description": "A brief summary of the issue."
                },
                "category": {
                    "type": "string",
                    "description": "The classified issue category: Network, Account, OS, or Application.",
                    "enum": ["Network", "Account", "OS", "Application"]
                },
                "user_id": {
                    "type": "string",
                    "description": "The ID of the user reporting the issue."
                }
            },
            "required": ["issue_summary", "category", "user_id"]
        }
    },
    {
        "name": "ReportTool",
        "description": "Generates a structured resolution report summarizing the issue, troubleshooting steps taken, and ticket reference if available.",
        "input_schema": {
            "type": "object",
            "properties": {
                "kb_articles": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "The knowledge base articles retrieved during the session."
                },
                "category": {
                    "type": "string",
                    "description": "The classified issue category.",
                    "enum": ["Network", "Account", "OS", "Application"]
                },
                "ticket_id": {
                    "type": "string",
                    "description": "The ticket ID if one was created, otherwise omit."
                }
            },
            "required": ["kb_articles", "category"]
        }
    }
]
