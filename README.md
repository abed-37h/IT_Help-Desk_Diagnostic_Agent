# IT Help-Desk Diagnostic Agent

An LLM-powered diagnostic agent that provides first-line IT support for network, account, operating-system, and application issues. The agent gathers symptoms and user context, classifies the issue against a structured knowledge base, delivers grounded troubleshooting steps, and — when the issue cannot be resolved — opens or updates a support ticket and generates a structured handoff report for human technicians.

Built as a course project demonstrating an explainable, controlled, and production-oriented agent architecture: explicit workflow states, typed tools, short-term/working/long-term memory, human-in-the-loop confirmation gates, and full observability through structured logging.

## Overview

The agent acts as a triage and decision-support system, not an authoritative fix-it tool. It only recommends troubleshooting steps sourced from a curated knowledge base, states its confidence and limitations, and escalates to a human technician (via a support ticket) whenever an issue is unresolved, low-confidence, or out of scope.

## Architecture

The system follows a layered design:

| Layer | Implementation |
|---|---|
| Chat interface | Streamlit |
| Orchestration | LangGraph state machine (router, tool execution, summarization, interrupts) |
| LLM reasoning core | Google Gemini (via `langchain-google-genai`), bound to typed tools |
| Tool layer | Four typed tools with Pydantic input/output schemas |
| Memory layer | Short-term, working, and optional long-term memory |
| Data layer | JSON knowledge base + SQLite ticket database |
| Container layer | Docker image with health check, orchestrated via `compose.yaml` |

### Workflow

The orchestrator is a LangGraph graph with the following nodes and transitions:

1. **agent** — the LLM reasons over the conversation and decides whether to call a tool, continue, or end the turn.
2. **tools** — executes the requested tool, validates arguments, and returns a typed result or a structured `Error`.
3. **summarize_conversation** — condenses conversation history once it exceeds a message-count threshold, keeping the agent's context bounded.

Ticket creation and ticket updates are gated by an `interrupt()` call that pauses the graph and requires explicit user approval before any state-changing action is executed. Rejected actions return a structured error instead of silently failing.

## Tools

Each tool has a documented purpose, a typed input schema, a typed output schema, and explicit error handling via a shared `Error` model.

| Tool | Category | Purpose |
|---|---|---|
| `classify_and_validate` | Analysis | Classifies reported symptoms against the knowledge base, computes a confidence score, and validates that required user/device fields are present for the matched category. |
| `fetch_issue_knowledge` | Information | Retrieves the grounded knowledge-base entry (symptoms, troubleshooting steps, severity, escalation flag) for a classified issue ID. |
| `open_support_ticket` | Action | Creates a new support ticket in SQLite for an unresolved or escalated issue. Requires explicit user confirmation. |
| `update_support_ticket` | Action | Updates the status and adds an audit note to an existing ticket. Requires explicit user confirmation. |
| `generate_report` | Reporting | Produces a structured session report summarizing the ticket, issue classification, steps provided, and handoff status. |

Classification only proceeds to knowledge retrieval when confidence is at or above `0.3` and required user context has been validated, otherwise the agent asks a clarifying question or requests the missing fields.

## Memory

- **Short-term memory** — preserves the active conversation, including the user's name, ID, and recent message history, for the duration of the session.
- **Working memory** — explicitly tracks current intent, collected information, missing required fields, pending confirmations, the latest tool result, and the current workflow stage.
- **Long-term memory (optional)** — persists user identity and preferences across sessions in SQLite, loaded automatically once a user is identified.

## Data

- `app/data/knowledge.json` — structured knowledge-base articles (symptoms, keywords, troubleshooting steps, severity, escalation rules) used for grounded classification and retrieval. No RAG pipeline, vector store, or embeddings are used; matching is deterministic keyword-based scoring against structured records.
- `app/data/tickets.db` — SQLite database storing tickets and their status-change history, initialized automatically on startup (`app/data/init_db.py`).

## Safety and Controls

- All tool arguments are validated against Pydantic schemas before execution.
- Sensitive or state-changing actions (ticket creation, ticket updates) require explicit user confirmation via a LangGraph interrupt.
- Every intent, tool call, tool result, error, and confirmation event is logged for observability (`app/logger`).
- Unsupported or out-of-scope requests are declined explicitly rather than answered from general knowledge, with an offer of simulated human handoff.
- The agent never claims a ticket action succeeded unless confirmed by the corresponding tool result.

## Evaluation

`tests/run_evaluation.py` runs a documented suite of scripted conversations (`tests/cases/evaluation_cases.json`) covering grounded queries, valid/invalid inputs, ticket actions, multi-turn memory, missing information, unsupported requests, and misuse attempts. It reports task-completion rate, tool-selection accuracy, fallback accuracy, and the count of unsafe or invalid actions executed, while backing up and restoring the ticket database so evaluation runs do not affect demo data.

## Getting Started

### Prerequisites

- Docker and Docker Compose
- A Google Gemini API key

### Configuration

Copy `.env.example` to `.env` in the project root and set the required API key:

```
GOOGLE_API_KEY=your_key_here
```

### Run with Docker

```bash
docker compose -f docker/compose.yaml up --build
```

The application is served at `http://localhost:8501`.

### Run locally (without Docker)

```bash
pip install -r requirements.txt
streamlit run app/ui/main.py
```

## Project Structure

```
app/
├── agent/          # LangGraph orchestrator, state schema, system prompt
├── data/            # Knowledge base, SQLite schema and initialization
├── logger/         # Structured event and orchestrator logging
├── memory/         # Short-term, working, and long-term memory management
├── tools/          # Typed tools: analysis, information, action, reporting
└── ui/             # Streamlit chat interface
docker/             # Dockerfile and compose configuration
tests/              # Evaluation suite and test cases
```

## Team

| Role | Responsibilities |
|---|---|
| Agent & Brain | Orchestrator, prompts, workflow state, tools, memory |
| Platform & Interface | Streamlit UI, Docker packaging, domain data, database, evaluation suite |