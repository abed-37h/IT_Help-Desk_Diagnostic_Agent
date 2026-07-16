from app.logger.logger import get_logger, log_event, log_error

from app.data.init_db import connect, fetch_ticket_by_id

orchestrator_logger = get_logger('orchestrator')

def log_workflow_event(event: str, session_id: str) -> None:
    log_event(orchestrator_logger, event, session_id=session_id)

def log_tool_execution(tool_name: str) -> None:
    log_event(orchestrator_logger, 'tool_execution', tool=tool_name)

def log_tool_result(tool_name: str, result: dict) -> None:
    log_event(orchestrator_logger, 'tool_result', tool=tool_name, result=result)

def log_tool_error(tool_name: str, error: dict) -> None:
    log_error(orchestrator_logger, f'Tool {tool_name} error: {error["error"]} | {error["message"]}', tool=tool_name, error=error)

def log_llm_error(llm_action: str, error_type: str, error: str) -> None:
    log_error(orchestrator_logger, f'LLM {llm_action} error: {error}', llm_action=llm_action, error_type=error_type, error=error)

def log_state_update(where: str, update: dict) -> None:
    log_event(orchestrator_logger, 'state_update', where=where, update=update)

def log_extracted_info(info: dict) -> None:
    log_event(orchestrator_logger, 'extracted_info', info=info)
    
def log_confirmation_request(request: dict) -> None:
    log_event(orchestrator_logger, 'confirmation_request', request=request)

def log_confirmation_response(response: dict) -> None:
    log_event(orchestrator_logger, 'confirmation_response', response=response)

def log_ticket_creation(ticket_id: str) -> None:
    with connect() as conn:
        ticket_info = fetch_ticket_by_id(conn, ticket_id)
        if ticket_info:
            log_event(orchestrator_logger, 'ticket_creation', ticket_info=ticket_info)
        else:
            log_error(orchestrator_logger, 'Ticket creation failed', ticket_id=ticket_id)

def log_ticket_update(ticket_id: str) -> None:
    with connect() as conn:
        ticket_info = fetch_ticket_by_id(conn, ticket_id)
        if ticket_info:
            log_event(orchestrator_logger, 'ticket_update', ticket_info=ticket_info)
        else:
            log_error(orchestrator_logger, 'Ticket not found', ticket_id=ticket_id)

def log_report_generation(report: dict) -> None:
    log_event(orchestrator_logger, 'report_generation', report=report)