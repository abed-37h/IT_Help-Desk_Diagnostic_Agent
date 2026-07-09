from langchain_core.tools import tool
import sqlite3

from app.data.init_db import (
    connect,
    fetch_ticket_by_id,
    utc_now,
)

from app.tools.schema import (
    GenerateReportInput,
    ReportUser,
    ReportIssue,
    ReportTicket,
    Report,
    Error,
)

@tool(args_schema=GenerateReportInput)
def generate_report(ticket_id: str, steps_provided: list[str], handoff_required: bool) -> Report | Error:
    '''
    Generates a complete structured resolution report for a support session using 
    the ticket ID. Call this tool at the end of every session after a ticket has 
    been created or an existing ticket has been updated. The report includes all 
    issue details, troubleshooting steps, ticket status, and whether human handoff 
    is required. Present the report contents to the user as a session summary. If 
    handoff_required is true, remind the user that a technician will follow up.
    '''
    
    try:
        with connect() as conn:
            ticket = fetch_ticket_by_id(conn, ticket_id)

            if ticket is None:
                return Error(
                    error='not_found',
                    message=f'Ticket not found: {ticket_id}'
                )


            return Report(
                ticket_id=ticket_id,
                generated_at=utc_now(),
                user=ReportUser(
                    user_id=ticket['user_id'],
                    user_name=ticket['user_name'],
                ),
                issue= ReportIssue(
                    title=ticket['title'],
                    category=ticket['category'],
                    severity=ticket['priority'],
                ),
                steps_provided=steps_provided,
                ticket=ReportTicket(
                    status=ticket['status'],
                    created_at=ticket['created_at'],
                    resolved_at=ticket['resolved_at'],
                    resolution_notes=ticket['resolution_notes']
                ),
                handoff_required=handoff_required,
            )
    except ValueError as e:
        return Error(error='validation_error', message=str(e))
    except sqlite3.Error as e:
        return Error(error='db_error', message=str(e))
    except Exception as e:
        return Error(error='unexpected_error', message=str(e))
