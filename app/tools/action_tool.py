from langchain_core.tools import tool
from typing import Literal
import sqlite3

from app.data.init_db import (
    connect,
    create_ticket,
    update_ticket,
    fetch_ticket_by_id,
)

from app.tools.schema import (
    IssueCategory,
    OpenTicketInput,
    OpenTicketOutput,
    UpdateTicketInput,
    UpdateTicketOutput,
    Error,
)

@tool(args_schema=OpenTicketInput)
def open_support_ticket(
    user_id: str,
    user_name: str,
    title: str,
    description: str,
    category: IssueCategory,
    priority: Literal['low', 'medium', 'high', 'critical'],
) -> OpenTicketOutput | Error:
    '''
    Create a support ticket for an unresolved or escalated issue.

    Uses validated user, issue, category, and priority data. Returns the created ticket
    identifier and initial status, or a structured error.
    '''
    
    try:
        with connect() as conn:
            ticket_id = create_ticket(
                conn,
                user_id,
                user_name,
                title,
                description,
                category.value,
                priority,
            )
            
            ticket = fetch_ticket_by_id(conn, ticket_id)
            
            return OpenTicketOutput(
                ticket_id=ticket['ticket_id'],
                status=ticket['status'],
                created_at=ticket['created_at']
            )
    except ValueError as e:
        return Error(error='validation_error', message=str(e))
    except sqlite3.Error as e:
        return Error(error='db_error', message=str(e))
    except Exception as e:
        return Error(error='unexpected_error', message=str(e))

@tool(args_schema=UpdateTicketInput)
def update_support_ticket(
    ticket_id: str,
    new_status: Literal['open', 'in_progress', 'pending_user', 'resolved', 'closed'],
    changed_by: str,
    note: str,
) -> UpdateTicketOutput | Error:
    '''
    Update an existing support ticket status.

    Records the new status and audit note. Returns the previous and updated status,
    or a structured error.
    '''
    
    try:
        with connect() as conn:
            old_status = fetch_ticket_by_id(conn, ticket_id)['status']
            update_ticket(
                conn,
                ticket_id,
                new_status,
                changed_by,
                note,
            )
            
            ticket = fetch_ticket_by_id(conn, ticket_id)
            
            return UpdateTicketOutput(
                ticket_id=ticket['ticket_id'],
                old_status=old_status,
                new_status=ticket['status'],
                updated_at=ticket['updated_at'],
            )
    except ValueError as e:
        return Error(error='validation_error', message=str(e))
    except sqlite3.Error as e:
        return Error(error='db_error', message=str(e))
    except Exception as e:
        return Error(error='unexpected_error', message=str(e))
        
