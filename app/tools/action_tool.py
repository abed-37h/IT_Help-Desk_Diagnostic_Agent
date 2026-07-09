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
    CreateTicketInput,
    CreateTicketOutput,
    UpdateTicketInput,
    UpdateTicketOutput,
    Error,
)

@tool(args_schema=CreateTicketInput)
def open_support_ticket(
    user_id: str,
    user_name: str,
    title: str,
    description: str,
    category: IssueCategory,
    priority: Literal['low', 'medium', 'high', 'critical'],
) -> CreateTicketOutput | Error:
    '''
    Creates a new incident support ticket in the IT database. Only call this tool 
    after the user has explicitly confirmed they want to open a ticket. Before 
    calling, present the ticket details (title, category, priority) to the user 
    and wait for a clear yes confirmation. Use the title and category from the 
    fetch_issue_knowledge result, and map severity to priority directly. Never 
    call this tool more than once per session for the same issue.
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
            
            return CreateTicketOutput(
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
    Updates the status of an existing support ticket in the IT database. Only call 
    this tool after the user has explicitly confirmed the status change. Present 
    the current status, the new status, and the reason to the user before calling. 
    Use this tool when the user reports that their issue has been resolved, needs 
    further investigation, or requires escalation. Always provide a clear note 
    explaining the reason for the status change.
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
        
