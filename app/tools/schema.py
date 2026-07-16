from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, Literal

class IssueCategory(str, Enum):
    NETWORK = 'Network'
    ACCOUNT = 'Account'
    OS = 'OS'
    APPLICATION = 'Application'
    UNKNOWN = 'unknown'

class UserInfo(BaseModel):
    user_id: str | None = None
    user_name: str | None = None
    device_type: str | None = None
    os: Optional[str] = None
    app_name: Optional[str] = None
    since_when: Optional[str] = None
    
class ClassifyValidateInput(BaseModel):
    symptoms: list[str] | str
    user_info: UserInfo
    
class ClassifyValidateOutput(BaseModel):
    is_valid: bool
    missing_fields: list[str] = Field(description="Required fields not provided by the user for the matched category.")
    issue_id: str | None
    category: IssueCategory
    confidence: float = Field(description="Score between 0 and 1. Below 0.3 is considered low confidence.")
    escalate: bool

class FetchIssueInput(BaseModel):
    issue_id: str

class Issue(BaseModel):
    id: str
    category: IssueCategory
    title: str
    symptoms: list[str]
    steps: list[str]
    severity: Literal['low', 'medium', 'high', 'critical']
    escalate: bool
    source: Optional[str]
    
class Error(BaseModel):
    error: str
    message: str
    
class OpenTicketInput(BaseModel):
    user_id: str
    user_name: str
    title: str
    description: str
    category: IssueCategory
    priority: Literal['low', 'medium', 'high', 'critical']

class OpenTicketOutput(BaseModel):
    ticket_id: str
    status: Literal['open', 'in_progress', 'pending_user', 'resolved', 'closed']
    created_at: str

class UpdateTicketInput(BaseModel):
    ticket_id: str
    new_status: Literal['open', 'in_progress', 'pending_user', 'resolved', 'closed']
    changed_by: str
    note: str

class UpdateTicketOutput(BaseModel):
    ticket_id: str
    old_status: Literal['open', 'in_progress', 'pending_user', 'resolved', 'closed']
    new_status: Literal['open', 'in_progress', 'pending_user', 'resolved', 'closed']
    updated_at: str

class GenerateReportInput(BaseModel):
    ticket_id: str
    steps_provided: list[str]
    handoff_required: bool

class ReportUser(BaseModel):
    user_id: str
    user_name: str

class ReportIssue(BaseModel):
    title: str
    category: IssueCategory
    severity: Literal['low', 'medium', 'high', 'critical']

class ReportTicket(BaseModel):
    status: Literal['open', 'in_progress', 'pending_user', 'resolved', 'closed']
    created_at: str
    resolved_at: str
    resolution_notes: str

class Report(BaseModel):
    ticket_id: str
    generated_at: str
    user: ReportUser
    issue: ReportIssue
    steps_provided: list[str]
    ticket: ReportTicket
    handoff_required: bool
