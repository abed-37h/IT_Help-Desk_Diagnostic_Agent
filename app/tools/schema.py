from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

class IssueCategory(str, Enum):
    NETWORK = 'Network'
    ACCOUNT = 'Account'
    OS = 'OS'
    APPLICATION = 'Application'
    UNKNOWN = 'unknown'

class UserInfo(BaseModel):
    user_id: str
    name: str
    device_type: str
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
