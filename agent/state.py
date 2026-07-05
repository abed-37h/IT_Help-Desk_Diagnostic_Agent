from typing import TypedDict, List, Optional, Annotated
from enum import Enum
from langgraph.graph.message import add_message

class WorkflowStage(Enum):
    GATHERING = 'gathering'
    ANALYSIS = 'analysis'
    ACTION = 'action'
    REPORTING = 'reporting'
    
class IssueCategory(Enum):
    NETWORK = 'Network'
    ACCOUNT = 'Account'
    OS = 'OS'
    APP = 'Application'
    UNKNOWN ='unknown'
    
class AgentState(TypedDict):
    messages: Annotated[list, add_message]
    stage: WorkflowStage
    
    # Gathering
    symptoms: List[str]
    user_id: Optional[str]

    # Analysis
    issue_category: IssueCategory
    kb_articles: List[dict]
    
    # Action
    user_confirmed: bool
    ticket_id: Optional[str]

    # Reporting
    reporting: Optional[str]
    resolved: bool
