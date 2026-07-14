from typing import TypedDict, List, Optional, Annotated
from enum import Enum
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class WorkflowStage(Enum):
    IDLE = 'idle'
    GATHERING = 'gathering'
    ANALYSIS = 'analysis'
    AWAITING_CONFIRMATION = 'awaiting_confirmation'
    ACTION = 'action'
    REPORTING = 'reporting'
    ESCALATING = 'escalating'
    RESOLVED = 'resolved'

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

    # Gathering
    user_info: Optional[dict]
    symptoms: List[str]
    
    # Workflow
    workflow_stage: WorkflowStage
    remaining_iterations: int
    
    # Tool results
    classification_result: Optional[dict]
    knowledge_result: Optional[dict]
    ticket_id: Optional[str]
    report: Optional[dict]
    
    # Action control
    pending_confirmation: bool
    pending_action: Optional[str]
    pending_action_args: Optional[dict]
    
    # Fallback
    fallback_triggered: bool
    fallback_reason: Optional[str]

