from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError
from functools import wraps
from dotenv import load_dotenv
from typing import Any, Optional
from enum import Enum
from time import sleep

from app.data.init_db import init_db

from app.tools.analysis_tool import classify_and_validate
from app.tools.info_tool import fetch_issue_knowledge
from app.tools.action_tool import open_support_ticket, update_support_ticket
from app.tools.report_tool import generate_report
from app.tools.schema import UserInfo, IssueCategory, Error

from app.agent.state import AgentState
from app.agent.prompt import SYSTEM_PROMPT
from app.agent.utils import format_report

import app.logger.orchestrator_logger as logger

# ---------------------------- Initialize database --------------------------- #

init_db()

# ---------------------------------------------------------------------------- #
#                                   LLm Setup                                  #
# ---------------------------------------------------------------------------- #
load_dotenv()

TOOLS = [
    classify_and_validate,
    fetch_issue_knowledge,
    open_support_ticket,
    update_support_ticket,
    generate_report,
]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

tooled_llm = llm.bind_tools(TOOLS)

# ---------------------------------------------------------------------------- #
#                                 Define Nodes                                 #
# ---------------------------------------------------------------------------- #

# ------------------------------- Tool Handlers ------------------------------ #

def handle_classification(state: AgentState, args: dict) -> dict:
    try:
        return classify_and_validate.invoke({
            'symptoms': args.get('symptoms'),
            'user_info': state.get('user_info', {}),
        })
    except ValidationError as e:
        return Error(
            error='validation_error',
            message=f'Classification failed: {str(e)}',
        )
    except Exception as e:
        return Error(
            error='classification_error',
            message=f'Classification failed: {str(e)}',
        )

def handle_fetch_kb(state: AgentState, args: dict) -> dict:
    issue_id = state.get('issue_id', None)
    valid_user_info = state.get('valid_user_info', False)
    confidence = state.get('classification_result', {}).get('confidence', 0)
    if not issue_id:
        return Error(
            error='missing_issue_id',
            message='Cannot fetch knowledge without a classified issue_id.',
        )
    if not valid_user_info:
        return Error(
            error='invalid_user_info',
            message='Cannot fetch knowledge with invalid user information.',
        )
    if confidence < 0.3:
        return Error(
            error='low_confidence',
            message='Cannot fetch knowledge with low classification confidence.',
        )

    try:
        return fetch_issue_knowledge.invoke({
            'issue_id': issue_id,
        })
    except ValidationError as e:
        return Error(
            error='validation_error',
            message=f'Failed to fetch knowledge base: {str(e)}',
        )
    except Exception as e:
        return Error(
            error='kb_fetch_error',
            message=f'Failed to fetch knowledge base: {str(e)}',
        )

def handle_open_ticket(state: AgentState, args: dict) -> dict:
    user_info = state.get('user_info') or {}
    category = state.get('category', IssueCategory.UNKNOWN.value)
    priority = state.get('severity', 'medium')
    
    logger.log_confirmation_request(request={
        'action': 'open_ticket',
        'title': args.get('title'),
        'description': args.get('description'),
    })
    
    approved = interrupt({
        'type': 'approval',
        'content': "I couldn't resolve this with the available troubleshooting steps. "
                    f"I can open a support ticket with title '{args.get('title')}', "
                    f"category '{category}', "
                    f"and priority '{priority}'. Do you approve?",
    })
        
    logger.log_confirmation_response(response={
        'action': 'open_ticket',
        'approved': approved,
    })
    
    if not approved:
        return Error(
            error='open_ticket_rejected',
            message='Opening a ticket was rejected',
        )

    try:
        result = open_support_ticket.invoke({
            'user_id': user_info.get('user_id'),
            'user_name': user_info.get('user_name'),
            'title': args.get('title'),
            'description': args.get('description'),
            'category': category,
            'priority': priority,
        })
    except ValidationError as e:
        return Error(
            error='validation_error',
            message=f'Failed to create support ticket: {str(e)}',
        )
    except Exception as e:
        return Error(
            error='ticket_creation_error',
            message=f'Failed to create support ticket: {str(e)}',
        )
    
    if not isinstance(result, Error):
        logger.log_ticket_creation(result.ticket_id)
    
    return result

def handle_update_ticket(state: AgentState, args: dict) -> dict:
    logger.log_confirmation_request(request={
        'action': 'update_ticket',
        'ticket_id': state.get('ticket_id'),
        'new_status': args.get('new_status'),
        'note': args.get('note'),
    })
    
    approved = interrupt({
        'type': 'approval',
        'content': f"I can update ticket {state.get('ticket_id')} "
                    f"to status '{args.get('new_status')}' with this note: "
                    f"'{args.get('note')}'. Do you approve?",
    })
    
    logger.log_confirmation_response(response={
        'action': 'update_ticket',
        'approved': approved,
    })
    
    if not approved:
        return Error(
            error='update_ticket_rejected',
            message='Updating ticket was rejected',
        )

    try:
        result = update_support_ticket.invoke({
            'ticket_id': state.get('ticket_id'),
            'new_status': args.get('new_status'),
            'changed_by': args.get('changed_by'),
            'note': args.get('note'),
        })
    except ValidationError as e:
        return Error(
            error='validation_error',
            message=f'Failed to update support ticket: {str(e)}',
        )
    except Exception as e:
        return Error(
            error='ticket_update_error',
            message=f'Failed to update support ticket: {str(e)}',
        )
    
    if not isinstance(result, Error):
        logger.log_ticket_update(state.get('ticket_id'))
    
    return result

def handle_generate_report(state: AgentState, args: dict) -> dict:
    try:
        result = generate_report.invoke({
            'ticket_id': state.get('ticket_id'),
            'steps_provided': state.get('steps') or [],
            'handoff_required': state.get('escalate') or False,
        })
    except ValidationError as e:
        return Error(
            error='validation_error',
            message=f'Failed to generate report: {str(e)}',
        )
    except Exception as e:
        return Error(
            error='report_generation_error',
            message=f'Failed to generate report: {str(e)}',
        )
    
    return result

# ---------------------------- Exception Handlers ---------------------------- #

class ResponseError(Error):
    content: Optional[str] = 'A system error occurred. Please try again later.'
    retryable: bool = False
    retry_delay: int = 0

class LLMErrorType(str, Enum):
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    SERVICE_UNAVAILABLE = "service_unavailable"
    BAD_REQUEST = "bad_request"
    OUTPUT_ERROR = "output_error"
    UNKNOWN = "unknown"


LLM_ERROR_MAP = {
    # OpenRouter / OpenAI-compatible errors
    "RateLimitError": LLMErrorType.RATE_LIMIT,
    "AuthenticationError": LLMErrorType.AUTHENTICATION,
    "APITimeoutError": LLMErrorType.TIMEOUT,
    "APIConnectionError": LLMErrorType.NETWORK,
    "BadRequestError": LLMErrorType.BAD_REQUEST,

    # Google GenAI errors
    "ResourceExhausted": LLMErrorType.RATE_LIMIT,
    "DeadlineExceeded": LLMErrorType.TIMEOUT,
    "ServiceUnavailable": LLMErrorType.SERVICE_UNAVAILABLE,
    "PermissionDenied": LLMErrorType.AUTHENTICATION,
    "InvalidArgument": LLMErrorType.BAD_REQUEST,

    # LangChain errors
    "OutputParserException": LLMErrorType.OUTPUT_ERROR,
    "ValidationError": LLMErrorType.OUTPUT_ERROR,

    # Generic Python errors
    "TimeoutError": LLMErrorType.TIMEOUT,
    "ConnectionError": LLMErrorType.NETWORK,
}


def classify_llm_error(error: Exception) -> LLMErrorType:
    return LLM_ERROR_MAP.get(
        type(error).__name__,
        LLMErrorType.UNKNOWN,
    )

def handle_llm_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        retries = 3
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_type = classify_llm_error(e)
                match error_type:
                    case LLMErrorType.RATE_LIMIT:
                        error = ResponseError(
                            error='rate_limit',
                            message=f'Rate limit exceeded in {func.__name__}: {str(e)}',
                            content='The system is currently experiencing high demand. Please try again later.',
                            retryable=True,
                            retry_delay=10,
                        )
                    case LLMErrorType.TIMEOUT:
                        error = ResponseError(
                            error='timeout',
                            message=f'Timeout in {func.__name__}: {str(e)}',
                            content='The request timed out. Please try again later.',
                            retryable=True,
                            retry_delay=5,
                        )
                    case LLMErrorType.NETWORK:
                        error = ResponseError(
                            error='network',
                            message=f'Network error in {func.__name__}: {str(e)}',
                            content='A network error occurred. Please try again later.',
                            retryable=True,
                            retry_delay=5,
                        )
                    case LLMErrorType.AUTHENTICATION:
                        error = ResponseError(
                            error='authentication',
                            message=f'Authentication error in {func.__name__}: {str(e)}',
                            retryable=False,
                        )
                    case LLMErrorType.SERVICE_UNAVAILABLE:
                        error = ResponseError(
                            error='service_unavailable',
                            message=f'Service unavailable in {func.__name__}: {str(e)}',
                            retryable=True,
                            retry_delay=5,
                        )
                    case LLMErrorType.BAD_REQUEST:
                        error = ResponseError(
                            error='bad_request',
                            message=f'Bad request in {func.__name__}: {str(e)}',
                            content='The request was invalid. Please check your input and try again.',
                        )
                    case LLMErrorType.OUTPUT_ERROR:
                        error = ResponseError(
                            error='output_error',
                            message=f'Output error in {func.__name__}: {str(e)}',
                            retryable=True,
                            retry_delay=5,
                        )
                    case LLMErrorType.UNKNOWN:
                        error = ResponseError(
                            error='llm_exception',
                            message=f'LLM exception in {func.__name__}: {str(e)}',
                            content='An unexpected error occurred. Please try again later.',
                            retryable=True,
                            retry_delay=5,
                        )
                logger.log_llm_error(func.__name__, error.error, error.message)
                
                if error.retryable and retries > 0:
                    sleep(error.retry_delay * (2 ** (3 - retries)))  # Exponential backoff
                    retries -= 1
                    continue
                
                return error
    return wrapper

@handle_llm_exceptions
def extract_user_info(state: AgentState) -> dict:
    return llm.with_structured_output(UserInfo).invoke([
        'Extract user info - if found - from the next message',
        state.get('messages')[-1],
    ])

@handle_llm_exceptions
def llm_handle_response(state: AgentState) -> dict:
    context = state['messages']
    
    if state.get('user_info'):
        context = [
            SystemMessage(content=f"Known user info: {state['user_info']}")
        ] + context

    response = tooled_llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + context
    )
    
    return response

# ---------------------------------------------------------------------------- #

def agent(state: AgentState) -> AgentState:
    logger.log_state_update('agent', {
        'remaining_iterations': state.get('remaining_iterations', 10),
        'valid_user_info': state.get('valid_user_info', False),
        'ticket_id': state.get('ticket_id', None),
        'ticket_status': state.get('ticket_status', None),
        'has_classification': state.get('issue_id', None) is not None,
        'has_knowledge': state.get('category', None) is not None,
        'fallback_triggered': state.get('fallback_triggered', False)
    })
    
    update = {}
    
    last_message = state.get('messages')[-1]
    remaining_iters = state.get('remaining_iterations', 10)
        
    if remaining_iters <= 0:
        if state.get('ticket_id'):
            message = f"Your ticket {state.get('ticket_id')} is open. A technician will follow up shortly."
        else:
            message = "I wasn't able to resolve your issue. Please contact IT support directly for further assistance."
        
        return {
            'messages': [AIMessage(content=message)],
            'fallback_triggered': True,
            'fallback_reason': 'iteration_limit_reached',
        }
    
    if isinstance(last_message, HumanMessage):
        update['remaining_iterations'] = remaining_iters - 1
        
    if (not state.get('valid_user_info', False) and isinstance(last_message, HumanMessage)):
        user_info = extract_user_info(state)
        if isinstance(user_info, UserInfo):
            logger.log_extracted_info(user_info.model_dump())
            
            update['user_info'] = {
                **state.get('user_info', {}),
                **user_info.model_dump(exclude_none=True),
            }
    
    response = llm_handle_response({**state, **update})
    
    if isinstance(response, ResponseError):
        response = AIMessage(content=response.content, additional_kwargs={
            'error': response.error,
        })
        
    return {
        'messages': [response],
        **update,
    }

def route(state: AgentState) -> str:
    last_message = state['messages'][-1]
    if getattr(last_message, "tool_calls", None):
        return 'tools'
    if len(state['messages'] > 6):
        return 'summarize_conversation'
    return 'end'

def execute_tool(state: AgentState) -> AgentState:
    update = {}
    last_message = state["messages"][-1]
    tool_messages = []
    
    for tool_call in last_message.tool_calls:
        result = ''
        try:
            tool_name = tool_call['name']
            
            logger.log_tool_execution(tool_name)
            
            match tool_name:
                case 'classify_and_validate':
                    result = handle_classification(state, tool_call['args'])
                    update['classification_result'] = result.model_dump()
                    if not isinstance(result, Error):
                        update['valid_user_info'] = result.is_valid
                        update['issue_id'] = result.issue_id or None
                    
                case 'fetch_issue_knowledge':
                    result = handle_fetch_kb(state, tool_call['args'])
                    update['knowledge_result'] = result.model_dump()
                    if not isinstance(result, Error):
                        result.category = result.category.value
                        update['category'] = result.category
                        update['severity'] = result.severity
                        update['steps'] = result.steps
                        update['escalate'] = result.escalate
                    
                case 'open_support_ticket':
                    result = handle_open_ticket(state, tool_call['args'])
                    update['open_ticket_result'] = result.model_dump()
                    if not isinstance(result, Error):
                        update['ticket_id'] = result.ticket_id
                        update['ticket_status'] = result.status
                        
                case 'update_support_ticket':
                    result = handle_update_ticket(state, tool_call['args'])
                    update['update_ticket_result'] = result.model_dump()
                    if not isinstance(result, Error):
                        update['ticket_status'] = result.new_status
                        
                case 'generate_report':
                    result = handle_generate_report(state, tool_call['args'])
                    update['generate_report_result'] = result.model_dump()
                    if not isinstance(result, Error):
                        update['report'] = result.model_dump()
                        
                case _:
                    result = Error(
                        error='unknown_tool',
                        message=f'Unsupported tool: {tool_name}',
                    )
                
            if isinstance(result, Error):
                logger.log_tool_error(tool_name, result.model_dump())
            else:
                logger.log_tool_result(tool_name, result.model_dump())
        except:
            result = Error(
                error="tool_execution_error",
                message=f"Failed to execute tool call: {str(e)}",
            )
            
        tool_messages.append(ToolMessage(
            content=str(result.model_dump()),
            tool_call_id=tool_call.get('id'),
            name=tool_name,
        ))
    
    logger.log_state_update('execute_tool', update)
    
    return {
        'messages': tool_messages,
        **update
    }

def summarize_conversation(state: AgentState):
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {
        "summary": response.content,
        "messages": delete_messages,
    }

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                         Graph Builder - UI Interface                         #
# ---------------------------------------------------------------------------- #

def build_graph() -> CompiledStateGraph:
    '''
    Build the LangGraph workflow with agent, tool execution, routing, and memory.
    '''
    
    builder = StateGraph(AgentState)
    
    builder.add_node('agent', agent)
    builder.add_node('tools', execute_tool)
    builder.add_node('summarize_conversation', summarize_conversation)
    
    builder.add_edge(START, 'agent')
    builder.add_edge('tools', 'agent')
    builder.add_edge('summarize', END)
    builder.add_conditional_edges(
        'agent',
        route,
        {
            'tools': 'tools',
            'summarize_conversation': 'summarize_conversation',
            'end': END,
        }
    )

    graph = builder.compile(
        checkpointer=MemorySaver(),
    )
    
    return graph

def invoke_agent(graph: CompiledStateGraph, user_message: str, session_id: str) -> AgentState:
    '''
    Send a user message into the graph using the provided session_id.
    '''
    
    logger.log_workflow_event('session_started', session_id)
    
    return graph.invoke(
        {'messages': [HumanMessage(user_message)]},
        config={'configurable': {'thread_id': session_id}},
    )

def resume_workflow(graph: CompiledStateGraph, feedback: Any, session_id: str) -> AgentState:
    '''
    Resume a graph paused by interrupt(), using the user's feedback.
    '''

    logger.log_workflow_event('session_resumed', session_id)
    
    return graph.invoke(
        Command(resume=feedback),
        config={'configurable': {'thread_id': session_id}},
    )
    
def is_interrupt(response: AgentState):
    '''
    Return whether the graph is waiting for user feedback.
    '''

    return '__interrupt__' in response

def get_interrupt_metadata(response: AgentState):
    '''
    Return the interrupt payload for UI rendering.
    '''

    return response['__interrupt__'][0].value

# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    import uuid
    
    session_id = str(uuid.uuid4())
    graph = build_graph()
    
    while True:
        user_prompt = input('You: ')
        
        # Mock exiting chat to avoid infinite loop
        if user_prompt.strip().lower() == '/exit': break
        
        response = invoke_agent(graph, user_prompt, session_id)
        
        while is_interrupt(response):
            interrupt_metadata = get_interrupt_metadata(response)
            feedback = None
            
            match interrupt_metadata['type']:
                case 'approval':
                    user_message = input(interrupt_metadata['content'] + '([Yes]/no) ')
                    feedback = user_message.strip().lower() != 'no'
                case 'report':
                    print('Report:\n'+ interrupt_metadata['content'])
                    feedback = None
                
            response = resume_workflow(graph, feedback, session_id)
        
        last = response['messages'][-1]
        content = last.content if isinstance(last.content, str) else last.content[0]['text']
        print('Assistant: ' + content)
