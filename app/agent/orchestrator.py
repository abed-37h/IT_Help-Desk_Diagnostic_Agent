from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Any

from app.data.init_db import init_db

from app.tools.analysis_tool import classify_and_validate
from app.tools.info_tool import fetch_issue_knowledge
from app.tools.action_tool import open_support_ticket, update_support_ticket
from app.tools.report_tool import generate_report
from app.tools.schema import UserInfo, IssueCategory, Error

from app.agent.state import AgentState
from app.agent.prompt import SYSTEM_PROMPT
from app.agent.utils import format_report

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
    return classify_and_validate.invoke({
        'symptoms': args.get('symptoms'),
        'user_info': state.get('user_info', {})
    })

def handle_fetch_kb(state: AgentState, args: dict) -> dict:
    issue_id = state.get('issue_id', None)
    if not issue_id:
        return Error(
            error='missing_issue_id',
            message='Cannot fetch knowledge without a classified issue_id.',
        )
    
    return fetch_issue_knowledge.invoke({
        'issue_id': issue_id,
    })

def handle_open_ticket(state: AgentState, args: dict) -> dict:
    user_info = state.get('user_info') or {}
    category = state.get('category', IssueCategory.UNKNOWN.value)
    priority = state.get('severity', 'medium')
    
    approved = interrupt({
        'type': 'approval',
        'content': "I couldn't resolve this with the available troubleshooting steps. "
                    f"I can open a support ticket with title '{args.get('title')}', "
                    f"category '{category}', "
                    f"and priority '{priority}'. Do you approve?",
    })
    
    if not approved:
        return Error(
            error='open_ticket_rejected',
            message='Opening a ticket was rejected',
        )

    result = open_support_ticket.invoke({
        'user_id': user_info.get('user_id'),
        'user_name': user_info.get('user_name'),
        'title': args.get('title'),
        'description': args.get('description'),
        'category': category,
        'priority': priority,
    })
    
    return result

def handle_update_ticket(state: AgentState, args: dict) -> dict:
    approved = interrupt({
        'type': 'approval',
        'content': f"I can update ticket {state.get('ticket_id')} "
                    f"to status '{args.get('new_status')}' with this note: "
                    f"'{args.get('note')}'. Do you approve?",
    })
    
    if not approved:
        return Error(
            error='update_ticket_rejected',
            message='Updating ticket was rejected',
        )

    result = update_support_ticket.invoke({
        'ticket_id': state.get('ticket_id'),
        'new_status': args.get('new_status'),
        'changed_by': args.get('changed_by'),
        'note': args.get('note'),
    })
    
    return result

def handle_generate_report(state: AgentState, args: dict) -> dict:
    result = generate_report.invoke({
        'ticket_id': state.get('ticket_id'),
        'steps_provided': state.get('steps') or [],
        'handoff_required': state.get('escalate') or False,
    })
    
    return result

# ---------------------------------------------------------------------------- #

def agent(state: AgentState) -> AgentState:
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
        user_info = llm.with_structured_output(UserInfo).invoke([
            'Extract user info - if found - from the next message',
            last_message
        ])
        if user_info:
            update['user_info'] = {
                **state.get('user_info', {}),
                **user_info.model_dump(exclude_none=True),
            }
    
    context = state['messages']

    if update.get('user_info'):
        context = [
            SystemMessage(content=f"Known user info: {update['user_info']}")
        ] + context

    response = tooled_llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + context
    )
    
    return {
        'messages': [response],
        **update,
    }

def route(state: AgentState) -> str:
    last_message = state['messages'][-1]
    if getattr(last_message, "tool_calls", None):
        return 'tools'
    return 'end'

def execute_tool(state: AgentState) -> AgentState:
    update = {}
    last_message = state["messages"][-1]
    tool_messages = []
    
    for tool_call in last_message.tool_calls:
        result = ''
        tool_name = tool_call['name']
        
        match tool_name:
            case 'classify_and_validate':
                result = handle_classification(state, tool_call['args'])
                update['classification_result'] = result.model_dump()
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
            
        tool_messages.append(ToolMessage(
            content=str(result.model_dump()),
            tool_call_id=tool_call['id'],
            name=tool_name,
        ))
    
    return {
        'messages': tool_messages,
        **update
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
    
    builder.add_edge(START, 'agent')
    builder.add_edge('tools', 'agent')
    builder.add_conditional_edges(
        'agent',
        route,
        {
            'tools': 'tools',
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
    
    return graph.invoke(
        {'messages': [HumanMessage(user_message)]},
        config={'configurable': {'thread_id': session_id}},
    )

def resume_workflow(graph: CompiledStateGraph, feedback: Any, session_id: str) -> AgentState:
    '''
    Resume a graph paused by interrupt(), using the user's feedback.
    '''

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
