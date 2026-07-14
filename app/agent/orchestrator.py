from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
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
from app.tools.schema import IssueCategory, Error

from app.agent.state import AgentState, WorkflowStage
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
).bind_tools(TOOLS)

# ---------------------------------------------------------------------------- #
#                                 Define Nodes                                 #
# ---------------------------------------------------------------------------- #

def agent(state: AgentState) -> AgentState:
    update = {}
    
    last_message = state.get('messages')[-1]
    remaining_iters = state.get('remaining_iterations', 10)
        
    if remaining_iters <= 0:
        if state.get('ticket_id'):
            # ticket already created, end gracefully
            message = f"I've reached my limit. Your ticket {state.get('ticket_id')} is already open. A technician will follow up."
        else:
            # no ticket, offer escalation
            message = "I've reached my limit without resolving your issue. Please contact IT support directly."
        
        return {
            'messages': [AIMessage(content=message)],
            'fallback_triggered': True,
            'fallback_reason': 'iteration_limit_reached',
        }
    
    if isinstance(last_message, HumanMessage):
        update['remaining_iterations'] = remaining_iters - 1
    
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] +
        state['messages']
    )
    
    return {
        'messages': [response],
        **update,
    }

def route(state: AgentState) -> str:
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return 'tools'
    return 'end'

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                         Graph Builder - UI Interface                         #
# ---------------------------------------------------------------------------- #

def build_graph() -> CompiledStateGraph:
    '''
    Builds and compiles the agent graph with memory checkpointing.
    Flow: START → agent → confirmation → tools → agent → END
    Called once at startup. Reused across sessions via thread_id.
    '''
    
    builder = StateGraph(AgentState)
    
    builder.add_node('agent', agent)
    builder.add_node('tools', ToolNode(tools=TOOLS))
    
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
    Sends a user message to the agent and returns updated state.
    Check is_interrupt() on the result before reading the last message.
    '''
    
    return graph.invoke(
        {'messages': [HumanMessage(user_message)]},
        config={'configurable': {'thread_id': session_id}},
    )

def resume_workflow(graph: CompiledStateGraph, feedback: Any, session_id: str) -> AgentState:
    '''
    Resumes an interrupted graph with user feedback.
    feedback: True/False for approvals, None for reports.
    session_id must match the one used in invoke_agent().
    '''

    return graph.invoke(
        Command(resume=feedback),
        config={'configurable': {'thread_id': session_id}},
    )
    
def is_interrupt(response: AgentState):
    '''Returns True if the graph paused and is waiting for user input.'''

    return '__interrupt__' in response

def get_interrupt_metadata(response: AgentState):
    '''
    Returns interrupt payload: {type: "approval"|"report", content: str}.
    Only call when is_interrupt() is True.
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
