from langchain_core.tools import tool
from pathlib import Path
import json

from app.tools.schema import (
    FetchIssueInput,
    IssueCategory,
    Issue,
    Error
)

_KNOWLEDGE_PATH = (Path(__file__).parent / '..' / 'data' / 'knowledge.json').resolve()

def _load_knowledge() -> dict:
    try:
        with _KNOWLEDGE_PATH.open('r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load knowledge base: {e}")
KNOWLEDGE = _load_knowledge()

@tool(args_schema=FetchIssueInput)
def fetch_issue_knowledge(issue_id: str) -> Issue | Error:
    '''
    Retrieve the approved knowledge-base entry for a classified issue.

    Returns grounded symptoms, troubleshooting steps, severity, and escalation guidance.
    Returns an error when the issue_id is unknown.
    '''
    
    target_issue = next(
        (
            issue for issue in KNOWLEDGE['articles']
            if issue['id'] == issue_id
        ),
        None
    )
    
    if target_issue is None:
        return Error(
            error='not_found',
            message=f'No knowledge entry found for issue_id: {issue_id}!'
        )
    
    return Issue(
        id=target_issue['id'],
        category=IssueCategory(target_issue['category']),
        title=target_issue['title'],
        symptoms=target_issue['symptoms'],
        steps=target_issue['steps'],
        severity=target_issue['severity'],
        escalate=target_issue['escalate'],
        source='knowledge.json',
    )

if __name__ == '__main__':
    result = fetch_issue_knowledge.invoke({
        'issue_id': 'KB004'
    })
    
    print(result)