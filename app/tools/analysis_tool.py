from langchain_core.tools import tool
from pathlib import Path
import json

from app.tools.schema import (
    IssueCategory,
    UserInfo,
    ClassifyValidateInput,
    ClassifyValidateOutput,
)

_KNOWLEDGE_PATH = (Path(__file__).parent / '..' / 'data' / 'knowledge.json').resolve()

def _load_knowledge() -> dict:
    try:
        with _KNOWLEDGE_PATH.open('r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load knowledge base: {e}")

KNOWLEDGE = _load_knowledge()

@tool(args_schema=ClassifyValidateInput)
def classify_and_validate(symptoms: list[str] | str, user_info: UserInfo) -> ClassifyValidateOutput:
    '''
    Classify the issue from symptoms and validate required user context.

    Returns the best matching knowledge-base issue, confidence score, missing fields,
    and escalation flag. If no reliable match exists, returns UNKNOWN with low confidence.
    confidence less 
    '''

    if isinstance(symptoms, list): symptoms = ' '.join(symptoms)
    symptoms = symptoms.lower()
    
    best_match_count = 0
    best_match_index = -1
    total_match_count = 0
    
    for index, issue in enumerate(KNOWLEDGE['articles']):
        kw_match = sum(1 for kw in issue['keywords'] if kw.lower() in set(symptoms.split()))
        
        if kw_match > best_match_count:
            best_match_count = kw_match
            best_match_index = index
        total_match_count += kw_match
        
    if total_match_count == 0:
        return ClassifyValidateOutput(
            is_valid=False,
            missing_fields=[],
            issue_id=None,
            category=IssueCategory.UNKNOWN,
            confidence=0.0,
            escalate=False,
        )
    best_match = KNOWLEDGE['articles'][best_match_index]
    
    coverage = best_match_count / len(best_match['keywords'])
    dominance = best_match_count / total_match_count
    confidence = 0.7 * coverage + 0.3 * dominance
    if dominance < 0.2: confidence *= 0.5
    
    required_fields: list[str] = [
        'user_id',
        'user_name',
        'device_type',
    ]
    
    if (IssueCategory(best_match['category']) == IssueCategory.APPLICATION):
        required_fields.append('app_name')
    elif(IssueCategory(best_match['category']) in [IssueCategory.NETWORK, IssueCategory.ACCOUNT, IssueCategory.OS]):
        required_fields.append('os')

    return ClassifyValidateOutput(
        is_valid=all(
            getattr(user_info, field, None)
            for field in required_fields
        ),
        missing_fields=[
            field for field in required_fields
            if not getattr(user_info, field, None)
        ],
        issue_id= best_match['id'],
        category=IssueCategory(best_match['category']),
        confidence=confidence,
        escalate=best_match['escalate']
    )

if __name__ == '__main__':
    result = classify_and_validate.invoke({
        'symptoms': 'I have a problem with microsoft office',
        'user_info': {
            'user_id': 'USR001',
            'user_name': 'john',
            'device_type': 'laptop',
            'os': 'linux',
            'since_when': 'yesterday',
        }
    })
    print(result)