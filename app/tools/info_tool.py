from pathlib import Path
import json

def KnowledgeBaseTool(query: str) -> list:
    # Validation
    if not isinstance(query, str):
        return []
    
    # Retrieve knowledge data
    filename = (Path(__file__).parent / '..' / 'data' / 'knowledge.json').resolve()
    try:
        with filename.open('r', encoding='utf-8') as f:
            knowledge = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    
    query = query.lower()
    entries: list = []
    
    # Lookup articles
    for article in knowledge['articles']:
        if any(
            keyword.lower() in query
            for keyword in article['keywords']
        ):
            entries.append({
                'resource_type': 'article',
                **article,
            })
    
    # Lookup policies
    for policy in knowledge['policies']:
        if query in policy['title'].lower() or query in policy['content'].lower():
            entries.append({
                'resource_type': 'policy',
                **policy,
            })
    
    # Lookup faqs
    for faq in knowledge['faqs']:
        if query in faq['question'].lower() or query in faq['answer'].lower():
            entries.append({
                'resource_type': 'faq',
                **faq,
            })
    
    return entries
