from pathlib import Path
import json

CATEGORY_KEYWORDS: dict[str, set[str]] = {
    'Network': set(),
    'Account': set(),
    'OS': set(),
    'Application': set(),
}
filled: bool = False

def fill_category_kw() -> None:
    filename = (Path(__file__).parent / '..' / 'data' / 'knowledge.json').resolve()
    try:
        with filename.open('r', encoding='utf-8') as f:
            knowledge = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return
    
    for article in knowledge['articles']:
        CATEGORY_KEYWORDS[article['category']].update(article['keywords'])

def DiagnosticTool(symptoms: list[str]) -> dict:
    if not symptoms or not isinstance(symptoms, list):
        return {
            'category': 'Unknown',
            'confidence': 0.0,
        }
    
    global filled
    if not filled:
        fill_category_kw()
        filled = True
    
    category_counts = {
        category: sum(1 for kw in CATEGORY_KEYWORDS[category]
                        if any(kw in symptom.lower() for symptom in symptoms))
        for category in CATEGORY_KEYWORDS
    }
    
    most_likely_cat = max(category_counts, key=category_counts.get)
    max_count = category_counts[most_likely_cat]

    total_matches = sum(category_counts.values())
    
    if total_matches == 0:
        return {
            'category': 'Unknown',
            'confidence': 0.0
        }
    
    return {
        'category': most_likely_cat,
        'confidence': max_count / total_matches,
    }
    
