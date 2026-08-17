from app.intent.engine import IntentEngine


def test_parse_json_with_markdown_fence():
    value = IntentEngine.parse_json('```json\n{"queries": []}\n```')
    assert value == {'queries': []}


def test_normalize_queries_and_reasoning_fallback():
    value = IntentEngine.normalize_plan({
        'reasoning': [{'query': 'fallback', 'context_type': 'memory', 'priority': 2}],
    })
    assert value['queries'] == [
        {'query': 'fallback', 'context_type': 'memory', 'priority': 2},
    ]


def test_normalize_limits_to_five_and_drops_invalid_rows():
    value = IntentEngine.normalize_plan({
        'queries': [
            {'query': f'q{i}', 'context_type': 'resource', 'priority': 1}
            for i in range(7)
        ] + [{'query': 123}],
    })
    assert len(value['queries']) == 5
    assert all(row['context_type'] == 'resource' for row in value['queries'])
