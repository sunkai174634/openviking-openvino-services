from pathlib import Path


def test_intent_prompt_template_has_queries_schema():
    text = Path('ov_intent_analysis_sft_v7.yaml').read_text()
    assert '"queries": [' in text
    assert 'temperature: 0.1' in text
    assert 'context_type' in text
