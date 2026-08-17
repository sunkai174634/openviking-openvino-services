from pathlib import Path


def test_intent_http_entry_contains_ollama_compatibility_routes():
    text = Path('app/intent/main.py').read_text()
    assert "@app.post('/api/chat')" in text
    assert "@app.post('/api/generate')" in text
    assert 'async def ollama_generate(request: Request)' in text
    assert 'payload = await request.json()' in text


def test_generate_route_ignores_optional_ollama_fields():
    text = Path('app/intent/main.py').read_text()
    assert "payload.get('prompt')" in text
    assert "payload.get('stream') is True" in text
