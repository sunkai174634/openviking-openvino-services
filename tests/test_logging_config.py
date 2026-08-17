"""Tests for the shared structured-logging layer.

No FastAPI app import needed — these exercise the formatter and helpers
directly, plus request-id contextvar propagation across a thread, which is
how engine code sees the id inside sync endpoints.
"""

import json
import logging


from app.logging_config import JsonFormatter, current_request_id, redact


def _record(msg, **extra):
    return logging.LogRecord('test', logging.WARNING, 'p', 1, msg, (), None)


def test_json_formatter_emits_one_line_json():
    fmt = JsonFormatter()
    line = fmt.format(_record('something_happened'))
    obj = json.loads(line)
    assert obj['event'] == 'something_happened'
    assert obj['level'] == 'WARNING'
    assert 'ts' in obj and 'logger' in obj


def test_json_formatter_extra_fields_become_top_level():
    rec = _record('queue_timeout')
    rec.lane = 'fast'
    rec.timeout_s = 10
    obj = json.loads(JsonFormatter().format(rec))
    assert obj['lane'] == 'fast'
    assert obj['timeout_s'] == 10


def test_json_formatter_survives_non_json_values():
    rec = _record('weird')
    rec.payload = {'set': {1, 2}}  # not JSON-serializable
    obj = json.loads(JsonFormatter().format(rec))
    assert obj['event'] == 'weird'


def test_redact_caps_length_and_flattens_whitespace():
    long_text = 'a' * 300
    assert len(redact(long_text, limit=50)) == 51  # 50 chars + ellipsis char
    assert redact('line1\n line2\t tab') == 'line1 line2 tab'
    assert redact(None) is None


def test_request_id_propagates_into_threadpool():
    """Mirrors the real path: FastAPI sync endpoint -> anyio/asyncio threadpool.

    A bare threading.Thread does NOT inherit contextvars; the threadpool used
    by Starlette (anyio.to_thread / asyncio.to_thread) does. Verify on that path.
    """
    import asyncio

    from app.logging_config import _request_id_var

    token = _request_id_var.set('req-abc-123')
    seen = {}

    async def main():
        seen['id'] = await asyncio.to_thread(current_request_id)

    asyncio.run(main())
    _request_id_var.reset(token)
    assert seen['id'] == 'req-abc-123'
    assert current_request_id() is None
