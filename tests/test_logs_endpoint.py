"""Tests for the /v1/logs ring-buffer query and version unification."""

import json
import logging

from app.logging_config import JsonFormatter, RingBufferLogHandler, ensure_logging, recent_logs
from app.version import __version__


def _record(event, level=logging.INFO, **extra):
    rec = logging.LogRecord('t', level, 'p', 1, event, (), None)
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


def test_ring_buffer_filters():
    h = RingBufferLogHandler(100)
    h.emit(_record('embed_ok', request_id='r1', lane='fast'))
    h.emit(_record('input_truncated', level=logging.WARNING, request_id='r2'))
    h.emit(_record('embed_ok', request_id='r3', infer_ms=91.0))
    assert len(h.query()) == 3
    assert len(h.query(event='embed_ok')) == 2
    assert len(h.query(request_id='r2')) == 1
    assert h.query(request_id='r2')[0]['event'] == 'input_truncated'


def test_ring_buffer_substring_search():
    h = RingBufferLogHandler(100)
    h.emit(_record('http_request', path='/v1/embeddings', status=200))
    h.emit(_record('http_request', path='/v1/logs', status=200))
    hits = h.query(q='/v1/embeddings')
    assert len(hits) == 1 and hits[0]['path'] == '/v1/embeddings'


def test_ring_buffer_newest_first_and_limit():
    h = RingBufferLogHandler(100)
    for i in range(10):
        h.emit(_record(f'ev{i}'))
    out = h.query(limit=3)
    assert [e['event'] for e in out] == ['ev9', 'ev8', 'ev7']


def test_recent_logs_wired_to_root():
    ensure_logging()
    logging.getLogger('wire.test').info('wired_event', extra={'request_id': 'w1'})
    hits = recent_logs(event='wired_event')
    assert hits and hits[0]['request_id'] == 'w1'


def test_version_is_semver_and_shared():
    parts = __version__.split('.')
    assert len(parts) == 3 and all(p.isdigit() for p in parts)


def test_vec_digest_deterministic_and_compact():
    import numpy as np

    from app.logging_config import vec_digest

    v = np.random.default_rng(42).normal(size=1024).astype(np.float32)
    d1, d2 = vec_digest(v), vec_digest(v.copy())
    assert d1 == d2, 'same vector must yield identical digest'
    assert d1['dim'] == 1024
    assert len(d1['sha8']) == 8
    serialized = json.dumps(d1)
    assert len(serialized) < 80, f'digest must stay compact, got {len(serialized)} chars'


def test_vec_digest_handles_none_and_list():
    from app.logging_config import vec_digest

    assert vec_digest(None) is None
    d = vec_digest([0.1, 0.2, 0.3])
    assert d['dim'] == 3 and d['norm'] > 0


def test_redact_limit_parameter():
    from app.logging_config import redact

    long = 'x' * 500
    out = redact(long, limit=400)
    assert out.endswith('…') and len(out) <= 401
    assert redact('short') == 'short'


def test_ring_buffer_level_filter_strict_match():
    h = RingBufferLogHandler(64)
    h.emit(_record('e1', level=logging.WARNING))
    h.emit(_record('e2', level=logging.INFO))
    h.emit(_record('e3', level=logging.ERROR))
    assert [e['event'] for e in h.query(level='WARNING')] == ['e1']
    assert [e['event'] for e in h.query(level='ERROR')] == ['e3']
    assert len(h.query()) == 3


def test_buffered_log_count_o1():
    from app.logging_config import buffered_log_count, ensure_logging

    ensure_logging()
    before = buffered_log_count()
    logging.getLogger('t.count').info('c1')
    assert buffered_log_count() == before + 1
