"""Structured logging for OpenViking OpenVINO sidecars.

One JSON object per line on stdout — docker logs / Loki / jq friendly, no
extra dependencies. ``LOG_FORMAT=text`` switches to human-readable lines for
local development.

Components:

- :class:`JsonFormatter` — renders each record as one JSON line; ``extra=``
  fields become top-level keys.
- :class:`RingBufferLogHandler` — keeps the most recent records in memory so
  ``GET /v1/logs`` can serve them (with filtering) without touching docker
  logs. Capacity via ``LOG_BUFFER_SIZE`` (default 2000).
- :func:`install_request_logging` — per-request access log with the REAL
  request path (not a route template), a request id (echoed back via the
  ``X-Request-ID`` response header), status code and duration.
- :func:`current_request_id` — contextvar lookup so engine-layer logs can
  tag themselves with the ambient request id.
"""

import contextvars
import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone

_RESERVED = set(logging.LogRecord('', 0, '', 0, '', (), None).__dict__) | {'message', 'asctime'}

_request_id_var: contextvars.ContextVar = contextvars.ContextVar('request_id', default=None)


def current_request_id():
    """Request id of the current async context / threadpool copy, if any.

    Starlette copies the context into sync-endpoint threadpool threads, so
    engine code running inside a request can tag its log lines without
    threading the id through every call signature. (A bare threading.Thread
    does NOT inherit contextvars — internal worker threads log without an id
    by design.)
    """
    return _request_id_var.get()


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as one JSON line; ``extra=`` fields become top-level keys."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            'ts': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec='milliseconds'),
            'level': record.levelname,
            'logger': record.name,
            'event': record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith('_'):
                payload[key] = value
        if record.exc_info:
            payload['exc'] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class RingBufferLogHandler(logging.Handler):
    """Thread-safe in-memory ring buffer of structured log entries."""

    def __init__(self, capacity: int):
        super().__init__(level=logging.INFO)
        self._entries: deque = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.setFormatter(JsonFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:  # noqa: BLE001 — logging must never crash the app
            return
        with self._lock:
            self._entries.append(line)

    def query(
        self,
        limit: int = 200,
        level: str | None = None,
        event: str | None = None,
        request_id: str | None = None,
        q: str | None = None,
    ) -> list[dict]:
        """Return matching entries, newest first.

        ``q`` is a case-insensitive substring match against the raw line —
        it catches logger names, event names and any extra field value.
        """
        level_no = getattr(logging, (level or '').upper(), None) if level else None
        needle = q.lower() if q else None
        out: list[dict] = []
        with self._lock:
            snapshot = list(self._entries)
        for line in reversed(snapshot):
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if level_no is not None and logging.getLevelName(entry.get('level', '')) != level_no:
                if entry.get('level') != (level or '').upper():
                    continue
            if event and entry.get('event') != event:
                continue
            if request_id and entry.get('request_id') != request_id:
                continue
            if needle and needle not in line.lower():
                continue
            out.append(entry)
            if len(out) >= limit:
                break
        return out


_ring_handler: RingBufferLogHandler | None = None
_stream_configured = False


def ensure_logging() -> None:
    """Idempotently configure the root logger (JSON by default) + ring buffer.

    Robust against foreign handlers (pytest's LogCaptureHandler, a launcher
    that pre-configures root): each of our own pieces is attached exactly
    once, detected by marker/identity instead of "root already has handlers".

    Also silences ``uvicorn.access``: the request-logging middleware emits
    richer structured access lines (request id + real path), so the uvicorn
    plain-text access log would only duplicate noise.
    """
    global _ring_handler, _stream_configured
    root = logging.getLogger()
    if not _stream_configured:
        stream = logging.StreamHandler()
        setattr(stream, '_ovs_stream', True)  # marker: this is ours
        if os.environ.get('LOG_FORMAT', 'json').lower() == 'text':
            stream.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
        else:
            stream.setFormatter(JsonFormatter())
        root.addHandler(stream)
        root.setLevel(os.environ.get('LOG_LEVEL', 'INFO').upper())
        logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
        _stream_configured = True
    if _ring_handler is None:
        _ring_handler = RingBufferLogHandler(int(os.environ.get('LOG_BUFFER_SIZE', '2000')))
        root.addHandler(_ring_handler)


def get_logger(name: str) -> logging.Logger:
    ensure_logging()
    return logging.getLogger(name)


def recent_logs(limit=200, level=None, event=None, request_id=None, q=None) -> list[dict]:
    """Query the in-memory ring buffer (empty list if logging not yet set up)."""
    if _ring_handler is None:
        return []
    return _ring_handler.query(limit=limit, level=level, event=event, request_id=request_id, q=q)


def redact(text, limit: int = 120):
    """Flatten and cap a string for logging; never leak full prompts into logs."""
    if not isinstance(text, str):
        return text
    flat = ' '.join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + '…'


def install_request_logging(app, service_name: str) -> None:
    """Attach per-request structured logging + X-Request-ID to a FastAPI app."""
    from fastapi import Request
    from starlette.middleware.base import BaseHTTPMiddleware

    logger = get_logger(f'{service_name}.http')

    class RequestLoggingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            request_id = request.headers.get('x-request-id') or uuid.uuid4().hex[:12]
            request.state.request_id = request_id
            _request_id_var.set(request_id)
            start = time.perf_counter()
            try:
                response = await call_next(request)
            except Exception:
                duration_ms = round((time.perf_counter() - start) * 1000, 1)
                logger.exception(
                    'http_request',
                    extra={
                        'request_id': request_id,
                        'method': request.method,
                        'path': request.url.path,
                        'status': 500,
                        'duration_ms': duration_ms,
                    },
                )
                raise
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            response.headers['X-Request-ID'] = request_id
            logger.info(
                'http_request',
                extra={
                    'request_id': request_id,
                    'method': request.method,
                    'path': request.url.path,  # real path, not the route template
                    'status': response.status_code,
                    'duration_ms': duration_ms,
                },
            )
            return response

    app.add_middleware(RequestLoggingMiddleware)
