"""Structured logging for OpenViking OpenVINO sidecars.

One JSON object per line on stdout — docker logs / Loki / jq friendly, no
extra dependencies. ``LOG_FORMAT=text`` switches to human-readable lines for
local development.

Every HTTP request is logged by :func:`install_request_logging` with the
REAL request path (not a route template), a request id (echoed back via the
``X-Request-ID`` response header, or taken from the inbound header), status
code and duration — this is the access-log layer the sidecars were missing.
"""

import contextvars
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone

_RESERVED = set(logging.LogRecord('', 0, '', 0, '', (), None).__dict__) | {'message', 'asctime'}

_request_id_var: contextvars.ContextVar = contextvars.ContextVar('request_id', default=None)


def current_request_id():
    """Request id of the current async context / threadpool copy, if any.

    Starlette copies the context into sync-endpoint threadpool threads, so
    engine code running inside a request can tag its log lines without
    threading the id through every call signature.
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


def redact(text, limit: int = 120):
    """Flatten and cap a string for logging; never leak full prompts into logs."""
    if not isinstance(text, str):
        return text
    flat = ' '.join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + '…'


def ensure_logging() -> None:
    """Idempotently configure the root logger (JSON by default).

    Also silences ``uvicorn.access``: the request-logging middleware below
    emits richer structured access lines (request id + real path), so the
    uvicorn plain-text access log would only duplicate noise.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    if os.environ.get('LOG_FORMAT', 'json').lower() == 'text':
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    else:
        handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(os.environ.get('LOG_LEVEL', 'INFO').upper())
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    ensure_logging()
    return logging.getLogger(name)


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
