import time
import threading
from typing import List, Tuple

from threading import BoundedSemaphore, Condition, Lock

import numpy as np
import openvino as ov
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer

from logging_config import (
    current_request_id,
    get_logger,
    install_request_logging,
    recent_logs,
    redact,
    vec_digest,
)
from version import __version__
from .config import (
    MODEL_ID,
    MODEL_DIR,
    OPENVINO_DEVICE,
    MAX_INPUT_TOKENS,
    MAX_QUEUE_SIZE,
    QUEUE_TIMEOUT_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    LONG_REQUEST_TOKENS,
    SHORT_REQUEST_TOKENS,
    LONG_QUEUE_TIMEOUT_SECONDS,
)

logger = get_logger('embedding')
app = FastAPI(title='OpenViking OpenVINO Embedding Service', version=__version__)
install_request_logging(app, 'embedding')
engine = None
started_at = time.time()


class EmbeddingOverloadedError(RuntimeError):
    pass


class EmbeddingQueueTimeoutError(TimeoutError):
    pass


class _Work:
    """One queued inference job. The worker sets done=True and fills result."""

    __slots__ = ('input_ids', 'attention_mask', 'lane', 'done', 'emb', 'error', 'infer_ms', 'enqueued_at', 'picked_at')

    def __init__(self, input_ids, attention_mask, lane):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.lane = lane
        self.done = False
        self.emb = None
        self.error = None
        self.infer_ms = 0.0
        self.enqueued_at = 0.0
        self.picked_at = 0.0


class Embedder:
    """
    Two-lane priority scheduler over a single InferRequest, v0.2.0.

    Changes vs 0.1.x (which starved during backlog replay):

    1. A single worker thread owns the InferRequest. Request threads never
       call infer(), so the old infer_lock and its 10s acquire timeout are
       gone entirely.
    2. Tokenization runs in the request thread BEFORE queueing. Token counts
       are therefore known when choosing a lane (0.1.x tokenized inside the
       infer lock, after queueing, which made prioritization impossible).
    3. Two FIFO lanes: fast (<= SHORT_REQUEST_TOKENS: readiness probes, live
       queries) and slow (background memory extraction). The worker always
       drains fast before slow. A probe can only ever wait behind the one
       currently-running infer plus earlier fast jobs.
    4. Slow-lane jobs wait up to LONG_QUEUE_TIMEOUT_SECONDS instead of dying
       at QUEUE_TIMEOUT_SECONDS, so replay bursts no longer turn into retry
       storms of 429s.
    5. Lock discipline: stats_lock is a leaf (never held across lane ops or
       infer); sched_cond is held only for lane push/pop/wait. No nesting.

    Fast-lane worst case wait = current infer (bounded by MAX_INPUT_TOKENS;
    ~8.4s at 2048 tokens on N150) + queued fast jobs (~0.1s each). Configure
    MAX_INPUT_TOKENS=2048 so this stays under QUEUE_TIMEOUT_SECONDS=10.
    """

    def __init__(self):
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.tokenizer_load_ms = (time.perf_counter() - t0) * 1000
        self.core = ov.Core()
        self.available_devices = list(self.core.available_devices)
        self.gpu_name = self.core.get_property(OPENVINO_DEVICE, 'FULL_DEVICE_NAME') if OPENVINO_DEVICE in self.available_devices else None
        t1 = time.perf_counter()
        model = self.core.read_model(f'{MODEL_DIR}/openvino_model.xml')
        self.compiled = self.core.compile_model(model, OPENVINO_DEVICE)
        self.compile_ms = (time.perf_counter() - t1) * 1000
        self.request = self.compiled.create_infer_request()
        self.capacity = BoundedSemaphore(value=1 + MAX_QUEUE_SIZE)
        self.sched_cond = Condition()
        self.fast_lane: List[_Work] = []
        self.slow_lane: List[_Work] = []
        self.stats_lock = Lock()
        self.total_requests = 0
        self.failed_requests = 0
        self.rejected_requests = 0
        self.queue_timeouts = 0
        self.request_timeout_warnings = 0
        self.long_requests = 0
        self.in_flight = 0
        self.active_infer = 0
        self.waiting_requests = 0
        self.fast_lane_waiting = 0
        self.slow_lane_waiting = 0
        self.last_queue_wait_ms = 0.0
        self.last_tokenize_ms = 0.0
        self.last_infer_ms = 0.0
        self.last_request_ms = 0.0
        self.last_prompt_tokens = 0
        self.last_batch_size = 0
        self.last_error = None
        self.last_warning = None

    # ---- stats helpers (leaf lock only) ----

    def snapshot(self) -> dict:
        with self.stats_lock:
            return {
                'max_queue_size': MAX_QUEUE_SIZE,
                'queue_timeout_seconds': QUEUE_TIMEOUT_SECONDS,
                'long_queue_timeout_seconds': LONG_QUEUE_TIMEOUT_SECONDS,
                'request_timeout_seconds': REQUEST_TIMEOUT_SECONDS,
                'short_request_tokens': SHORT_REQUEST_TOKENS,
                'long_request_tokens': LONG_REQUEST_TOKENS,
                'total_requests': self.total_requests,
                'failed_requests': self.failed_requests,
                'rejected_requests': self.rejected_requests,
                'queue_timeouts': self.queue_timeouts,
                'request_timeout_warnings': self.request_timeout_warnings,
                'long_requests': self.long_requests,
                'in_flight': self.in_flight,
                'active_infer': self.active_infer,
                'waiting_requests': self.waiting_requests,
                'fast_lane_waiting': self.fast_lane_waiting,
                'slow_lane_waiting': self.slow_lane_waiting,
                'last_queue_wait_ms': self.last_queue_wait_ms,
                'last_tokenize_ms': self.last_tokenize_ms,
                'last_infer_ms': self.last_infer_ms,
                'last_request_ms': self.last_request_ms,
                'last_prompt_tokens': self.last_prompt_tokens,
                'last_batch_size': self.last_batch_size,
                'last_error': self.last_error,
                'last_warning': self.last_warning,
            }

    def _begin_request(self):
        with self.stats_lock:
            self.total_requests += 1
            self.in_flight += 1

    def _finish_request(self, error: Exception | None = None):
        with self.stats_lock:
            self.in_flight = max(0, self.in_flight - 1)
            if error is not None:
                self.failed_requests += 1
                self.last_error = str(error)
            else:
                self.last_error = None

    def _note_queue_wait(self, ms: float):
        with self.stats_lock:
            self.last_queue_wait_ms = ms

    def _note_timings(self, tok_ms: float, infer_ms: float, prompt_tokens: int, batch: int):
        with self.stats_lock:
            self.last_tokenize_ms = tok_ms
            self.last_infer_ms = infer_ms
            self.last_prompt_tokens = prompt_tokens
            self.last_batch_size = batch

    def _note_long_request(self):
        with self.stats_lock:
            self.long_requests += 1

    # ---- worker (single owner of the InferRequest) ----

    def worker_loop(self):
        logger.info('infer_worker_started')
        while True:
            with self.sched_cond:
                while not self.fast_lane and not self.slow_lane:
                    self.sched_cond.wait()
                if self.fast_lane:
                    work = self.fast_lane.pop(0)
                else:
                    work = self.slow_lane.pop(0)
                work.picked_at = time.perf_counter()
            with self.stats_lock:
                self.active_infer += 1
            try:
                t0 = time.perf_counter()
                res = self.request.infer({'input_ids': work.input_ids, 'attention_mask': work.attention_mask})
                work.infer_ms = (time.perf_counter() - t0) * 1000
                hidden = list(res.values())[0]
                emb = hidden[:, -1, :]
                emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                work.emb = emb.astype(np.float32).tolist()
                work.error = None
            except Exception as exc:  # noqa: BLE001
                work.error = exc
                logger.exception('infer_worker_error', extra={'lane': work.lane, 'error': str(exc)})
            finally:
                with self.stats_lock:
                    self.active_infer = max(0, self.active_infer - 1)
                work.done = True
                with self.sched_cond:
                    self.sched_cond.notify_all()

    def _submit(self, work: _Work, timeout: float) -> bool:
        """Queue the work and wait up to timeout. False = timed out and removed."""
        work.enqueued_at = time.perf_counter()
        with self.sched_cond:
            if work.lane == 'fast':
                self.fast_lane.append(work)
            else:
                self.slow_lane.append(work)
            self.sched_cond.notify_all()
        deadline = time.perf_counter() + timeout
        while not work.done:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                with self.sched_cond:
                    try:
                        self.fast_lane.remove(work)
                        return False
                    except ValueError:
                        pass
                    try:
                        self.slow_lane.remove(work)
                        return False
                    except ValueError:
                        pass
                # worker already picked it up: wait for completion
                while not work.done:
                    time.sleep(0.005)
                return True
            time.sleep(0.005)
        return True

    # ---- public API ----

    def embed(self, texts: List[str]) -> Tuple[List[List[float]], dict]:
        request_start = time.perf_counter()
        if not self.capacity.acquire(blocking=False):
            msg = f'queue full: max_queue_size={MAX_QUEUE_SIZE}'
            with self.stats_lock:
                self.rejected_requests += 1
                self.last_error = msg
            logger.warning('queue_full', extra={'max_queue_size': MAX_QUEUE_SIZE})
            raise EmbeddingOverloadedError(msg)
        self._begin_request()
        try:
            if not texts:
                raise ValueError('input must not be empty')
            # tokenize outside scheduler -> token count known before lane choice
            raw_tokens = sum(len(t) for t in texts)
            t0 = time.perf_counter()
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_TOKENS,
                return_tensors='np',
            )
            tok_ms = (time.perf_counter() - t0) * 1000
            # truncation visibility: tokenizer caps each sequence at MAX_INPUT_TOKENS
            truncated = any(len(self.tokenizer.encode(t, add_special_tokens=False)) > MAX_INPUT_TOKENS for t in texts)
            input_ids = enc['input_ids'].astype(np.int64)
            attention_mask = enc['attention_mask'].astype(np.int64)
            prompt_tokens = int(attention_mask.sum())
            is_long = prompt_tokens > LONG_REQUEST_TOKENS
            lane = 'fast' if prompt_tokens <= SHORT_REQUEST_TOKENS else 'slow'
            timeout = QUEUE_TIMEOUT_SECONDS if lane == 'fast' else LONG_QUEUE_TIMEOUT_SECONDS
            work = _Work(input_ids, attention_mask, lane)
            wait_start = time.perf_counter()
            with self.stats_lock:
                self.waiting_requests += 1
                if lane == 'fast':
                    self.fast_lane_waiting += 1
                else:
                    self.slow_lane_waiting += 1
            try:
                completed = self._submit(work, timeout)
            finally:
                with self.stats_lock:
                    self.waiting_requests = max(0, self.waiting_requests - 1)
                    if lane == 'fast':
                        self.fast_lane_waiting = max(0, self.fast_lane_waiting - 1)
                    else:
                        self.slow_lane_waiting = max(0, self.slow_lane_waiting - 1)
            queue_wait_ms = (work.picked_at - work.enqueued_at) * 1000 if work.picked_at else (time.perf_counter() - wait_start) * 1000
            if not completed:
                with self.stats_lock:
                    self.queue_timeouts += 1
                logger.warning(
                    'queue_timeout',
                    extra={'lane': lane, 'timeout_s': timeout, 'prompt_tokens': prompt_tokens, 'batch_size': len(texts)},
                )
                raise EmbeddingQueueTimeoutError(f'{lane} lane wait timed out after {timeout}s')
            if work.error is not None:
                logger.error(
                    'infer_failed',
                    extra={'request_id': current_request_id(), 'lane': lane, 'error': str(work.error), 'prompt_tokens': prompt_tokens},
                )
                raise work.error
            emb = work.emb
            self._note_queue_wait(queue_wait_ms)
            self._note_timings(tok_ms, work.infer_ms, prompt_tokens, len(texts))
            if is_long:
                self._note_long_request()
            request_ms = (time.perf_counter() - request_start) * 1000
            timeout_warning = request_ms > REQUEST_TIMEOUT_SECONDS * 1000
            warning = None
            if truncated:
                warning = f'input truncated: prompt_tokens={prompt_tokens} > MAX_INPUT_TOKENS={MAX_INPUT_TOKENS}'
            if is_long:
                warning = (warning + '; ' if warning else '') + f'long request: prompt_tokens={prompt_tokens} > {LONG_REQUEST_TOKENS}'
            if timeout_warning:
                warning = (warning + '; ' if warning else '') + f'request exceeded timeout budget: {request_ms:.1f}ms'
            meta = {
                'queue_wait_ms': queue_wait_ms,
                'tokenize_ms': tok_ms,
                'infer_ms': work.infer_ms,
                'request_ms': request_ms,
                'prompt_tokens': prompt_tokens,
                'batch_size': len(texts),
                'is_long_request': is_long,
                'input_truncated': truncated,
                'lane': lane,
                'request_timeout_warning': timeout_warning,
                'warning': warning,
            }
            log_extra = {
                'request_id': current_request_id(),
                'lane': lane,
                'prompt_tokens': prompt_tokens,
                'batch_size': len(texts),
                'queue_wait_ms': round(queue_wait_ms, 1),
                'tokenize_ms': round(tok_ms, 1),
                'infer_ms': round(work.infer_ms, 1),
                'request_ms': round(request_ms, 1),
                'input_truncated': truncated,
                'input_chars': raw_tokens,
                'input_preview': redact(texts[0]),
                'result_digest': vec_digest(emb[0]) if emb else None,
            }
            if truncated:
                logger.warning('input_truncated', extra=log_extra)
            else:
                logger.info('embed_ok', extra=log_extra)
            self._finish_request()
            return emb, meta
        except Exception as exc:
            self._finish_request(exc)
            raise
        finally:
            self.capacity.release()


@app.on_event('startup')
def startup():
    global engine
    engine = Embedder()
    threading.Thread(target=engine.worker_loop, daemon=True, name='infer-worker').start()


@app.get('/health')
def health():
    assert engine is not None
    stats = engine.snapshot()
    return {
        'status': 'ok',
        'model': MODEL_ID,
        'dimension': 1024,
        'max_input_tokens': MAX_INPUT_TOKENS,
        'devices': engine.available_devices,
        'gpu_name': engine.gpu_name,
        'tokenizer_load_ms': round(engine.tokenizer_load_ms, 3),
        'compile_ms': round(engine.compile_ms, 3),
        'uptime_seconds': round(time.time() - started_at, 3),
        'stats': {k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()},
    }


@app.get('/v1/models')
def models():
    return {'object': 'list', 'data': [{'id': MODEL_DIR, 'object': 'model', 'created': 0, 'owned_by': 'openviking-openvino-services'}]}


@app.get('/v1/logs')
def logs(limit: int = 200, level: str | None = None, event: str | None = None,
         request_id: str | None = None, q: str | None = None):
    """Structured log tail with filtering (newest first)."""
    return {'service': 'embedding', 'version': __version__, 'total': len(recent_logs(limit=100000)), 'entries': recent_logs(limit=limit, level=level, event=event, request_id=request_id, q=q)}


@app.post('/v1/embeddings')
def embeddings(req: dict):
    assert engine is not None
    texts = req.get('input') if isinstance(req, dict) else None
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list) or not texts or any(not isinstance(x, str) or x == '' for x in texts):
        raise HTTPException(status_code=400, detail='input must be a non-empty string or list of non-empty strings')
    try:
        vectors, meta = engine.embed(texts)
    except EmbeddingOverloadedError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except EmbeddingQueueTimeoutError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        'object': 'list',
        'model': MODEL_ID,
        'data': [{'object': 'embedding', 'index': i, 'embedding': v} for i, v in enumerate(vectors)],
        'usage': {'prompt_tokens': meta['prompt_tokens'], 'total_tokens': meta['prompt_tokens']},
        'meta': {k: (round(v, 3) if isinstance(v, float) else v) for k, v in meta.items()},
    }
