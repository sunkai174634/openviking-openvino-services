import time
from typing import List, Tuple
from threading import BoundedSemaphore, Lock

import numpy as np
import openvino as ov
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer

from .config import (
    MODEL_DIR,
    OPENVINO_DEVICE,
    MAX_INPUT_TOKENS,
    MAX_QUEUE_SIZE,
    QUEUE_TIMEOUT_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    LONG_REQUEST_TOKENS,
)

app = FastAPI(title='OpenViking OpenVINO Embedding Sidecar', version='0.1.0')
engine = None
started_at = time.time()

class EmbeddingOverloadedError(RuntimeError):
    pass

class EmbeddingQueueTimeoutError(TimeoutError):
    pass

class Embedder:
    def __init__(self):
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            trust_remote_code=True,
            fix_mistral_regex=True,
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
        self.infer_lock = Lock()
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
        self.last_queue_wait_ms = 0.0
        self.last_tokenize_ms = 0.0
        self.last_infer_ms = 0.0
        self.last_request_ms = 0.0
        self.last_prompt_tokens = 0
        self.last_batch_size = 0
        self.last_error = None
        self.last_warning = None

    def snapshot(self) -> dict:
        with self.stats_lock:
            return {
                'max_queue_size': MAX_QUEUE_SIZE,
                'queue_timeout_seconds': QUEUE_TIMEOUT_SECONDS,
                'request_timeout_seconds': REQUEST_TIMEOUT_SECONDS,
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
                'last_queue_wait_ms': self.last_queue_wait_ms,
                'last_tokenize_ms': self.last_tokenize_ms,
                'last_infer_ms': self.last_infer_ms,
                'last_request_ms': self.last_request_ms,
                'last_prompt_tokens': self.last_prompt_tokens,
                'last_batch_size': self.last_batch_size,
                'last_error': self.last_error,
                'last_warning': self.last_warning,
            }

    def embed(self, texts: List[str]) -> Tuple[List[List[float]], dict]:
        request_start = time.perf_counter()
        if not self.capacity.acquire(blocking=False):
            msg = f'queue full: max_queue_size={MAX_QUEUE_SIZE}'
            raise EmbeddingOverloadedError(msg)
        try:
            if not texts:
                raise ValueError('input must not be empty')
            wait_start = time.perf_counter()
            with self.stats_lock:
                self.waiting_requests += 1
            got_lock = self.infer_lock.acquire(timeout=QUEUE_TIMEOUT_SECONDS)
            with self.stats_lock:
                self.waiting_requests = max(0, self.waiting_requests - 1)
            queue_wait_ms = (time.perf_counter() - wait_start) * 1000
            if not got_lock:
                with self.stats_lock:
                    self.queue_timeouts += 1
                raise EmbeddingQueueTimeoutError(f'queue wait timed out after {QUEUE_TIMEOUT_SECONDS}s')
            try:
                with self.stats_lock:
                    self.active_infer += 1
                t0 = time.perf_counter()
                enc = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_INPUT_TOKENS,
                    return_tensors='np',
                )
                tok_ms = (time.perf_counter() - t0) * 1000
                input_ids = enc['input_ids'].astype(np.int64)
                attention_mask = enc['attention_mask'].astype(np.int64)
                prompt_tokens = int(attention_mask.sum())
                t1 = time.perf_counter()
                res = self.request.infer({'input_ids': input_ids, 'attention_mask': attention_mask})
                infer_ms = (time.perf_counter() - t1) * 1000
                hidden = list(res.values())[0]
                emb = hidden[:, -1, :]
                emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                emb = emb.astype(np.float32)
            finally:
                with self.stats_lock:
                    self.active_infer = max(0, self.active_infer - 1)
                self.infer_lock.release()
            request_ms = (time.perf_counter() - request_start) * 1000
            is_long = prompt_tokens > LONG_REQUEST_TOKENS
            timeout_warning = request_ms > REQUEST_TIMEOUT_SECONDS * 1000
            warning = None
            if is_long:
                warning = f'long request: prompt_tokens={prompt_tokens} > {LONG_REQUEST_TOKENS}'
            if timeout_warning:
                warning = (warning + '; ' if warning else '') + f'request exceeded timeout budget: {request_ms:.1f}ms'
            meta = {
                'queue_wait_ms': queue_wait_ms,
                'tokenize_ms': tok_ms,
                'infer_ms': infer_ms,
                'request_ms': request_ms,
                'prompt_tokens': prompt_tokens,
                'batch_size': len(texts),
                'is_long_request': is_long,
                'request_timeout_warning': timeout_warning,
                'warning': warning,
            }
            return emb.tolist(), meta
        finally:
            with self.stats_lock:
                self.in_flight = max(0, self.in_flight - 1)
            self.capacity.release()

@app.on_event('startup')
def startup():
    global engine
    engine = Embedder()

@app.get('/health')
def health():
    assert engine is not None
    stats = engine.snapshot()
    return {
        'status': 'ok',
        'model': MODEL_DIR,
        'dimension': 1024,
        'max_input_tokens': MAX_INPUT_TOKENS,
        'devices': engine.available_devices,
        'gpu_name': engine.gpu_name,
        'tokenizer_load_ms': round(engine.tokenizer_load_ms, 3),
        'compile_ms': round(engine.compile_ms, 3),
        'uptime_seconds': round(time.time() - started_at, 3),
        'stats': {k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()},
    }

@app.get('/v1/models')
def models():
    return {'object': 'list', 'data': [{'id': MODEL_DIR, 'object': 'model', 'created': 0, 'owned_by': 'baymax'}]}

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
        'model': MODEL_DIR,
        'data': [{'object': 'embedding', 'index': i, 'embedding': v} for i, v in enumerate(vectors)],
        'usage': {'prompt_tokens': meta['prompt_tokens'], 'total_tokens': meta['prompt_tokens']},
        'meta': {k: round(v, 3) if isinstance(v, float) else v for k, v in meta.items()},
    }
