import json
import time
from pathlib import Path
from typing import List, Union

import numpy as np
import openvino as ov
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from .config import MODEL_ID, MODEL_DIR, OPENVINO_DEVICE, MAX_INPUT_TOKENS, MAX_NEW_TOKENS, TEMPERATURE

app = FastAPI(title='OpenViking OpenVINO Intent Sidecar', version='0.1.0')
engine = None
started_at = time.time()

PROMPT_TEMPLATE = Path(__file__).with_name('ov_intent_analysis_sft_v7.yaml').read_text()

class IntentRequest(BaseModel):
    model: str = Field(default=MODEL_ID)
    input: Union[str, List[str]]
    recent_messages: str = ''
    compression_summary: str = ''
    context_type: str = ''
    target_abstract: str = ''

class IntentEngine:
    def __init__(self):
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
        self.tokenizer_load_ms = (time.perf_counter() - t0) * 1000
        self.core = ov.Core()
        self.available_devices = list(self.core.available_devices)
        self.gpu_name = self.core.get_property(OPENVINO_DEVICE, 'FULL_DEVICE_NAME') if OPENVINO_DEVICE in self.available_devices else None
        model = self.core.read_model(f'{MODEL_DIR}/openvino_language_model.xml')
        t1 = time.perf_counter()
        self.compiled = self.core.compile_model(model, OPENVINO_DEVICE)
        self.compile_ms = (time.perf_counter() - t1) * 1000
        self.request = self.compiled.create_infer_request()

    def render_prompt(self, req: IntentRequest, message: str) -> str:
        text = PROMPT_TEMPLATE
        text = text.replace('{{ compression_summary }}', req.compression_summary.strip())
        text = text.replace('{{ recent_messages }}', req.recent_messages.strip())
        text = text.replace('{{ current_message }}', message.strip())
        text = text.replace('{{ context_type }}', req.context_type.strip())
        text = text.replace('{{ target_abstract }}', req.target_abstract.strip())
        return text

    def plan(self, req: IntentRequest, message: str) -> tuple[dict, dict]:
        prompt = self.render_prompt(req, message)
        text = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = self.tokenizer(text, truncation=True, max_length=MAX_INPUT_TOKENS, return_tensors='np')
        inputs = {
            'inputs_embeds': np.zeros((1, 1, 1024), dtype=np.float32),
            'attention_mask': np.ones((1, 1), dtype=np.int64),
            'position_ids': np.zeros((4, 1, 1), dtype=np.int64),
            'beam_idx': np.array([0], dtype=np.int64),
        }
        t0 = time.perf_counter()
        out = self.request.infer(inputs)
        infer_ms = (time.perf_counter() - t0) * 1000
        logits = list(out.values())[0]
        meta = {
            'tokenizer_load_ms': round(self.tokenizer_load_ms, 3),
            'compile_ms': round(self.compile_ms, 3),
            'infer_ms': round(infer_ms, 3),
            'available_devices': self.available_devices,
            'gpu_name': self.gpu_name,
            'max_input_tokens': MAX_INPUT_TOKENS,
            'max_new_tokens': MAX_NEW_TOKENS,
            'temperature': TEMPERATURE,
            'input_shape': list(enc['input_ids'].shape),
            'logits_shape': list(np.asarray(logits).shape),
            'prompt_preview': prompt[:240],
        }
        # This remains a service shell: the generation loop still needs a real
        # decode path aligned with the exported model's contract.
        return {'queries': [], 'reasoning': ''}, meta

@app.on_event('startup')
def startup():
    global engine
    engine = IntentEngine()

@app.get('/health')
def health():
    assert engine is not None
    return {
        'status': 'ok',
        'model': MODEL_ID,
        'devices': engine.available_devices,
        'gpu_name': engine.gpu_name,
        'tokenizer_load_ms': round(engine.tokenizer_load_ms, 3),
        'compile_ms': round(engine.compile_ms, 3),
        'uptime_seconds': round(time.time() - started_at, 3),
    }

@app.post('/v1/intent')
def intent(req: IntentRequest):
    assert engine is not None
    if req.model != MODEL_ID:
        raise HTTPException(status_code=404, detail=f'model not found: {req.model}')
    prompt_text = req.input if isinstance(req.input, str) else '\n'.join(req.input)
    try:
        plan, meta = engine.plan(req, prompt_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {'object': 'intent', 'model': MODEL_ID, 'plan': plan, 'meta': meta}
