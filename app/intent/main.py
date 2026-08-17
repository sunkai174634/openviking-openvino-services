import json
import time
from datetime import datetime, timezone
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .config import MODEL_ID
from .engine import IntentEngine

app = FastAPI(title='OpenViking OpenVINO Intent Sidecar', version='0.3.0')
engine = None
started_at = time.time()


class IntentRequest(BaseModel):
    model: str = Field(default=MODEL_ID)
    input: Union[str, List[str]]
    recent_messages: str = ''
    compression_summary: str = ''
    context_type: str = ''
    target_abstract: str = ''


class ChatMessage(BaseModel):
    role: str
    content: str = ''


class OllamaChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    format: Optional[str] = None
    keep_alive: Optional[str] = None
    think: Optional[bool] = None
    options: dict = Field(default_factory=dict)
    tools: list = Field(default_factory=list)


class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str = ''
    system: Optional[str] = None
    template: Optional[str] = None
    stream: bool = False
    format: Optional[str] = None
    keep_alive: Optional[str] = None
    think: Optional[bool] = None
    options: dict = Field(default_factory=dict)
    images: list = Field(default_factory=list)


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
        'model_load_ms': round(engine.model_load_ms, 3),
        'uptime_seconds': round(time.time() - started_at, 3),
    }


@app.get('/api/tags')
def ollama_tags():
    return {'models': [{'name': MODEL_ID, 'model': MODEL_ID, 'modified_at': datetime.now(timezone.utc).isoformat()}]}


def _messages_to_prompt(messages: List[ChatMessage]) -> str:
    return '\n'.join(m.content for m in messages if m.content).strip()


def _json_text(plan: dict) -> str:
    return json.dumps({'queries': plan.get('queries', [])}, ensure_ascii=False)


@app.post('/v1/intent')
def intent(req: IntentRequest):
    assert engine is not None
    if req.model != MODEL_ID:
        raise HTTPException(status_code=404, detail=f'model not found: {req.model}')
    message = req.input if isinstance(req.input, str) else '\n'.join(req.input)
    try:
        plan, meta = engine.plan(
            current_message=message,
            recent_messages=req.recent_messages,
            compression_summary=req.compression_summary,
            context_type=req.context_type,
            target_abstract=req.target_abstract,
        )
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {'object': 'intent', 'model': MODEL_ID, 'plan': plan, 'meta': meta, 'truncated': bool(meta.get('truncated'))}


@app.post('/api/chat')
def ollama_chat(req: OllamaChatRequest):
    assert engine is not None
    if req.stream:
        raise HTTPException(status_code=400, detail='streaming is not supported')
    prompt = _messages_to_prompt(req.messages)
    meta: dict = {}
    try:
        raw_text, meta = engine.complete_prompt(prompt)
        plan = engine.normalize_plan(engine.parse_json(raw_text))
        content = _json_text(plan)
    except ValueError as e:
        if meta.get('truncated'):
            raise HTTPException(
                status_code=502,
                detail=f'output truncated at {meta.get("generated_tokens")} tokens (MAX_NEW_TOKENS={meta.get("max_new_tokens")}); JSON incomplete',
            ) from e
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {
        'model': req.model,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'message': {'role': 'assistant', 'content': content},
        'done': True,
        'done_reason': 'length' if meta.get('truncated') else 'stop',
        'total_duration': int(meta.get('generate_ms', 0) * 1_000_000),
        'eval_count': meta.get('generated_tokens', 0),
        'prompt_eval_count': meta.get('prompt_tokens', 0),
    }


@app.post('/api/generate')
async def ollama_generate(request: Request):
    assert engine is not None
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail='JSON object required')
    if payload.get('stream') is True:
        raise HTTPException(status_code=400, detail='streaming is not supported')
    model = str(payload.get('model') or MODEL_ID)
    prompt = str(payload.get('prompt') or '')
    try:
        raw_text, meta = engine.complete_prompt(prompt)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {
        'model': model,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'response': raw_text,
        'done': True,
        'done_reason': 'length' if meta.get('truncated') else 'stop',
        'total_duration': int(meta.get('generate_ms', 0) * 1_000_000),
        'eval_count': meta.get('generated_tokens', 0),
        'prompt_eval_count': meta.get('prompt_tokens', 0),
    }


@app.post('/v1/chat/completions')
def openai_chat(req: OpenAIChatRequest):
    assert engine is not None
    if req.stream:
        raise HTTPException(status_code=400, detail='streaming is not supported')
    prompt = _messages_to_prompt(req.messages)
    meta: dict = {}
    try:
        raw_text, meta = engine.complete_prompt(prompt)
        plan = engine.normalize_plan(engine.parse_json(raw_text))
        content = _json_text(plan)
    except ValueError as e:
        if meta.get('truncated'):
            raise HTTPException(
                status_code=502,
                detail=f'output truncated at {meta.get("generated_tokens")} tokens (MAX_NEW_TOKENS={meta.get("max_new_tokens")}); JSON incomplete',
            ) from e
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {
        'id': f'chatcmpl-openvino-{int(time.time())}',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': req.model,
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'length' if meta.get('truncated') else 'stop'}],
        'usage': {'prompt_tokens': meta.get('prompt_tokens', 0), 'completion_tokens': meta.get('generated_tokens', 0), 'total_tokens': meta.get('prompt_tokens', 0) + meta.get('generated_tokens', 0)},
    }
