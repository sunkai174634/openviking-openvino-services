import json
import logging
import os
import re
import time
from pathlib import Path
from threading import Lock
from typing import Any

import openvino as ov
import yaml
from jinja2 import Template
from optimum.intel import OVModelForVisualCausalLM
from transformers import AutoTokenizer

from .config import MODEL_DIR, OPENVINO_DEVICE, MAX_INPUT_TOKENS, MAX_NEW_TOKENS, TEMPERATURE

logger = logging.getLogger(__name__)

def _load_prompt_path() -> Path:
    configured = os.environ.get('PROMPT_PATH')
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend([
        Path(__file__).with_name('ov_intent_analysis_sft_v7.yaml'),
        Path(__file__).parents[2] / 'ov_intent_analysis_sft_v7.yaml',
    ])
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f'v7 prompt template not found; checked: {candidates}')


PROMPT_PATH = _load_prompt_path()
PROMPT_SPEC = yaml.safe_load(PROMPT_PATH.read_text())
PROMPT_TEMPLATE = Template(PROMPT_SPEC['template'])


class IntentEngine:
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
        self.gpu_name = (
            self.core.get_property(OPENVINO_DEVICE, 'FULL_DEVICE_NAME')
            if OPENVINO_DEVICE in self.available_devices
            else None
        )
        t1 = time.perf_counter()
        self.model = OVModelForVisualCausalLM.from_pretrained(
            MODEL_DIR,
            device=OPENVINO_DEVICE,
            local_files_only=True,
            ov_config={'CACHE_DIR': os.environ.get('OPENVINO_CACHE_DIR', '/tmp/openviking-openvino-cache')},
        )
        self.model_load_ms = (time.perf_counter() - t1) * 1000
        self.generate_lock = Lock()
        self.model_max_tokens = self._resolve_model_max_tokens(self.tokenizer)
        self.prompt_budget_tokens = min(MAX_INPUT_TOKENS, self.model_max_tokens)

    @staticmethod
    def _resolve_model_max_tokens(tokenizer: Any) -> int:
        limit = getattr(tokenizer, 'model_max_length', MAX_INPUT_TOKENS)
        if not isinstance(limit, int) or limit <= 0 or limit > 1_000_000:
            return MAX_INPUT_TOKENS
        return limit

    def _token_count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _truncate_prompt_preserving_edges(self, prompt: str) -> tuple[str, dict[str, Any]]:
        prompt = prompt.strip()
        before_tokens = self._token_count(prompt)
        meta = {
            'prompt_budget_tokens': self.prompt_budget_tokens,
            'prompt_tokens_before': before_tokens,
            'prompt_tokens_after': before_tokens,
            'prompt_truncated': False,
        }
        if before_tokens <= self.prompt_budget_tokens:
            return prompt, meta

        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        head_keep = max(256, self.prompt_budget_tokens // 4)
        tail_keep = max(self.prompt_budget_tokens - head_keep, 0)
        if head_keep > self.prompt_budget_tokens:
            head_keep = self.prompt_budget_tokens
            tail_keep = 0

        def rebuild(h_keep: int, t_keep: int) -> str:
            if t_keep <= 0:
                ids = tokens[:h_keep]
            else:
                ids = tokens[:h_keep] + tokens[-t_keep:]
            return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

        truncated = rebuild(head_keep, tail_keep)
        after_tokens = self._token_count(truncated)
        while after_tokens > self.prompt_budget_tokens and (head_keep > 128 or tail_keep > 128):
            if tail_keep >= head_keep and tail_keep > 128:
                tail_keep = max(128, tail_keep - max(64, tail_keep // 8))
            elif head_keep > 128:
                head_keep = max(128, head_keep - max(64, head_keep // 8))
            else:
                break
            truncated = rebuild(head_keep, tail_keep)
            after_tokens = self._token_count(truncated)

        meta.update({
            'prompt_tokens_after': after_tokens,
            'prompt_truncated': True,
            'prompt_head_tokens_kept': head_keep,
            'prompt_tail_tokens_kept': tail_keep,
        })
        logger.warning(
            '[IntentEngine] Prompt truncated: before=%s after=%s budget=%s head=%s tail=%s',
            before_tokens,
            after_tokens,
            self.prompt_budget_tokens,
            head_keep,
            tail_keep,
        )
        return truncated, meta

    @staticmethod
    def parse_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        decoder = json.JSONDecoder()
        try:
            value, _ = decoder.raw_decode(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find('{')
            if start < 0:
                raise ValueError('intent model returned no JSON object')
            value, _ = decoder.raw_decode(cleaned[start:])
        if not isinstance(value, dict):
            raise ValueError('intent model JSON root must be an object')
        return value

    @staticmethod
    def normalize_plan(value: dict[str, Any]) -> dict[str, Any]:
        queries = value.get('queries')
        if not isinstance(queries, list) and isinstance(value.get('reasoning'), list):
            queries = value['reasoning']
        if not isinstance(queries, list):
            queries = []
        normalized = []
        for item in queries[:5]:
            if not isinstance(item, dict) or not isinstance(item.get('query'), str):
                continue
            query = item['query'].strip()
            if not query:
                continue
            row = {'query': query}
            if item.get('context_type') in {'skill', 'resource', 'memory'}:
                row['context_type'] = item['context_type']
            if isinstance(item.get('priority'), int) and 1 <= item['priority'] <= 5:
                row['priority'] = item['priority']
            if isinstance(item.get('intent'), str) and item['intent'].strip():
                row['intent'] = item['intent'].strip()
            normalized.append(row)
        reasoning = value.get('reasoning', '')
        if not isinstance(reasoning, str):
            reasoning = ''
        return {'reasoning': reasoning, 'queries': normalized}

    def render_prompt(
        self,
        *,
        compression_summary: str,
        recent_messages: str,
        current_message: str,
        context_type: str = '',
        target_abstract: str = '',
    ) -> str:
        return PROMPT_TEMPLATE.render(
            compression_summary=compression_summary.strip(),
            recent_messages=recent_messages.strip(),
            current_message=current_message.strip(),
            context_type=context_type.strip(),
            target_abstract=target_abstract.strip(),
        )

    def _generate_from_chat_text(self, chat_text: str, temperature: float | None = None) -> tuple[str, dict[str, Any]]:
        inputs = self.tokenizer(
            chat_text,
            truncation=False,
            return_tensors='pt',
        )
        effective_temperature = TEMPERATURE if temperature is None else float(temperature)
        generate_kwargs = {
            'max_new_tokens': MAX_NEW_TOKENS,
            'do_sample': effective_temperature > 0,
        }
        if effective_temperature > 0:
            generate_kwargs['temperature'] = effective_temperature
        with self.generate_lock:
            t0 = time.perf_counter()
            generated = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
            generate_ms = (time.perf_counter() - t0) * 1000
        prompt_len = inputs['input_ids'].shape[1]
        new_tokens = generated[0, prompt_len:]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        meta = {
            'tokenizer_load_ms': round(self.tokenizer_load_ms, 3),
            'model_load_ms': round(self.model_load_ms, 3),
            'generate_ms': round(generate_ms, 3),
            'available_devices': self.available_devices,
            'gpu_name': self.gpu_name,
            'device': OPENVINO_DEVICE,
            'max_input_tokens': MAX_INPUT_TOKENS,
            'max_new_tokens': MAX_NEW_TOKENS,
            'temperature': effective_temperature,
            'prompt_tokens': int(prompt_len),
            'generated_tokens': int(new_tokens.shape[0]),
            'truncated': bool(new_tokens.shape[0] >= MAX_NEW_TOKENS),
        }
        return raw_text, meta

    def complete_prompt(self, prompt: str, temperature: float | None = None) -> tuple[str, dict[str, Any]]:
        budgeted_prompt, budget_meta = self._truncate_prompt_preserving_edges(prompt)
        chat_text = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': budgeted_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        raw_text, meta = self._generate_from_chat_text(chat_text, temperature=temperature)
        meta.update(budget_meta)
        return raw_text, meta

    def plan(
        self,
        *,
        current_message: str,
        recent_messages: str = '',
        compression_summary: str = '',
        context_type: str = '',
        target_abstract: str = '',
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt = self.render_prompt(
            compression_summary=compression_summary,
            recent_messages=recent_messages,
            current_message=current_message,
            context_type=context_type,
            target_abstract=target_abstract,
        )
        chat_text = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        raw_text, meta = self._generate_from_chat_text(chat_text)
        plan = self.normalize_plan(self.parse_json(raw_text))
        return plan, meta
