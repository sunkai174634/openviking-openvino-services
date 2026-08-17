#!/usr/bin/env sh
set -eu

embedding_url=${EMBEDDING_URL:-http://127.0.0.1:33038}
intent_url=${INTENT_URL:-http://127.0.0.1:33039}

echo '== embedding health =='
curl -fsS "$embedding_url/health" | python3 -m json.tool

echo '== embedding request =='
curl -fsS \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding-0.6b-openvino-int8","input":"OpenViking embedding smoke test"}' \
  "$embedding_url/v1/embeddings" \
  | python3 -c 'import json,sys; obj=json.load(sys.stdin); print({"model": obj["model"], "count": len(obj["data"]), "dimension": len(obj["data"][0]["embedding"]), "meta": obj.get("meta")})'

echo '== intent health =='
curl -fsS "$intent_url/health" | python3 -m json.tool

echo '== intent request =='
curl -fsS --max-time 120 \
  -H 'Content-Type: application/json' \
  -d '{"model":"guoxuter/ov_intent_analysis_sft:v7_q8","input":"OpenViking /ready 503 那次怎么恢复的？"}' \
  "$intent_url/v1/intent" \
  | python3 -m json.tool
