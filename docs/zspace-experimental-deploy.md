# ZSpace Experimental Deployment

This deployment starts the OpenVINO services as an isolated experiment. It does not modify OpenViking production configuration.

## Ports

- `33038`: experimental embedding sidecar
- `33039`: experimental intent/query-planner sidecar

The current production OpenViking stack is left unchanged.

## Host Directory

```text
<your NAS docker app root>/openviking-openvino-services
```

Model paths in `compose.yml` use `/path/to/your/...` placeholders — replace them with your own model directories before starting.

## Start

```bash
docker compose -f compose.yml up -d
./smoke.sh
```

## Stop

```bash
docker compose -f compose.yml down
```

## Promotion Gate

Only consider OpenViking integration after:

- `embedding /health` is healthy on GPU.
- `embedding /v1/embeddings` returns one 1024-dimensional vector.
- `embedding` queue sizing is tuned so OpenViking retry traffic does not hit a 2s wait timeout under normal search load.
- `intent /health` is healthy on GPU.
- `intent /v1/intent` returns top-level `queries` for a real OpenViking-style prompt.
- Existing production OpenViking `/health` remains healthy.
