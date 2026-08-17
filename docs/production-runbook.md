# Production Runbook — 运维数据与同步规约

本服务的生产事实与操作规程。**真源永远是运行中的容器；GitHub main 是源码权威副本。**

## 生产拓扑（2026-08-17）

```text
OpenViking 实例 (端口 33031)
├── embedding 路由 ──► sidecar :33038  image openviking-openvino-embedding:0.2.0-lanes
└── query_planner ──► sidecar :33039  image openviking-openvino-intent:edge-preserve-budget-20260816

ZSpace 部署根: <你的 NAS Docker 应用根>/openviking-openvino-services/
模型挂载:
  embedding: <模型根>/Qwen3-Embedding-0.6B-int8-ov
  intent:    <模型根>/ov_intent_analysis_sft_int8_ov (+可写 OpenVINO cache)
```

## 生产环境变量（实测自运行容器）

```text
embedding (33038):
  MAX_INPUT_TOKENS=3584  MAX_QUEUE_SIZE=64  QUEUE_TIMEOUT_SECONDS=10
  LONG_QUEUE_TIMEOUT_SECONDS=120  REQUEST_TIMEOUT_SECONDS=60
  SHORT_REQUEST_TOKENS=256  LONG_REQUEST_TOKENS=2048  restart=unless-stopped

intent (33039):
  MAX_INPUT_TOKENS=4096  MAX_NEW_TOKENS=512  TEMPERATURE=0.1
```

## 关键性能数据（N150 iGPU 实测）

### Embedding 输入上限扫描（隔离容器，2026-08-16）

| prompt_tokens | infer_ms | wall_ms | 结果 |
|---:|---:|---:|---|
| 512 | 1,353 | 1,394 | OK |
| 1024 | 3,183 | 3,215 | OK |
| 1536 | 5,538 | 5,574 | OK |
| 2048 | 8,411 | 8,446 | OK |
| 2560 | 11,819 | 12,035 | OK |
| 3072 | 15,749 | 15,874 | OK |
| 3200 | 17,033 | 17,081 | OK |
| 3328 | 17,894 | 17,981 | OK |
| 3456 | 19,257 | 19,464 | OK |
| 3584 | 20,056 | 20,185 | OK ← 生产护栏来源 |
| 3712 | >120s | >120s | 超时/卡死 |

### 调度器验收（lanes 上线时）

6 并发 2048-token 长提取 + 每 1.5s 探针：6/6 探针通过（最长等待 4.9s），全部长任务完成，无队列超时。

### Intent 输出预算扫描（2026-08-17，INT8/INT4 双档）

预算 128→320 扫描，最低安全线 192，生产推荐 256（现配 512）。实测输出：INT8 health_check 137 / multi_intent 161；INT4 133-139 token。INT4 仅快 ~2%（p50 7.02 vs 7.18s）→ INT8 留产。

### 全链路 search（生产路由，2026-08-15 切换验证）

simple 12.9s / complex 10.5s / negative 10.9s，均 200 且命中知识库。intent 直连 ~7.3s 是主瓶颈，embedding ~153ms。

## 健康判定三层

```bash
curl -s http://<sidecar-host>:33038/health   # sidecar 进程活
curl -s http://<openviking-host>:33031/ready # OpenViking 就绪（含 embedding probe）
# + 真实 search 必跑一次才算恢复
```

`/health` OK 但 `/v1/embeddings` 500（CL_OUT_OF_RESOURCES）的情况真实发生过——**/health 不判 GPU 推理可用性**。

## 故障处置

### QueueFS 积压 → /ready 503（flush 流程）

```bash
# 1. 停 OpenViking 断粮
docker compose -f <openviking-compose> stop openviking
# 2. 备份 queue.db
QDB=<openviking-data>/data/_system/queue/queue.db
cp "$QDB" "${QDB}.bak-before-flush"
# 3. 删 Embedding 队列 pending/processing 消息（不碰向量/内容库）
sqlite3 "$QDB" "DELETE FROM queue_messages WHERE queue_name='Embedding' AND status IN ('pending','processing');"
# 4. 重启 sidecar 清 in_flight → 起来后 /health ok
# 5. 再启 OpenViking，/ready 应 200
```

注意：只重启 OpenViking 无效（积压重放）；`embedding.max_concurrent=1` 写在 ov.conf 顶层 `embedding` 下（`embedding.dense.max_concurrent` 是非法字段，曾致 OpenViking 拒启）。

### GPU 卡死（CL_OUT_OF_RESOURCES）

重启 embedding 容器可清 wedged 状态，但**必须**复跑 `/health` + `/ready` + 真实 search 才算恢复。

### JSON parse 失败 / search 500

先查 intent 截断：`usage.completion_tokens == MAX_NEW_TOKENS` 或 meta.truncated → 加输出预算，别先怪模型。

## 热修与同步规约（本仓库核心纪律）

1. 现场任何热修（sidecar src/compose 就地改）完成后，**必须**同步回 GitHub main
2. 同步方法：拉回 src → 与容器内 `/srv/openviking-openvino/{embedding,intent}/*.py` 的 sha256 比对 → 覆盖仓库文件 → pytest → commit + push
3. compose 真源是现场 `compose.yml`；仓库 `deploy/zspace/compose.yml` 跟随现场值
4. 二级镜像（lanes / edge-preserve-budget）构建后同步回仓库，标签日期即构建日期

### 2026-08-17 漂移事件（前车之鉴）

GitHub main 曾停在 8-15，生产已跑到 0.2.0-lanes + edge-preserve-budget，源码一度只存在临时目录。当日补同步（SHA256 逐文件核验 + 8/8 测试通过）。**教训：热修当天就 push，别攒。**

## 构建与部署

```bash
# Mac (arm64) 构建 amd64
docker buildx build --platform linux/amd64 --load -t <tag> .

# 二级热修镜像（NAS 上构建，或同法 Mac 构建）
# embedding: FROM openviking-openvino-embedding:2026.3 + COPY 3 个 py
# intent:    FROM openviking-openvino-intent:2026.3 + COPY 4 个 py + yaml

# 重建单个服务
docker compose -f <root>/compose.yml up -d --no-deps --force-recreate intent
```

Promotion gate：`/health` + `/ready` + 真实 search + 只读烟测先行，改路由最后。

## 模型资产

| 模型 | 用途 |
|---|---|
| Qwen3-Embedding-0.6B OpenVINO INT8 | embedding 生产 |
| ov_intent_analysis_sft v7 INT8 | intent 生产 |
| ov_intent_analysis_sft v7 INT4 | 备用（NNCF bits=4 sym=False group_size=128） |
