# Design Notes — 设计思路与取舍

本文记录这套服务每个关键决策的"为什么"。原则：**正确性优先，其次稳态延迟，最后才是吞吐**。

## 1. 语义策略与能力上限分离（Limit Ownership）

```text
OpenViking ov.conf:   embedding.max_input_tokens = 2048   ← 语义策略（多少源文本算一次 embedding）
Sidecar compose:      MAX_INPUT_TOKENS = 3584             ← 能力护栏（服务侧硬顶，实测稳定上限）
```

为什么不让两者相等？sidecar 是 OpenViking 之后的最后一道闸。若 sidecar 上限低于上游，会在 OpenViking 已完成文本准备后静默截断，掐掉靠后的结论/细节，检索质量偏离上游策略且难以归因。若把 sidecar 提到模型标称上限，N150 iGPU 会更早炸（GPU 单对象分配约 4GiB 上限）。3584 是实测值：3584 token 稳定（20.0s 完成），3712+ 卡死。**上限归属：语义归 OpenViking，护栏归 sidecar，护栏 ≥ 语义上限。**

## 2. CPU tokenizer + GPU model 分层

tokenizer 是前处理，放 CPU 更稳（GPU 上 tokenize 无收益且排障混杂）；OpenVINO 模型图放 iGPU 执行。附带的工程收益：排障时能明确区分是 tokenizer、cache 还是 GPU compile 层的问题。

## 3. 双车道调度器（为什么不是优先级队列/加权公平）

目标函数是"**探针与实时查询永远不被后台提取饿死**"，而不是整体吞吐最优。单 InferRequest 的硬件约束决定了：快请求可以插队等待中的慢任务，但**不能打断已在跑的慢推理**。所以诚实的 SLA 是"最多等完当前一个慢推理"（2048 token 档约 5-6s），不是零延迟。实现选了最简单的双 FIFO（fast ≤256 token / slow >256 token，lane 内 FIFO），因为：

- token 数是**类别边界**不是最短作业优先排序——10/20/5 token 的实时请求都按到达序走快车道，可预测、无饥饿反转
- 多级反馈队列在单执行单元上不产生额外收益，只增加复杂度
- 时间可分解可归因：`queue_wait_ms`（入队→被取走）与 `infer_ms` 分开上报

tokenize 移到入队前是同一次修复的一半：旧设计 tokenize 在锁内，一个 4096-token 的 tokenize 就能占锁 17s。

## 4. Prompt 预算：保头保尾掐中间

错误做法是在渲染完的 chat text 上 `truncation=True, max_length=N` 右截断——这会掐掉尾部（当前消息 + `Output Format` + `Please output JSON`），模型失去输出契约。正确做法是在 apply_chat_template **之前**做显式预算：

- 保头：角色/任务指令、context 类型定义
- 保尾：当前消息、生成标记、JSON 输出契约
- 掐中间：`compression_summary`、较早的冗余消息
- 元数据全暴露：`prompt_tokens_before/after`、`prompt_truncated`、head/tail 保留量

预算值 = min(sidecar MAX_INPUT_TOKENS, tokenizer model_max_length)。**model_max_length 标称值不可信**：本模型标 262144，N150 实际远早于此就 GPU 分配失败。

## 5. 截断必须可见（Truncation Visibility）

任何一层截断都不允许静默：

```text
生成顶格      → finish_reason=length（不再谎报 stop）
截断+JSON 破  → 502 detail: "output truncated at N tokens (MAX_NEW_TOKENS=512); JSON incomplete"
/v1/intent   → 顶层 truncated: true/false
快速人查      → usage.completion_tokens == MAX_NEW_TOKENS 即顶格
embedding    → meta.input_truncated: true/false + WARNING 级 input_truncated 日志事件
```

embedding 侧在 tokenize 后重数一次 token（与 MAX_INPUT_TOKENS 比较），截断既进响应 meta 也进结构化日志——这是对 0.2.0-lanes 时代"静默右截断"盲点的补齐。

## 5b. 结构化日志（2026-08-17 起）

`app/logging_config.py` 提供共享 JSON 日志层（每行一个 JSON 对象，`LOG_FORMAT=text` 切人类可读）：

- **请求访问日志**：中间件记录 method / **真实 path（非路由模板）** / status / duration_ms，并生成或透传 `X-Request-ID`（响应头回显）。uvicorn access log 被显式降级避免重复
- **request_id 跨层传播**：contextvars 实现；FastAPI sync endpoint 的线程池会复制 context，engine 层日志自动带上同一 request_id（注意：裸 `threading.Thread` 不复制 contextvars——内部 infer worker 的日志不带请求 id，属预期）
- **关键事件**：`embed_ok`（INFO，含 lane/token/infer/queue 耗时拆解）、`input_truncated` / `prompt_truncated` / `queue_timeout` / `queue_full`（WARNING）
- 日志里只放 `input_preview`（redact 截断到 120 字符），不落完整 prompt
- **输入预算全覆盖**（1.1.0 起）：`/v1/intent` 的 `plan()` 与 `complete_prompt()` 走同一条 `_truncate_prompt_preserving_edges` 预算路径（此前 plan() 未限长直送 iGPU，且 prompt_truncated 恒 False）
- **结果可见性**（1.0.1 起）：`embed_ok.result_digest` = `{dim, norm, sha8}` 向量指纹（确定性：同输入必同指纹，用于验证可复现性，不往日志里灌 1024 维浮点）；`intent_ok.plan_preview` = 生成的查询计划 JSON 前 400 字符（redact）

理由：截断不可见时，故障表现为"模型输出质量下降"，会误导去怀疑模型/量化，而真实原因只是预算不够。诊断规则：**JSON parse 失败且 generated_tokens == MAX_NEW_TOKENS → 先加输出预算**。

## 5c. 日志查询接口与中文日志台（1.0.0 起）

- 每个服务暴露 `GET /v1/logs`：`limit/level/event/request_id/q`（关键词）过滤，newest-first，数据源为进程内 ring buffer（`LOG_BUFFER_SIZE` 默认 2000 条，重启即清——持久化仍以 docker logs 为准）
- `dashboard/` 为纯静态中文日志台（无构建步骤、无依赖）：服务切换、级别/事件/RequestID/关键词筛选、5s 自动刷新；点日志中的 Request ID 即可按该请求过滤
- 部署形态：nginx:alpine 容器（33050）同源反代 33038/33039（`/emb`、`/int`），规避浏览器 CORS，仓库内 dashboard 用相对路径不含任何内网 IP
- 版本规范：`app/version.py` 单一来源（semver），两服务与日志接口统一汇报；发版流程 = 改 version.py → tag `vX.Y.Z` → 同 tag 构建镜像

## 6. 镜像结构：共享 base + 二级薄服务镜像

```text
openviking-openvino-base:2026.3            ← 统一依赖层（CPU-only torch、protobuf 等）
├── openviking-openvino-embedding:2026.3   ← 一级服务镜像
│   └── :0.2.0-lanes                       ← 二级热修镜像（FROM 一级，COPY 顶层 py）
└── openviking-openvino-intent:2026.3
    └── :edge-preserve-budget-20260816     ← 二级热修镜像
```

共享 base 让依赖升级一次生效；服务层隔离故障半径（embedding 崩不牵连 intent）；二级镜像让热修只重传 3-4 个 py 文件。代价：镜像 tag 需要纪律——**每次二级构建必须同步回 GitHub 仓库**（本仓库 2026-08-17 曾漂移两天才补同步，见 `docs/production-runbook.md`）。

## 7. INT8 留生产的量化决策

同硬件同 prompt 实测：INT4 仅快 ~2%（p50 7.02s vs 7.18s），质量与 JSON 合法率持平。2% 换量化风险不划算，INT8 留生产，INT4 存 NAS `baymax-igpu-lab/hf-v7-intent-analysis-sft-openvino-int4` 作低内存备件。INT4 从验证过的 FP16 IR 重新量化（不是从 Q8 二压），配置 `bits=4, sym=False, group_size=128`。

## 8. 验证分层（Promotion Gate）

"能 compile"≠"能 serve"。四层验证缺一不可：

1. Python 语法（compileall）
2. 本地引擎契约测试（无需模型/HTTP）
3. 目标硬件端到端 `generate()`（真实 GPU 路径）
4. HTTP smoke（服务真实可用）

生产切换前另加：`/health` + `/ready` + 真实 search 三点验证，且**先只读烟测再改路由**。

## 9. 运维红线

- `/health` 不足以判活——必须 `/ready` + 短 embedding probe + 真实 search
- 能力上限测试必须走隔离临时容器（如 33040），不得直打生产端口
- OpenViking QueueFS 积压导致的 503 单重启不解决，需 flush queue.db（备份→删 Embedding pending/processing→重启 sidecar→再启 OpenViking）
- compose 改动整文件替换，禁止正则重写 service 块
- Mac(arm64) 构建必须 `--platform linux/amd64`
