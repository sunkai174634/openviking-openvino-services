# OpenViking OpenVINO Services

在 Intel N150 iGPU 的 NAS 上，为 [OpenViking](https://github.com/volcengine/OpenViking) 提供本地 embedding 与 intent（query planner）推理服务的 sidecar 仓库：一张共享 OpenVINO base 镜像、两个薄服务、一套实测驱动的调度与预算策略。

> [!NOTE]
> 本项目**全程采用 Vibe Coding**——从代码、Dockerfile 到文档，由 AI agent（Hermes/BayMax，GLM 加持）在人类架构决策下结对完成：人负责方向、验收与取舍拍板，AI 负责实现、测量与排障。文中所有性能数字都来自真实测量，而非 LLM 生成值。

## 为什么做这个

OpenViking 的检索管线依赖两类模型：embedding（查询/文档向量化）和 query planner（把自然语言改写成结构化检索计划）。在纯本地方案里，这两类请求的延迟与稳定性直接决定检索体验。本项目把两者放到 NAS 的 N150 iGPU 上，实测调优到可长期生产，并把过程中踩过的每个坑记录成可复用的结论。

## 服务架构

```text
OpenViking 实例
├── embedding 路由 ──► /v1/embeddings   (端口 33038)  Qwen3-Embedding-0.6B INT8
└── query_planner 路由 ► /v1/intent 等   (端口 33039)  ov_intent_analysis_sft v7 INT8
        （OpenAI / Ollama / 原生三类协议兼容）
```

镜像分层：`openviking-openvino-base`（统一依赖，CPU-only torch + OpenVINO 2026.3）→ 每服务一级镜像 → 热修二级镜像（FROM 一级 + COPY 少量 py，秒级重建）。

## 模型选型

### Embedding：Qwen3-Embedding-0.6B（OpenVINO 官方 INT8 IR）

- **为什么是它**：OpenViking 上游按 1024 维向量库设计，Qwen3-Embedding 系列原生 1024 维，与现有向量库零迁移成本；0.6B 体量在 N150 iGPU 上可接受
- **为什么用 OpenVINO 官方转好的 INT8 IR**：优先"适配最优/官方已转好格式"，不自造量化；官方 IR 的 KV cache 和子图切分质量优于自行转换
- 实测单请求 256 token 内推理 ~84ms

### Intent（query planner）：guoxuter/ov_intent_analysis_sft v7 INT8

- **为什么是它**：v7 是 OpenViking 官方 SFT 的 query planner 模型（官方提供 GGUF/OpenVINO 发行），prompt 模板与 OpenViking `IntentAnalyzer` 契约对齐，避免自训模型与上游 prompt 演进脱节
- **量化档位**：INT8 留生产。INT4 实测仅快 ~2%（p50 7.02s vs 7.18s）且质量持平，2% 不值得量化风险 → INT4 留作低内存备件
- 关键契约对齐：`TEMPERATURE=0.1`（v7 yaml 中 `llm_config.temperature`），`MAX_NEW_TOKENS=512`（实测最低安全线 192，生产推荐 256+）

## Mac vs NAS 实测对比

同一模型、同一代码、同一 prompt：

### Intent 服务

| 环境 | 设备 | tokenizer 载入 | 模型载入 | generate（1065 tok prompt, 93 tok 输出） |
|---|---|---:|---:|---:|
| Mac (M-series, CPU) | CPU | 686ms | 7.3s | **4.75s** |
| NAS (N150 iGPU) | GPU | 2.0s | 15.8s | **12.34s** |

### Embedding（raw 单请求）

| 路径 | Mac | NAS |
|---|---:|---:|
| raw embedding（256 tok 内） | ~70ms | ~330-400ms |
| OpenViking 端到端 `find` | avg 237ms | avg 460-497ms |

**读法**：raw 层面 Mac 快约 5 倍；但 OpenViking 端到端里固定开销摊薄后，NAS 约慢 2 倍。选择 NAS 的理由不是速度，而是：24×7 低功耗常驻（Mac 得一直插电开盖）、与 OpenViking 同机部署零跨机流量、家庭内网零外发。等待时间的代价换来完全本地、可长期运行的检索基础设施。

## 核心设计

<details>
<summary><b>双车道调度器（修复 /ready 503 饥饿）</b></summary>

旧设计（每个请求线程竞争一把 10s 超时的锁，tokenize 在锁内）下，单个 ~2048 token 后台提取就能饿死所有就绪探针 → OpenViking /ready 永久 503。重写为：单 worker 线程独占 InferRequest；tokenize 提前到入队前（CPU 侧）；快车道（≤256 token 探针/实时查询，等待上限 10s）永远先于慢车道（后台提取，上限 120s）排空。验收：6 并发 2048-token 提取 + 每 1.5s 探针 → 6/6 探针通过（最长等待 4.9s），长任务全部完成。
</details>

<details>
<summary><b>Intent prompt 预算：保头保尾掐中间</b></summary>

对渲染完的 chat text 做 `truncation=True` 右截断会把尾部的 JSON 输出契约连同当前消息一起截掉 → 模型不再输出 JSON → search 500。正确姿势：在 apply_chat_template 之前做显式预算，保头（任务指令）保尾（当前消息 + Output Format），掐中间（压缩摘要等旧上下文），meta 全量暴露截断前后 token 数。
</details>

<details>
<summary><b>截断可见化</b></summary>

生成顶格 → `finish_reason=length`（不再谎报 stop）；截断+JSON 破 → 502 detail 明示 "output truncated at N tokens"；`/v1/intent` 顶层 `truncated` 字段。诊断规则：JSON parse 失败且 `generated_tokens == MAX_NEW_TOKENS` → 先怀疑输出截断，加预算再谈模型质量。
</details>

<details>
<summary><b>语义策略与能力护栏分离</b></summary>

OpenViking 层 `embedding.max_input_tokens=2048` 是语义策略（多少源文本算一次 embedding）；sidecar `MAX_INPUT_TOKENS=3584` 是能力护栏（N150 实测稳定上限，3712+ 卡死）。护栏必须 ≥ 上游语义上限，否则会在上游完成文本准备后静默截断。
</details>

完整的取舍记录（含 12 项踩坑表）、运维数据与故障处置手册见：

- [docs/design.md](docs/design.md) — 每个关键决策的"为什么"
- [docs/production-runbook.md](docs/production-runbook.md) — 实测数据、健康判定三层、QueueFS flush 流程

## 快速开始

```bash
# 1. 准备模型（NAS 上）
#    embedding: Qwen3-Embedding-0.6B OpenVINO INT8 IR
#    intent:    ov_intent_analysis_sft v7 OpenVINO INT8 IR
# 2. 构建镜像（Mac/任何 amd64 构建机）
docker buildx build --platform linux/amd64 --load -t openviking-openvino-base:2026.3 .
docker buildx build --platform linux/amd64 --load -f Dockerfile.embedding -t openviking-openvino-embedding:2026.3 .
docker buildx build --platform linux/amd64 --load -f Dockerfile.intent -t openviking-openvino-intent:2026.3 .
# 3. 部署（按 deploy/zspace/compose.yml 模板，替换你的模型路径）
# 4. 验证
./deploy/zspace/smoke.sh
```

## 已知限制

- 单 InferRequest 串行执行：快请求可插队等待中的慢任务，但不能打断已在跑的慢推理
- 模型与 OpenViking v7 prompt 契约绑定，上游 prompt 演进时需同步对齐采样参数
- N150 iGPU 的单对象显存分配约 4GiB 上限，长输入预算受硬件约束

## 贡献

欢迎 issue / PR。分享你的部署硬件与实测数据（复现本仓库的测量方法）比泛泛的 star 更有价值。

## 致谢

- [OpenViking](https://github.com/volcengine/OpenViking) 官方团队——从 sidecar API 契约答疑到 query planner 模型训练与 SFT 发行支持，本项目大量依赖他们的工作。特别感谢 **@郭昊** 与 **@秦浩杰** 在答疑和模型训练/调优方向的持续支持。
- [OpenVINO™ Toolkit](https://github.com/openvinotoolkit/openvino) 与 Intel — INT8 IR 与 iGPU runtime
- Qwen 团队 — Qwen3-Embedding 基座

## License

MIT
