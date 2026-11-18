# Nano-vLLM 四项优化设计与评测方案

日期：2026-11-18。状态：设计稿；本文中的接口、命令、脚本及性能目标均为拟实现方案，并非已实现或已测结果。

## 1. 目标与现状

目标是得到可验证的多 GPU 服务、较新架构适配、可控精度损失的缓存压缩，以及有证据支持的实际性能优化。先建立可重复的测量，再逐项修改；保留原路径作为回退开关。没有真实 GPU 测量时，只报告功能验证，不填提速百分比。

检查基准：本地通过普通 fast-forward 同步到 `727854c`；具体实验还必须记录完整 commit SHA。当前存在 Qwen3 目标模型、paged KV、prefix caching、Chunked Prefill、混合批处理、TP、W8A16、异步 OpenAI API，以及新加入的独立草稿/DSpark 推测解码。ModelRunner 仍直接构造 Qwen3ForCausalLM；草稿模型支持不代表目标架构已通用化。Attention 将浮点 KV 直接交给 FlashAttention，尚无独立 P/D 实例交接或量化 KV 后端。原 `bench.py` 是固定随机长度的离线输出 tok/s 测试，无法解释在线排队、TTFT 或尾部延迟。

环境按 Linux + NVIDIA CUDA/NCCL/Triton/FlashAttention 规划；Windows CPU/参考 SDPA 测试仅用于逻辑验证。用户计划使用同机4×RTX 4090；实际PCIe拓扑、P2P和可用显存待检查。先用其中两卡验证TP=1交接，再做四卡公平对照。当前开发机不能视为已有该多卡平台。

## 2. 路线选择与实施顺序

可选路线：①先做 PD，再补架构和缓存，最快展示双卡但易返工状态协议；②先统一缓存状态接口，再量化与 PD，最后扩展复杂架构；③直接覆盖异构 TP、跨机和 MoE，范围过大且难以归因。推荐路线②：先冻结协议，Qwen3 打通，再接较新混合架构。

| 阶段 | 工作与依赖 | 交付、独立提交边界 |
|---|---|---|
| S0 | 评测、可观测性、正确性参考 | 固定工作负载、原始记录、基线报告；不改算法 |
| S1 | 模型注册与缓存状态接口 | 保持 Qwen3 等价，消除硬编码；定义混合状态协议 |
| S2 | KV 压缩 | Qwen3 FP8 优先；不可用时研究 INT8 独立内核；不静默切换 |
| S3 | 单机双卡 PD | 同精度无压缩先验收，再开启压缩交接；TP=1→同构 TP>1 |
| S4 | 较新架构落地 | Qwen3.5 dense 文本路径为首选候选，逐步验证 chunk/state/PD；不是“最新模型全部支持” |
| S5 | 实用优化与组合实验 | 根据 profile 选调度、传输重叠、采样、前缀复用、推测解码等 |

S1 为用户第2项的基础工作，S4 为完整架构支持；这样不让较大架构工程阻塞 PD。每个阶段可拆成若干功能完整的提交，各自测试、CHANGELOG、普通推送。本文仅交付设计，暂不编写推理功能或评测脚本；脚本的具体合同见第7节。

### 四卡部署前置门禁

S0归档 `nvidia-smi -L`、`nvidia-smi topo -m`、各卡对CUDA peer-access、NCCL双进程带宽/延迟与TP smoke结果。不能假定四卡均有高速P2P；Connector提供显式pinned-host中转后端，传输和暂存成本计入报告，不静默回退。探测Ada上实际安装的attention后端是否支持FP8 paged prefill/decode，决定S2的FP8或独立INT8内核路线。首版先用能单卡容纳且有缓存余量的小型Qwen3。正式实验固定卡号、CPU亲和性，记录功率限制、温度/时钟和其他进程，排除资源争用。

## 3. 模型适配与状态协议

“最新”随时间变化，不作为无限承诺。首版候选 Qwen3.5 dense 的文本生成部分，模型大小按显存选择并冻结 HF revision、config、tokenizer 与 chat template。官方 Transformers 文档说明该系列有 Gated DeltaNet 与 full attention 混合层；不能只换 QKV 权重名。MoE、视觉输入、MLA、跨架构草稿和任意 checkpoint 自动兼容不列入首版。

拟新增 `models/registry.py`，根据受支持的 `model_type` 显式选择模型；不支持时启动报错。`ModelCapabilities` 声明 TP、chunked prefill、prefix sharing、speculative、quantized cache、PD 能力；不支持的组合必须拒绝，不能默默退回另一种语义。

拟新增 `cache/` 层：`CacheSpec` 描述每层状态类型、形状、dtype、scale 粒度和版本；`CacheBackend` 提供分配、写入、读取/注意力、导出、导入、释放；`RequestState` 包含 computed length、token history、positions、采样状态及状态句柄。

full-attention 层保存分页 K/V；线性层保存递归矩阵、卷积状态等模型要求的全部状态。线性状态不能套用 KV 整页哈希共享；初版禁用混合模型 prefix hit，验证 snapshot/restore 后再开放。抢占初版以重算恢复，避免错误恢复半截状态。新架构推测解码初版禁用，直到状态回滚验证通过。

验收：小型随机配置与固定 checkpoint 的 logits/贪心 token 对齐 Transformers；分别比较整段 prefill、不同 chunk 切分、逐 token decode、抢占重算、取消释放。数值容差按 dtype 预先设定并记录，不用随机采样文本逐字相同替代分布正确性。

## 4. KV Cache 压缩

先做数值量化，不默认丢弃 token。已有 W8A16 是权重量化，不会自动压缩 KV。前缀复用减少重算，也不等价于降低每个 KV 元素位宽。

优先探测当前硬件及实际安装的 attention 内核是否支持端到端 FP8 paged prefill/decode；同时检查 head dim、布局、GQA、scale 和混合批次。支持则加入显式 `kv_cache_dtype`、K/V 独立 scale 与校准配置；否则保留 FP16/BF16，另开 INT8 Triton 后端里程碑。仅存 INT8、每轮把全部历史恢复成 FP16 的原型可用于正确性，但不能作为生产性能完成标志。FlashAttention 现有调用不能直接接收任意 INT8 KV。

采用不可变页 scale 或固定校准 scale，避免新增 token 改变旧页解释；如做动态页 scale，必须重编码整页并更新版本。scale、对齐、暂存空间全部计入显存和 PD 载荷。首版不量化线性递归状态。

传统同构 attention 的理论容量：`bytes = 2 * layers * tokens * kv_heads * head_dim * element_bytes`，TP 实际分片、复制及 padding 按每 rank 实测。例：32层、8 KV头、head_dim=128、8192 token，FP16 约1 GiB，8bit数据约0.5 GiB，另加 scale；不是整个服务显存减半。

压缩验收目标（待测，不是结果）：固定长度/并发下实际缓存占用下降≥40%；固定留出预算下可容纳 token 增加≥1.5倍。质量门禁暂定固定文本集 PPL 相对增加≤1%、固定长上下文检索集准确率下降≤1个百分点；同意冻结后再跑，不在看过结果后修改阈值。短上下文可能不提速，应如实报告反量化开销。

## 5. 多卡 PD

拟新增 `distributed/kv_connector.py`、`pd_coordinator.py`，修改 engine 的阶段边界、scheduler 的导入队列和 server 路由。P、D 为独立进程/引擎，各自具备完整模型执行能力；不是按层切模型。首版关闭推测解码、权重量化和缓存量化验证基础交接，再逐个开启组合。

```text
API/Coordinator → P Engine → 导出状态 → Connector → D Engine → SSE
                       ↘ P处理下一请求     ↘ D合批生成
```

首版 P 仅完成 prompt 前向并返回最后位置 logits，不采样、不向用户输出；D 导入所有状态后依据这些 logits 采样首 token，后续正常 decode。由此明确首 token、随机数和输出唯一所有权，避免重复采样/漏 token。engine 需提供 prefill-only 接口，不能直接复用目前会 append token 的完整 step。

状态机：`QUEUED_P → PREFILLING → WAIT_D_CAPACITY → TRANSFERRING → READY_D → DECODING → FINISHED`，各状态可进入 CANCELLED/FAILED。D 先预留容量，P 导出页暂时 pin；传输完成用 CUDA event/流同步确认，再原子发布 D block table；D ACK 后 P 释放。逻辑页号在 D 重映射，不传裸指针/源物理页号作为目标索引。

交接元数据至少包括 request id+attempt、模型和 tokenizer revision、cache spec/version、层布局、token/position/computed length、dtype/scales、TP布局、采样参数和状态校验。P/D 参数不兼容时拒绝。单请求 prompt 状态必须完整可用后才能 decode；不同请求可 P/D 并行。混合架构额外传递递归/卷积状态。

首版同机 NCCL point-to-point 按层/页打包，CPU控制面交换元数据；独立于 TP 通信组，所有 rank 使用确定的操作顺序，避免 collective 次序死锁。先同步正确性再做异步流和计算重叠。TP=1验收后支持相同 TP、相同分片；P TP≠D TP、RDMA跨机、动态伸缩不列入首版。

Coordinator 维护容量与有界队列；慢客户端、取消、传输超时均释放两端页。传输中取消不能提前复用仍在 DMA 的页。首版失败明确终止请求，不做隐式重放；用 attempt id 防止迟到 ACK 和重复发布。测试覆盖满池、部分传输失败、D退出、并发取消、跨请求隔离和长时间页泄漏。

性能门禁采用同GPU预算比较：PD 1P1D 对双副本各TP1、单实例TP2；选两种共置基线中满足质量/稳定性的较优者。冻结混合长短请求和SLO后，目标 P95 ITL下降≥20%，或SLO goodput增加≥10%；若不满足，保留实验能力而不宣称PD胜出。传输可能使TTFT恶化，需同时报告。

## 6. 其他优化：按 profile 选，不重复建设已有功能

1. 低风险配置扫描：token budget、chunk阈值、active seq、prefix hit比例；绘制吞吐—P95延迟曲线，不能只挑最高 tok/s。
2. 观察 CPU tokenization、采样排序、CPU/GPU同步、SSE解码成本；有瓶颈才做采样 fast path、减少同步或异步传输。
3. PD分层传输/双缓冲：基础正确后才做，记录真实重叠比例，不能把提交异步操作当成已隐藏通信。
4. prefix缓存已有；增加命中率/驱逐指标和冷暖实验，必要时优化策略。用户数据集按固定revision或内容哈希冻结，匿名记录不保存敏感prompt。
5. 推测解码已有；测接受率、每轮提交token、draft成本和额外显存。单独开关，不能把模型或草稿更换收益算进PD。
6. CUDA graph已有；只针对覆盖不足/重新捕获成本优化。融合、专用kernel、低位权重需profile证明收益，再独立提交。

## 7. 拟编写的评测脚本与输出合同

所有脚本有 `--help`，CLI参数覆盖manifest；保存最终解析配置、版本、随机种子及时间。下面文件和命令尚未创建，不应直接当作可运行接口。

| 文件（拟新增） | 职责 |
|---|---|
| `benchmarks/workloads.py` | 固定seed生成token输入与到达时间，保存JSONL/hash；真实文本另用固定tokenizer处理 |
| `benchmarks/offline.py` | 绕过HTTP测吞吐/模型阶段，正确CUDA计时；区分纯GPU与端到端 |
| `benchmarks/serve.py` | asyncio HTTP SSE开环压测，记录计划/实际发送、首输出/末输出、错误、usage |
| `benchmarks/cache.py` | 写入/读取/attention/交接微基准；已用页、池预留、scale与workspace分开 |
| `benchmarks/quality.py` | Transformers参考、logits误差、PPL、长上下文检索、状态回滚 |
| `benchmarks/compare.py` | 配对聚合、指标方向、95%置信区间、配置一致性检查、Markdown/JSON报告 |
| `benchmarks/configs/*.json` | 基线、单项和组合实验矩阵；不下载权重或修改系统 |

接口示例（未来实现）：
```bash
python -m benchmarks.workloads --preset mixed --seed 42 --output runs/workload.jsonl
python -m benchmarks.serve --manifest benchmarks/configs/baseline.json --workload runs/workload.jsonl --output runs/baseline
python -m benchmarks.serve --manifest benchmarks/configs/pd.json --workload runs/workload.jsonl --output runs/pd
python -m benchmarks.compare --baseline runs/baseline --candidate runs/pd --output runs/comparison.md
```

基准输出：`manifest.json`（完整配置/硬件/软件/model revision/git SHA）、`requests.jsonl`（每请求时间/计数/状态）、`metrics.json`（聚合）、可选阶段trace、`report.md`。失败、超时和客户端发送延迟单列；不能只统计成功请求而隐藏过载。

在线使用 monotonic/perf_counter 计时，TTFT从实际发送到首个非空生成内容，另报告从计划到达开始的排队包含延迟；忽略role/空SSE事件。开环固定Poisson到达，固定seed，对照共用同一trace；负载点为基线饱和附近的0.25/0.5/0.75/1/1.25倍，脚本必须报告发送滞后且限制inflight，达到上限记客户端拒绝，不能悄悄变闭环。闭环并发1/4/16/64另作补充。

SSE事件不是token：推测解码可能一次输出多个token，UTF-8字符也不等于token。HTTP端记录可见chunk间隔与真实usage token数；精确ITL来自engine内部每token就绪trace，称为内部ITL，不能与客户端chunk间隔混用。TPOT=`(末token时间-首token时间)/(输出token数-1)`，只有输出≥2才定义；多token同轮提交需标记批量就绪。PD的TTFT包含排队、P、传输、D采样和网络。

离线GPU阶段用CUDA event并明确同步；整服务wall-clock测量包含主机成本，不逐step强制同步破坏重叠。采样显存分别记PyTorch allocated/reserved和NVML进程显存，额外记录权重、KV池大小、实际占用页、draft、scale、workspace；压缩后自动扩大KV池时，不能据NVML不降就判定压缩无效。

## 8. 工作负载与公平对比

| 场景 | 输入/输出token | 目的 |
|---|---|---|
| 短对话 | 128/128 | 小batch开销 |
| 长输入 | 2048/128，显存允许再8192/128 | Prefill与交接 |
| 长输出 | 128/1024 | Decode带宽 |
| 混合 | 80% 128/128 + 20% 2048/128 | Prefill干扰与尾延迟 |
| 前缀复用 | 固定公共前缀，命中0/50/90% | 冷暖缓存隔离 |
| 压力/取消 | 接近容量，取消10% | 泄漏、拒绝、OOM和恢复 |

输入+输出必须在checkpoint上下文与配置允许范围内；超限报错或显式skip，禁止静默裁短。合成token只测性能，token从合法词表抽样，真实质量用文本。性能用ignore_eos固定输出长度和greedy/top_k=1；独立质量实验保留实际EOS。HTTP若不支持ignore_eos，使用实际输出数并标明不可严格配对，后续增加显式基准参数再测固定长度。

基线B0为当前commit：浮点KV、无PD、推测关闭、权重不量化，记录chunk/prefix/graph的全部开关。B1统一接口等价检查；C1只换KV；P0只开PD；P1 PD+KV；M0新模型官方参考与本实现同checkpoint比较；O1每次只增加一个优化。新模型不能与Qwen3直接比较后声称引擎提速。

四GPU预算必须分别跑：四个TP1副本、两个TP2副本、单实例TP4、1P3D（1个P实例和3个D实例，均TP1）、2P2D（2个P实例和2个D实例，均TP1）、P组TP2+D组TP2。多D按容量/排队路由，状态只传给选中的D；实例数量不同不代表异构TP。报告总吞吐和每GPU吞吐。单卡和双卡仅用于定位，单卡装不下的模型将TP1配置标为不可行，使用同等总卡数的可行布局。

各配置至少预热10个请求并确认编译/graph完成，每负载点至少200个有效请求且运行≥120秒，重复5次，随机交替基线/候选顺序；全矩阵成本过高先smoke再正式跑。长任务允许分批运行并保存中间结果。P99需至少1000个成功请求且披露样本数，小样本不做稳定P99结论。清楚分离冷启动、冷prefix和稳态，避免前一实验缓存污染。

指标包括：成功/失败/超时、req/s、实际output tok/s、input tok/s、TTFT/TPOT/E2E的P50/P95/P99、内部ITL/客户端chunk间隔、峰值显存、有效缓存bytes/token、传输bytes和耗时、队列等待、页命中/抢占、GPU利用率、推测接受率。SLO goodput=同时满足预先冻结TTFT和TPOT阈值的成功请求/观察时长；各场景阈值须跑前写入manifest，未配置则不报告goodput。

指标方向：吞吐提升=`candidate/baseline-1`；延迟/显存下降=`1-candidate/baseline`。对5次独立重复作配对bootstrap 95%区间，明确以run为单位，不能把相关token当独立样本；区间跨0不声称稳定提升。0值/缺失/不同模型或预算不可比时显示N/A并解释。

## 9. 验证、里程碑与报告模板

S0先使用人工可核对的SSE样例验证时序、空chunk、多token、错误与usage，再对真实单卡服务测基线；不使用mock吞吐替代GPU性能。CPU测试覆盖页scale、导入元数据、重复ACK/取消、请求trace聚合；GPU测试覆盖真实paged attention、量化内核、两进程传输与长时间稳定性。保留全部现有推测/采样/调度回归测试。

每阶段报告固定格式：硬件/互连、commit/model/dataset revision、软件与配置、原始结果位置、正确性与质量、基线/候选指标、置信区间、失败与限制。建议结果表如下，当前全部待测：

| 配置 | GPU数 | output tok/s | P95 TTFT | P95内部ITL | 峰值显存 | goodput | 质量 |
|---|---|---|---|---|---|---|---|
| B0共置 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 | 参考 |
| 双副本TP1 | 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 共置TP2 | 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| PD 1P1D | 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| PD+压缩（验证） | 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 四副本TP1 / 双副本TP2 / TP4 | 4 | 分别待测 | 分别待测 | 分别待测 | 分别待测 | 分别待测 | 待测 |
| 1P3D / 2P2D / P-TP2+D-TP2 | 4 | 分别待测 | 分别待测 | 分别待测 | 分别待测 | 分别待测 | 待测 |

设计假设由实施者先采用：同机4×RTX4090、先双卡正确性再四卡评测、Qwen3作为系统基线、Qwen3.5 dense文本作为首个较新架构候选、FP8能力探测优先、无跨机/异构TP。互连拓扑未确定不阻塞报告；实际脚本实施前根据设备固定可行的模型/后端/矩阵，未具备环境的项目保留为未验证。

## 10. 官方参考

以下为2026-11-18查询的动态文档；实施时冻结实际依赖版本，不能把main文档能力直接当成本机已安装能力。

- [vLLM PD设计与Connector](https://docs.vllm.ai/en/latest/features/disagg_prefill/)：分离实例、缓存交接、TTFT/ITL独立优化；不保证吞吐收益。
- [vLLM量化KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)：量化格式、scale及后端约束的参考。
- [Transformers Qwen3.5](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5)：混合层与模型状态适配参考；不据此承诺所有变体兼容。

## 11. KV压缩路线调整：优先评估KVzap

用户提出使用KVzap。S2调整为优先开展KVzap剪枝的适配验证，FP8/INT8保留为独立后续组合项，不默认同时开启。技术评估见根目录explain.md的“KVzap与当前框架的适配”。这是方案评估，不表示已有可用后端。

新增S2a：Qwen3-8B+匹配官方评分器，在Transformers/kvpress冻结revision建立质量与策略参考，区分KVzapPress固定比例与DMS阈值路径；先整段prefill剪枝，暂不启用decode剪枝。S2b：实现原始位置/逻辑长度/保留长度分离、分层/head索引和物理compact，单请求对照后扩展批处理。S2c：实现chunked/mixed的稀疏历史因果mask、多请求生命周期、TP head映射及压缩PD状态传输。每一项独立验收。

近期窗口保护、阈值和实际保留率按官方参考固定；例示8K输入保留约4K或2K仅代表目标预算实验，不保证阈值法精确达到该长度。KVzap评分器需匹配目标checkpoint，0.6B或Qwen3.5无匹配权重时不得直接复用8B评分器。首版可使用官方linear评分器降低工程复杂度，但linear与MLP必须分别记录质量和成本。

现有FP8的40%占用下降等目标不直接沿用为KVzap既定收益。KVzap测完整保留、约2x、约4x的实际保留率/质量曲线，计入评分器、head索引、临时副本和碎片，使用真实回收页/有效bytes/token判断内存收益。先验证质量门禁，再比较评分/搬运额外成本、TTFT、decode延迟、吞吐和PD载荷。禁止用masked比例替代实际物理压缩率，禁止将论文2–4x结果写成本项目性能。

PD协议需额外传递每层/head原始位置、保留长度、有效页/索引、评分器版本及窗口状态；递归层暂不剪枝。实现复杂度高于同形状KV数值量化，S3依赖S2c完成；如果只是先做系统PD，可先用未剪枝KV验收再接KVzap。压缩路径初期禁用prefix复用、推测解码和graph，分别补足恢复/回滚/地址稳定性测试再开启。

## 12. 最新目标覆盖：MoE模型优先

用户明确最终目标为较新MoE，不使用dense Qwen3-8B作为最终模型。前述第3节dense候选和第11节Qwen3-8B评分器方案降为可选参考，不再约束最终实现；最终HF checkpoint/revision待用户指定。Qwen3.5-35B-A3B文本路径仅为候选。S1扩充模型能力和专家布局合同，先验证router、top-k专家、权重加载、MoE算子与TP/EP策略；S4完成目标MoE的attention/递归状态和推理对齐。EP跨卡通信会与PD/TP争用PCIe，首版优先可行的TP切分，不默认引入EP。

KV淘汰路线调整：S2首先对最终模型的attention类型分类。GQA路径优先研究PagedEviction式块回收，sink+recent为简单有损基线，SnapKV/AdaKV类作质量对照；MLA与混合递归层单独处理。KVzap作为有匹配评分器时的候选，不再默认优先。采用论文思路但聚合层/head评分以适配共享页表时，必须命名为变体，不能当作原论文复现。PagedEviction公开论文在Llama上验证，未证明目标MoE质量；参考vLLM内核免修改也不代表本框架FlashAttention路径免修改。

区分支持级别：vLLM模型支持表只证明架构进入支持范围，不能推出“该模型+某淘汰法+PD+量化+TP”组合已支持。KV-Compress有旧vLLM fork但缺TP等关键特性；PagedEviction主线feature请求已关闭not planned，不按即用主线功能规划。prefix缓存LRU与offload不是有损活动token淘汰。

四卡内存门禁：按全量参数而非激活参数估算权重。30B/35B模型BF16仅权重约60/70GB（十进制），一个四卡实例可能可行但仍需计缓存/workspace；P/D两个完整副本约120/140GB，超过四卡约96GB的标称总显存。故此前四TP1、双TP2、1P3D等布局不能直接套用未量化中型MoE。PD需先选可行的权重量化后端/checkpoint（含MoE支持验证），或更小MoE/更多GPU；现有dense W8A16不能直接视为支持MoE专家。4bit理想数据约15/17.5GB每副本，8bit约30/35GB，均另加scale、未量化权重、cache及workspace，不能仅据公式承诺装得下。可优先评估两组TP2的量化PD，与同权重精度的两TP2共置副本对照。四TP1和1P3D仅在单卡实测可装载时保留。

正式阶段顺序细化为：测量/拓扑→目标MoE完整缓存正确性与可行量化布局→淘汰质量及物理页回收→未淘汰PD→淘汰PD组合。淘汰降低未来attention读取不必然减少当前完整prefill峰值；若只在prefill后压缩，仍需能完成未压缩prefill，报告峰值与稳态分别记录。结果继续全部待测。

## 13. 多模态扩展可行性（候选范围，未实施）

用户询问多模态及PD/KV适用性。可复用调度、缓存管理与PD接口的基础设计，但增加processor、vision encoder/merger及逐层视觉特征注入、M-RoPE、图像content parts、视觉长度预算和encoder特征生命周期。完整分析见explain.md“多模态模型、PD与KV优化的关系”。

建议图像输入/文本输出先行，先实现单实例完整VLM，再将E与P共置、D独立，最后按profile决定是否拆E→P→D。四卡固定卡数比较时，将encoder副本、显存与传输全部计入。小型dense VLM仅为工程验证候选，不取代用户最终MoE目标。视觉token压缩、KV剪枝、KV量化和encoder cache复用分别开关、分别归因；视频/音频生成不包含于此可行性结论。所有性能与质量仍待实测。
