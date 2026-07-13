# M-FSDP DAPO E2E resync OOM — port plan (TASK-1.13.8)

Status: **Root-cause fix landed (bayan 2026-07-13 03:30) — the persistent
`DoubleBufferAllocator` retention of the export buffer is the real bug; the export
buffer is now scoped to `full_parameter_context` and returned to the driver on exit
(see §"Scoped export-buffer lifetime fix (bayan 03:30)"). CPU/unit green (41 passed).
Awaiting pre-GPU quick gate → gmu 0.7 收口炮. The 0.6 data 炮 curves are kept as a
comparison baseline. Prior Branch B (verl-layer empty_cache) + round-2 gate history
retained below for provenance.**

bayan 00:30 解禁: the "wait for DS4" gate is void — our OOM is mfsdp's own
(actor won't yield to vLLM wake), fixable entirely in our own code. Branch B is
implemented (see §"Implemented (2026-07-13)"): threshold-batched `empty_cache`
draining the export so the released all-gather buffer is returned to the driver
**before vLLM wakes**, plus the `expandable_segments` launch env. The deeper
per-bucket streaming refactor (the MoE use-after-free hazard below) stays
DEFERRED — the observed OOM is at wake_up *after* export drains, which the
empty_cache-before-wake fix targets directly; the per-bucket peak reduction only
matters if the 32-card residency probe shows the transient export peak (not the
handoff) is the OOM.

Historical (pre-00:30) gating context retained below for provenance.

---

# Scoped export-buffer lifetime fix (bayan 03:30)

bayan 2026-07-13 03:30 拍板:**`DoubleBufferAllocator` 持久持有导出缓冲 = 真 bug,修**。
不是可选优化(旧 Branch-B kill-switch 框架作废),是根因修复:导出缓冲的生命周期必须
**作用域化到 export**,`full_parameter_context` 退出时把缓冲归还/释放给 driver,**不得跨
wake 存留**。零 backend 分支。

**What was wrong.** `fsdp_double_buffer`(默认)把 all-gather 缓冲钉在
`DoubleBufferAllocator` 的两个持久 slot 里,好在训练一步内多次 acquire/release 复用。
但一次 *full-parameter export* 把整个 34.6B 稠密模型物化进这些 slot 后,`release_all()`
只把 lease 标记为空闲——slot 字典**仍持活引用**。torch 的 caching allocator(以及
`empty_cache`/`expandable_segments`)永远无法在还有活引用时把那段存储交回 driver,于是
它跨过下一个 colocated consumer(醒来的 vLLM engine cumem allocator)的回合并饿死它。
这就是 resync wake_up OOM 的根因缓冲。

**The fix (form (b), inside mfsdp's own release protocol — 对齐 bayan 01:09 红线).**
`MegatronFSDP.full_parameter_context` 的 `finally` 里,在 `release_all()` +
`discard_full_parameter_views()` 之后**无条件**调 `param_sync.release_cached_buffers()`
(`CommunicationPipelines.release_cached_buffers` 遍历 buckets、按 `id(allocator)` 去重、
调既有的 `DoubleBufferAllocator.release_cached()` 清空 `_slots`/`_busy`/`_reuse_events`)。
export 结束缓冲计数归零;slot 在下次 export 透明重建。与 FSDP2 的 `full_tensor` 路径
(不留持久缓冲)对齐——这才是"显存对齐 fsdp2"验收的正题。

**No kill-switch, no backend branch.** 删掉了旧的 `MLITE_EXPORT_RELEASE_CACHED` env
开关(它给"重开 bug"留了后门,与"不得跨 wake 存留"矛盾)。释放无条件。export 每
rollout 周期一次(`get_per_tensor_param` 唯一 caller),重建成本摊到整个 rollout+train
周期 ⇒ 可忽略,无吞吐理由保留 A/B 开关。

**Unit test (bayan 要求"导出后缓冲计数归零").**
`test_mfsdp_full_parameter_export_returns_cached_buffers_to_driver`:导出中
`_retained_buffer_count(allocator) == len(buckets)`;退出后 `== 0` 且 `_slots` 空、
无 busy slot。全量 `tests/unit/primitive/test_mfsdp.py` **41 passed**,wrapper import OK。

**drop-ref 是必要非充分——与 verl 层 empty_cache 互补,非冗余(待 pre-GPU 门判定).**
mfsdp 的 scoped release 只是**丢引用**,让存储变为可回收;把物理内存真正交回 driver
供 vLLM cumem 复用,仍需 `empty_cache` 或 `expandable_segments`。收口 run 两者都带:
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + verl 层的
`stream_export_with_empty_cache`(commit 21f14c2a5,mlite engine 通用导出路径,**无
`if backend==mfsdp` 分支**,form-(a),已过 round-2 门)。二者关系:mfsdp drop-ref 是
根因修(此前 empty_cache 对被钉住的 slot 是 no-op),expandable_segments/empty_cache 是
return-to-driver 机制。**开放问题**:drop-ref + expandable_segments 是否已足够、verl 层
empty_cache 是否变冗余,需 pre-GPU 门/GPU 实验判定;本次不擅自 revert 已批准代码。

---

# 32-card SMOKE result — job 13888949 (2026-07-13 02:14–02:37)

**Verdict: Branch B is a real PARTIAL win but did NOT close the收口 run. SMOKE
NOT fully passed. Budget exhausted (~12.2 GPU-h on one attempt). Fork escalated to
bayan.**

Config: `q35_mfsdp_dapo_fd760e969_smoke`, 4-node/32-card, mlite HEAD `fd760e969`
(Branch B `resync_export.py`), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(A) + `MLITE_RESYNC_EXPORT_EMPTY_CACHE_GIB=4` (B, ON), TP1·PP1·CP4·EP8, vLLM
`gpu_memory_utilization=0.7` (deliberately unchanged to isolate Branch B),
GEN_TP4, `--time=00:25:00`, FR full-on. Ended FAILED (rc 1) at 02:37:39.

## What Branch B bought (measured, not asserted)

| | pre-fix baseline 13838501 (c17a05eff only) | Branch B 13888949 (fd760e969) |
|---|---|---|
| completed training steps | **0** | **3** (stable loss 0.9999→1.0001→1.0000) |
| resync (wake_up) cycles survived | 0 — OOM at **first** wake_up | **3** succeeded, OOM at ~4th |
| CP4 dense-gather deadlock | (fixed, held) | **held** — 3× through update_actor, FR dump **empty** (no hang) |
| failure site | vLLM `wake_up` cumem_allocator.cpp:139 | **same** site, but 3 cycles later |

So `c17a05eff` (deadlock) + Branch B (empty_cache-before-wake) together moved the
run from "dies on cycle 0" to "3 healthy resync cycles then OOM on cycle 4." The
deadlock fix and Branch B both work; they are **not sufficient alone**.

## Empirical per-card residency table (bayan's demanded itemization — now measured)

From the per-process residency probe (`.procmem.csv`, one GPU shown) at the failing
wake_up (02:37:20) — this is phase C of §① and it is **thin-margin, not a leak**:

| process | GPU MiB @ failing wake | note |
|---|---|---|
| actor `WorkerDict.actor_rollout` | **~58,100** (≈56.7 GiB) | export materialize peak during update_weights |
| vLLM `Worker_TP*` (asleep) | ~4,200 (≈4.1 GiB) | slept weights, not yet remapped |
| **used / free (nvidia-smi)** | **~62.6 GiB used / ~18 GiB free** | out of 80 GiB |
| vLLM wake needs (remap) | ~19 GiB (seen post-crash update_weights) | 18 free < 19 needed ⇒ **OOM by ≈1 GiB** |

**Actor export peak is BOUNDED, not creeping** — per-minute actor-process peak over
the four resyncs: 02:28→57.6, 02:31→58.7, 02:34→58.4, 02:37→58.1 GiB. torch
`max_memory_reserved` plateaued 40→49 GiB (allocated plateaued 37.6→45.7 by step 2).
Cycles 1–3 succeeded at the *same* ~58 GiB actor peak that cycle 4 OOMed at → the
system sits **right at the 80 GiB cliff**; fragmentation/transient timing tips
cycle 4 over. (`cpu_memory` grew 150→250→275 GiB — host-side offload accumulation,
not the GPU killer.)

**DS4 21:05 sibling-buffer hypothesis: REFUTED here.** Each GPU shows exactly ONE
actor + ONE vLLM worker; no 7×2.5 GiB stray sibling P2P buffers. `device_count()==1`
effectively holds. That lever does not apply to this run.

## Why this is a fork for bayan (not an autonomous retry)

1. **Budget exhausted.** Attempt-1 did NOT fast-fail (it ran 3 healthy steps before
   OOM at 22:57), so it spent **~12.2 GPU-h** of the ≤16 budget — leaving ~3.8
   GPU-h, too little for a second 32-card 25-min attempt. Protocol §③ assumed a
   hang would fast-fail at ~3.7 GPU-h; a *slow* healthy-then-OOM path broke that
   assumption. A second attempt needs a fresh budget grant.
2. **Failure criterion ② says do NOT blind-retry a wake_up OOM** — re-derive the
   table (done above) first. The table shows a **bounded thin margin**, which points
   at cheap levers, but choosing one changes the moe-approved protocol.
3. The candidate levers trade off differently and touch bayan's stated constraints
   (throughput gate, no-backend-branch red line, two-gate/two-round process):

   - **(A) Lower vLLM `gpu_memory_utilization` 0.7 → ~0.5–0.6** (harness config, no
     code). *Recommended.* The margin is ~1 GiB and the actor peak is bounded, so
     shrinking vLLM's KV reservation almost certainly buys the headroom. Cost: some
     rollout throughput (must then check the ≤20% resync/throughput gate). This is
     the textbook colocated-OOM fix and was explicitly held back to isolate Branch B.
   - **(B) Branch B-deeper: streaming per-bucket export** to cut the ~58 GiB
     materialize peak. Heavy — multi-file primitive change across runtime→primitive,
     carries the live MoE expert use-after-free hazard (see pre-check §), and needs
     its own pre-GPU moe gate. Only if (A) can't hold throughput.
   - **(C) Investigate fragmentation** (vLLM cumem vs torch expandable_segments
     interaction) — the same-peak-different-outcome across cycles suggests
     driver-level fragmentation; a targeted fix could be cheaper than (B).

Recommendation to bayan: grant a small budget for **one (A) attempt** (gmu 0.7→0.55)
since the margin is thin and bounded; fall back to (C)/(B) only if (A) trades too
much throughput. Do NOT declare the task done — SMOKE's wake_up-OOM criterion is
unmet.

---

# Pre-GPU 门 round-2 补全 (bayan 2026-07-13 01:12 reject 的四项)

Round-1 pre-GPU moe 门 reject 了三个 BLOCKER(全是协议补全)+ 一条 bayan 架构红线
(01:09, 另发)。四项在此补齐;补齐即过 round-2(两轮封顶)后由 operator/巡检点火。

## ① Per-rank 显存包络表 (BLOCKER 补全)

Gate rule (bayan 2026-07-12 20:54): **每一相位的 per-card residency 必须 itemize 并证
`< 80 GiB with headroom`**,否则不点火。失败模式是 wake_up OOM,所以按**相位**列表(而
非单一求和)才对症——OOM 发生在 export 之后 vLLM 醒来那一刻(phase C),不是稳态。

数据边界(2026-07-12 repo-only recon 已核,commit 122d4b333):**本表不能纯从 repo
faithfully 预算**。repo 默认 `engine/mlite.yaml` 出厂 `tp/pp/cp/ep=1`、offload OFF;收口
run 的真实并行切分 + vLLM rollout TP/quant 由 **cw harness 在 launch 时**给定(不在本
repo,见 Branch A);且 bayan 20:54 gate **要求 optim-state residency 经验实测**(不是读
flag)。故三行里有两行(rollout weight、经验 optim residency)需 cw-side + GPU 数据,点火前
在此不可得。**禁止捏造 placeholder 数字"过门"**——这些行在 cw ignition-prep 时按下方
recipe 实测填入,填表本身是点火前的一道 gate 步。

| Row (per card) | Phase A 稳态训练 | Phase B export 瞬时峰值 | Phase C vLLM wake | 数值 / 实测 recipe |
|---|---|---|---|---|
| actor sharded weight (mfsdp) | ✔ | ✔ | ✔ | 静态 ≈ **8.6 GiB** (sharded, 见 §materialization peak) |
| export full-param materialize | — | ✔ | — | 静态 ≈ **69 GiB** (34.6B BF16, `materialize_all`);export 退出时 mfsdp scoped `release_cached_buffers()` 丢引用 + expandable_segments/empty_cache 归还 driver(见 §"Scoped export-buffer lifetime fix") |
| optim state (Adam moments+master) | ✔ | ✔ | ✔(除非 offload) | **实测**:repo 默认 offload OFF;cw 端 `nvidia-smi --query-compute-apps` 逐进程确认是否 GPU-resident;要么 flip offload ON 并验证 off-GPU,要么把 resident 全量计入并证 phase-A/B 仍 <80 |
| grad buffer | ✔ | ✔ | ✔ | **实测**(同上;repo grad_offload OFF) |
| activations | ✔ | ✔ | ✔ | **实测** peak(microbatch 已定,cw 端读 `max_memory_allocated`) |
| NCCL / P2P buffers | ✔ | ✔ | ✔ | **先审 DS4 21:05 行**:确认每 actor `torch.cuda.device_count()==1`(仅自卡),否则 7 兄弟各 ~2.5 GiB 虚增(§"DS4 21:05 diagnostic");审通过后填实测 |
| vLLM rollout weight | — | — | ✔ | **实测**:`model_bytes × quant ÷ rollout_TP`(cw harness 定 TP/quant,如 FP8+TP16 ⇒ ≈19 GiB) |

**Green-light 判据(逐相位,全部满足才点火):**
- Phase A(稳态,vLLM asleep):sharded weight + optim + grad + activations + NCCL `< 80 GiB` with headroom。
- Phase B(export 瞬时峰,vLLM asleep):+ 69 GiB full-param materialize `< 80 GiB`。静态已知
  sharded 8.6 + full 69 ≈ **77.6 GiB**——headroom 极薄,optim/grad/activations/NCCL 任何一
  行非零都可能越 80 → **这是 Branch B(流式导出砍 materialize 峰)是否必需的判据**:若
  phase-B 表越 80,`expandable_segments`(碎片修复,非绝对占用修复)救不了,必须上 Branch B。
- Phase C(export 退出 + 归还 driver 之后,vLLM wake):actor 已回落到 ~phase-A + vLLM weight
  `< 80 GiB`。**这正是根因修的相位**——mfsdp scoped `release_cached_buffers()` 丢掉被钉住的
  持久 slot 引用(否则 empty_cache 对它是 no-op),再由 expandable_segments/empty_cache 把
  69 GiB 交回 driver,vLLM cumem 才能 wake。此行越 80 = 修复未生效,FR/py-spy 留栈定案。

表在 cw ignition-prep 组装(逐进程 residency 探针 dump,commit 0b00a4028 harness 移植进收口
run);三行实测数到位、逐相位 <80 才是唯一 green-light。

## ② SMOKE 判据 (BLOCKER 补全,写死)

SMOKE = 点火后的 go/no-go,判两件事关闭 + 早期曲线不发散;**非** full-length 曲线收口
(后者是任务 AC,见下"边界")。

- **完成定义(completion):** job 到 RUNNING 后,在 --time 硬顶内完成 **≥1 个完整
  actor→rollout resync 周期 + ≥5 个 training step**,且:①无 CP4 dense-gather deadlock
  (已由 c17a05eff 验证守住,回归确认);②无 vLLM `wake_up` CUDA OOM(21f14c2a5 修的相位);
  ③FR dump 为空(无 hang)。三者齐 = SMOKE 完成。
- **完整性 + parity 判据:** SMOKE window 内 loss / reward / grad-norm 曲线与**既有 DAPO
  baseline**(同 data/seed,FSDP2 或既有 optimizer backend)**同向不发散**——前 5 step 逐点相对
  偏差 ≤ 既有 backend-间 run-to-run 抖动带(非 bitwise;mfsdp≠FSDP2 数值路径)。吞吐(tokens/s)
  记录并与 baseline 比,SMOKE 阶段只报不卡(<20% 退化为 CAVEAT,严重退化再议)。resync wall
  time 记录并对齐 DS4 ≤20% overhead gate(memory 不能拿 speed 换)。
- **超时(timeout):** sbatch `--time` 硬顶(见 ③)+ 脚本内 per-step watchdog;RUNNING+5min
  必上机诊断(GPU 铁律),hang/OOM 即 scancel 留栈,禁等超时白烧。
- **失败准则(failure):** ①hang → FR/py-spy 栈 = 根因判决书,scancel,log,不盲 retry;
  ②wake_up OOM → 显存表 phase-C 估错,**不 retry**,回 §① 重推表(大概率 Branch B);
  ③曲线发散超抖动带 → 数值回归,block 交人,不当"跑通";④deadlock 复发 → c17a05eff 回归失效,
  block。任一失败都先问"哪道闸本该拦住",不是加卡重跑。
- **边界:** SMOKE 绿 ≠ 任务收口。full-length 曲线/吞吐对照(任务 AC)是 SMOKE 绿之后的
  follow-on 长跑,若 30-min 硬顶容不下完整 rollout(CP4 rollout 历史观测 ~26min,见
  `[[mfsdp-cp4-diagnostic-budget-conflict]]`),SMOKE 用**缩量 config**(cw harness 调
  `max_response_len` / `n_samples`)把一个 resync+train 周期压进硬顶,先关两个失败模式;完整
  曲线另请预算,由 bayan grant。这条写明以防"SMOKE 通=收口"的误判。

## ③ 16 GPU-h 上限可执行化 (BLOCKER 补全)

bayan 定收口 = 32 卡,预算 ≤16 GPU-h。落到可执行:

- **sbatch `--time` 硬顶:** 16 GPU-h ÷ 32 card = 0.5 h/card = **30 min wall 绝对上限**。点火
  attempt 用 `--time=00:25:00`(= 13.3 GPU-h,留 5min RUNNING-诊断-scancel 的白烧余量)。脚本
  内再加 per-step watchdog 自杀,不只靠 --time。
- **重试次数上限:** **≤2 attempts 总计**,round-2 后交 human(与两轮封顶同调)。且 code-level
  OOM/hang **不盲 retry**——必先按 §②失败准则根因(FR/py-spy),第二炮只给"transient infra 失败"
  或"表/config 已按根因修正"的情形;两炮累计 GPU-h 必 ≤16(attempt-1 若 RUNNING+5min 诊断即
  scancel 只花 ~3.7 GPU-h,给 attempt-2 留 25-min 满跑余量)。
- **分 rank / 分 job 记账(GPU 铁律台账):** 每 job 在任务 log 记
  `job id / 申请时刻 / RUNNING 时刻 / 首次诊断时刻 / 结论 / 花费 GPU-h`;逐进程 residency 探针
  (§①,commit 0b00a4028)按 rank dump 落 evidence。台账缺失 = review 直接 reject。

## ④ verl 集成层无 backend 条件分支 (bayan 2026-07-13 01:09 红线 — compliance 证据)

红线:optimizer backend 必须对集成层透明,diff 里出现 `if backend==mfsdp` = BLOCKER。合法只两
形态:(a) 通用卫生动作(所有 backend 一视同仁,FSDP2 同走且不劣化);(b) 收进 mfsdp release
协议内部。**本实现是 (a),证据如下:**

- `MegatronLiteEngine.get_per_tensor_param`(engine/mlite_engine.py:374)**无条件**用
  `stream_export_with_empty_cache` 包住 `self.runtime.export_weights` 的返回生成器——**没有任何
  optimizer-backend 分支**。该 engine 以 `backend="mlite"`(device/engine backend,非 optimizer
  backend)注册(mlite_engine.py:249),是 mlite 所有 optimizer 后端(mfsdp / dist_opt / fsdp2)
  的唯一 export 入口。
- `runtime.export_weights`(runtime/backends/mlite/runtime.py:348)用 `getattr(chunk,
  "full_parameter_context", None)` **duck-typing** 派发,也无 `if backend==...`。整条 export 路径
  backend-agnostic。
- `stream_export_with_empty_cache`(resync_export.py:48)只按**导出字节数 + allocator 状态**触发
  flush——这两个量对每个 backend 同构存在,wrapper 从不 inspect optimizer 类型。FSDP2 走同一
  wrapper:其 export 同样 materialize→release,drain 后 empty_cache 是**同一有益的 colocated-wake
  卫生**(把 freed 段交回 driver),**行为不劣化**;成本 = 每 ≥4 GiB 一次 device sync(相对 export
  all-gather 可忽略),且 env `MLITE_RESYNC_EXPORT_EMPTY_CACHE_GIB=0` 可整体关。
- primitive(megatron_lite export)**零改动**;`resync_export.py` / docstring 里的 "M-FSDP" 字样是
  **描述性**(解释为何 colocated wake 需要 flush),非运行时条件。RL-colocation 关注点放 verl 层、
  不入 vLLM-agnostic primitive,正是分层正确(bayan 反复点名 primitive 感知 vLLM = auto-reject)。

结论:diff 无 backend 条件分支,符合红线合法形态 (a);FSDP2 路径同走同一 wrapper 且不劣化,
满足"显存对齐 fsdp2 让位合同"的验收条件。

## Where we are

- CP4 dense-gather deadlock root cause is **solved** — fix `c17a05eff`
  (`0ed992990` buffer.py overlap-gate) verified decisively on real 32-card /
  4-node E2E (job 13838501): mfsdp param-gather recompute ran to completion at
  92–97% util, no deadlock, empty FR dump. Milestone accounted.
- The 32-card run then died at a **different** site: vLLM `wake_up` CUDA OOM
  (`cumem_allocator.cpp:139`) during actor→rollout weight resync
  (`ray_trainer.py:1672 update_weights`). This is colocated memory contention,
  same family as DS4 128-card resync OOM (`[[ds4-128card-resync-oom]]`), NOT the
  CP4 deadlock.

## Implemented (2026-07-13) — Branch B, verl layer

Fix landed in the **verl integration layer** (not the megatron_lite export
primitive) — the empty-cache-before-vLLM-wake concern is RL-colocation-specific,
so putting it in `runtime.export_weights` would be a layering violation. Files:

- `examples/verl/verl_mlite/resync_export.py` (new, stdlib-only so it is CPU
  unit-testable without a verl/CUDA runtime):
  - `stream_export_with_empty_cache(gen, threshold_bytes, empty_cache_fn)` —
    drains the export generator, firing `empty_cache_fn` once per
    `threshold_bytes` of cumulative exported material **and once more after the
    generator drains** (its `finally`, i.e. the runtime `ExitStack` →
    `release_all()`, has by then freed the M-FSDP all-gather buffer). That final
    flush returns the freed segments to the driver so the colocated vLLM cumem
    allocator can reclaim them on `wake_up` — closing the observed OOM.
  - `resync_export_empty_cache_threshold_bytes()` — ≥4 GiB default (DS4 recipe),
    env `MLITE_RESYNC_EXPORT_EMPTY_CACHE_GIB` (`0` disables).
- `examples/verl/verl_mlite/engine/mlite_engine.py::get_per_tensor_param` wraps
  the returned export generator with the above, `empty_cache_fn =
  aggressive_empty_cache(force_sync=True)`.
- `tests/unit/verl/test_mlite_engine_resync_export.py` — 10 CPU tests: export
  transparency (correctness), flush-per-threshold + final-drain flush counts,
  disabled pass-through, early-abort still flushes, env parsing. `10 passed`.

**Why empty_cache is the fix, not per-bucket streaming:** the 32-card run
completed the export (transient peak sharded≈8.6 + full≈69 ≈ 77.6 GiB fit) and
OOMed at vLLM `wake_up` *afterward*. `release_all()` returns memory only to
torch's caching allocator; vLLM's separate cumem allocator cannot see it, so
without an `empty_cache` between export-drain and wake the 69 GiB stays pinned.
`get_per_tensor_param` only ran `aggressive_empty_cache` once (guarded, at the
*start* of the first sync) — never after export. This wrapper fills that gap.

`expandable_segments`: a launch-env line
(`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) applied on the cw harness at
the收口 run (no repo patch — the harness lives cw-side, per Branch A). Defense in
depth for fragmentation.

## The mfsdp materialization peak (concrete site)

Export path for resync:

- `mlite_engine.get_per_tensor_param` (engine/mlite_engine.py:370)
  → `runtime.export_weights` (runtime/backends/mlite/runtime.py:348)
  → enters `chunk.full_parameter_context()` for **every** model chunk inside one
    `ExitStack`, up front, holding them all for the whole generator lifetime.
- `full_parameter_context` (primitive/optimizers/mfsdp/wrapper.py:191-199) calls
  `param_sync.materialize_all()` → all-gathers the full dense params
  (≈34.6B BF16 ≈ **69 GiB/rank**) and keeps them resident until `release_all()`
  in the generator's `finally`.

So during resync the actor holds ~69–77 GiB of materialized full params resident
while colocated vLLM tries to `wake_up` (peak >80 GiB) → OOM. `materialize_all`
was already flagged OOM-prone in `[[mfsdp-cp4-deadlock-rootcause]]`.

Note: mfsdp DAPO config currently ships offload OFF
(`config/engine/mlite.yaml`: param_offload/optimizer_offload/grad_offload=false),
so the actor is fully resident even before export materialization.

## DS4 dependency — what NOT to port

- DS4's implemented free-grad evict/restore (`31cfecbdf`) was **proven NULL** on
  DS4 config (job 13840020 FAILED, first resync still OOM by ~128 MiB; MEMCURVE
  showed grads not GPU-resident → free-grad lever empirically no-op). Do NOT port
  the free-grad protocol.

### DS4 recipe as SETTLED by bayan 2026-07-12 20:48 (supersedes earlier candidates)

Per-tensor `empty_cache` was **rejected** — each call carries a device sync +
allocator compaction, and thousands of params would blow up resync wall time.
The DS4 relaunch recipe bayan approved is:

1. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` = the PRIMARY lever**
   (it is the proper fix for fragmentation / materialization peak; pure env, not a
   config knob). Run this + instrumentation FIRST, with no per-tensor cleanup, as
   the baseline.
2. **Threshold-batched `empty_cache` only** — call once per **≥4 GiB of
   accumulated dropped references**; small tensors never trigger it. (Not
   per-tensor.)
3. **`resync` wall time is an acceptance gate**: instrument resync wall time;
   >20% overhead vs. the no-(2) baseline = **fail**. Memory AND speed both must
   hold — no robbing Peter to pay Paul.

Port only what DS4 validates green, and carry the same wall-time gate.

### DS4 20:54 correction (supersedes 20:48 on two points)

bayan scancelled DS4 job 13870979 (止损) and corrected the recipe again at
2026-07-12 20:54. Two deltas that reshape this plan:

1. **Per-tensor `empty_cache` is hard-OFF** — reconfirmed (thousands of tensors
   would 等死). `expandable_segments` + instrumentation stay. Our Branch B is
   already threshold-batched (≥4 GiB), not per-tensor, so this only *strengthens*
   the existing ordering — do NOT resurrect per-tensor cleanup.
2. **NEW pre-ignition memory-budget gate** — before burning ANY card, itemize the
   per-card residency and prove `< 80 GiB with headroom`. bayan's algorithm:
   - rollout side: vLLM per-card weight (e.g. FP8+TP16 ⇒ ≈300GB/16 ≈ 19 GiB);
   - training side: actor weight after parallel sharding, **+ optim state proven
     *empirically* CPU-offloaded (not by reading a flag — verify it is not GPU
     resident; last time an `opt_offloaded=False` field's semantics were unclear),
     + activations + NCCL buffers, each itemized;
   - sum `< 80 GiB` with headroom = the only green-light to ignite; log the table.

### DS4 21:05 diagnostic — sibling-rank NCCL/CUDA buffer residency (HYPOTHESIS, not yet confirmed)

While assembling the 20:54 budget table, DS4 (job r2) observed **7 sibling ranks
each holding ≈2.51 GiB resident (≈17.5 GiB/card of "stray" occupancy)**. bayan's
2026-07-12 21:05 targeted diagnosis (top priority on DS4): this is **almost
certainly an NCCL-buffer / CUDA-context problem, prime suspect a
`CUDA_VISIBLE_DEVICES` / Ray `num_gpus` misconfig** — if each actor process sees
all 8 local GPUs instead of only its own, NCCL/CUDA builds a context + P2P buffers
on *every* visible device, so exactly 7 siblings each pin a buffer. If confirmed
and fixed, that reclaims ~17.5 GiB/card and **flips the budget table outright**.

**Why this is in OUR plan (transfer, not copy):** our 32-card收口 run uses the
**same colocated verl/Ray actor↔vLLM architecture** (`ray_trainer.py:1672
update_weights`, vLLM `wake_up` — see §"Where we are"), so this visibility hazard
is a generic property of that colocation, not a DS4-only path. It therefore
belongs in OUR pre-ignition gate as a **watch item**: before believing any
per-card residency row, first verify each actor's `CUDA_VISIBLE_DEVICES` /
`torch.cuda.device_count()` is 1 (its own card), and check whether the colocated
vLLM↔actor CUDA-IPC weight sync *intentionally* needs cross-process visibility
(if shared-on-purpose, the lever is `NCCL_P2P` / cumem, not visibility). Do NOT
port a "fix" yet — it is an unconfirmed DS4 hypothesis; treat it as the first row
to audit when the budget table is assembled cw-side at ignition-prep time.

Status note (2026-07-12): DS4 (TASK-1.1.12) is **still In Progress, NOT green** —
20:48 and 20:54 were both "must-change-before-ignition" corrections, the last job
(13870979) was scancelled, and 21:05 opened a fresh sibling-residency diagnostic
that may再 reshape the budget analysis. This task stays gated until DS4 validates a
recipe green.

## Port plan — two branches

### Branch A — `expandable_segments` suffices (zero mfsdp code) — LEADING

This is now DS4's **primary** lever (20:48), so it is also the leading — and most
likely sole — mfsdp port. It attacks fragmentation / materialization peak, not
absolute residency.

- The "port" is a single launch-env line
  (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) in the cw DAPO mfsdp
  harness (`qwen35_dapo_mfsdp_*`), matching the `CUDA_DEVICE_MAX_CONNECTIONS="2"`
  env pattern. **These harness scripts live on the cw cluster, NOT in this repo**,
  so there is no repo-side patch to stage for Branch A — it is applied at launch
  time on the收口 run. No mfsdp primitive change.
- Cheapest, most likely-first-to-try. Verify on the 32-card收口 run itself, and
  record resync wall time (the DS4 ≤20% overhead gate applies here too).

### Branch B — threshold-batched `empty_cache` needed (mfsdp primitive change)

Only if Branch A is insufficient. NOTE the DS4 20:48 correction reshapes this:
the analogue is **NOT** per-tensor `empty_cache` (rejected as too slow) but
**threshold-batched** release — stop materializing **all** params at once, and
call `empty_cache` at most once per ≥4 GiB of released material, never per small
tensor. Any Branch-B change must clear the same resync wall-time gate (≤20%
overhead vs. Branch-A baseline).

- Restructure the export so chunks/buckets are materialized, yielded to the
  vLLM consumer, and released as we go — with a single `empty_cache` fired only
  after cumulative released bytes cross the ≥4 GiB threshold — instead of
  `materialize_all()` up front in `full_parameter_context`.
- Touch points: `param_sync.materialize_all` / a new streaming
  `materialize_iter` in mfsdp `param_sync`, consumed by
  `runtime.export_weights` and `full_parameter_context`.
- Correctness constraint to check first: `export_hf_weights` for qwen3_5
  (model/qwen3_5/lite/protocol.py:306) must not require all chunks materialized
  simultaneously (e.g. cross-chunk fusion for vLLM TP sharding). If it does,
  per-chunk release breaks it — gate any change behind an opt-in env
  (default-off, like DS4's `MLITE_RESYNC_*`) and **GPU-verify before enabling**.
- This is a correctness-sensitive primitive edit → follow MLite skills
  (primitive/perf), keep invariants, and it CANNOT ship without a real GPU run.

#### Branch B correctness pre-check — DONE (static, 2026-07-12, no GPU)

Traced the real export generator
`primitive/ckpt/hf_weights.py::export_hf_weights` (protocol.py:306 →
checkpoint.py:752 → this). DAPO mfsdp config is **PP1 + MoE (qwen3_5,
`num_experts`, ran EP8)**, so the PP≤1 branch (hf_weights.py:419-466) is the live
path. Two distinct retention behaviours:

- **Dense params** (attn/router/norms): iterated per chunk via
  `base_chunk.named_parameters()`, gathered one-at-a-time by `_gather_dense`
  (`allgather_concat` → a *new* `torch.cat` tensor, independent of the mfsdp
  buffer) and **yielded immediately**. No cross-chunk retention → per-chunk
  materialize/iterate/release/`empty_cache` is SAFE for dense weights.
- **Expert params** (MoE, active here): when `limit is None` they are
  **accumulated across ALL chunks** into `expert_groups` (hf_weights.py:421-440)
  and gathered only at the end (458-465). Each accumulated entry is
  `_materialize_dtensor(param.data.detach())`. For **mfsdp** params
  `_materialize_dtensor` (hf_weights.py:32-46) is a **pass-through** — it only
  `full_tensor()`s a *DTensor* (FSDP2); mfsdp's `materialize_all()` populates
  `param.data` in place, so the stored tensor is a **view into the mfsdp
  materialized buffer**, NOT an independent copy. `full_parameter_context`'s
  `finally` (wrapper.py:198) calls `release_all()` +
  `discard_full_parameter_views()`, which frees that buffer.

**Conclusion:** naive per-chunk release is UNSAFE for the expert path — releasing
chunk *i*'s mfsdp buffer before the final `expert_groups` gather would invalidate
the retained `param.data.detach()` views (use-after-free / wrong data). This IS
the "cross-chunk fusion" hazard the plan warned about, and it is real for MoE.

Viable Branch-B shapes (whichever DS4 evidence justifies):
1. **Clone-on-extract**: `.clone()` (or `_gather_expert` immediately) each expert
   tensor at accumulation so it survives its chunk's release. Adds a transient
   expert-sized copy — smaller than holding the full mfsdp all-gather buffer, but
   verify it actually lowers the peak vs. just holding one chunk.
2. **Incremental per-group expert gather**: gather+yield each expert group as soon
   as it is complete instead of deferring all groups to the end.
   Dense weights get per-chunk release for free in either shape.

Architectural note: current `runtime.export_weights`
(runtime/backends/mlite/runtime.py:354-361) opens
**every** chunk's `full_parameter_context` up front in one `ExitStack`, holding
all chunks resident for the whole generator. Streaming requires moving context
management **into** the export generator's per-chunk loop — a refactor crossing
the runtime → primitive → model boundary. Keep it behind a default-off env and
GPU-verify the peak actually drops before enabling.

This pre-check strengthens the "try Branch A (`expandable_segments`, zero code)
first" ordering: Branch B is a genuine multi-file primitive change with a live
MoE use-after-free pitfall, not a mechanical edit.

## Execution sequence (once DS4 green)

1. Wait for DS4 (TASK-1.1.12) to go green, then read its validated recipe +
   evidence (incl. measured resync wall-time overhead). Expect Branch A
   (`expandable_segments`) to be the answer since it is DS4's primary lever.
2. **Pre-ignition memory-budget gate (mandatory, per DS4 20:54)** — before
   launching the 32-card收口 run, produce and log the per-card residency table for
   OUR colocated site and prove `< 80 GiB with headroom`:
   - vLLM per-card rollout weight (at the收口 run's actual TP/quant);
   - actor weight after mfsdp sharding at TP1·PP1·CP4·EP8·DP2;
   - **⚠ offload tension**: mfsdp DAPO config currently ships param/optim/grad
     offload **OFF** (`config/engine/mlite.yaml`, see §"materialization peak"). The
     gate demands optim state be *empirically* off-GPU — so either flip offload ON
     for the收口 run and verify residency, or budget the full resident optim state
     and confirm the sum still clears 80 GiB. Do NOT trust the flag; measure.
   - export materialization peak (≈69–77 GiB if `materialize_all` stays) +
     activations + NCCL buffers, itemized. **Audit the NCCL-buffer row FIRST per
     the DS4 21:05 diagnostic** (§"DS4 21:05 diagnostic"): confirm each actor sees
     only its own GPU (`CUDA_VISIBLE_DEVICES` / `device_count()==1`) so stray
     sibling-rank P2P buffers (~2.5 GiB each) aren't silently inflating the table.
   Sum < 80 GiB with headroom is the only green-light. If the table itself shows
   Branch A (`expandable_segments`, a fragmentation fix, not a residency fix)
   cannot close an *absolute*-residency overflow, that is the signal Branch B
   (streaming export to cut the materialization peak) is required — decide from the
   table BEFORE burning the card, not after another OOM.
   - **Data-availability boundary (verified 2026-07-12, repo-only recon):** this
     table CANNOT be faithfully pre-computed from this worktree. The repo default
     `engine/mlite.yaml` ships `tp/pp/cp/ep=1` and offload OFF; the收口 run's real
     parallel sizing AND the vLLM rollout TP/quant are set by the **cw harness at
     launch time** (not in this repo, per Branch A), and bayan's gate requires the
     optim-state residency be measured *empirically on the target config*, not read
     from a flag. So two of the three rows (rollout weight, empirical optim
     residency) need cw-side + GPU data that is unavailable pre-ignition here.
     Assemble the table at ignition-prep time on the cw side — do NOT fabricate
     placeholder numbers to "pass" the gate. The only rows knowable now are the
     static ones already in this doc (export materialization peak ≈69–77 GiB;
     ≈34.6B BF16 dense actor weight before sharding).
3. Apply the minimal port:
   - Branch A: add the env line to the cw DAPO mfsdp harness at launch time
     (no repo patch). This is the default plan.
   - Branch B (only if the budget table or A's result proves it needed): opt-in,
     default-off, threshold-batched streaming export + GPU verify the peak drops
     AND wall-time overhead ≤20%. Do NOT pre-write this speculatively — its shape
     depends on DS4's threshold/wall-time evidence and it carries the live MoE
     use-after-free hazard documented above.
4. Re-run the 32-card DAPO E2E收口 (fix `c17a05eff` + FR + watchdog, budget ≤16
   GPU-h) for curves/throughput vs existing DAPO. Register via
   `vicky work execute remote-job` to avoid auto-submit/reap.
