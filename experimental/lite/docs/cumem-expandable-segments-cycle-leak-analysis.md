# Colocated RL per-cycle VRAM net-growth — cumem × expandable_segments 根因分析 (Branch C)

Status: **零-GPU 分析出假设 → Exp-1 已跑,决定性结果已回填 (2026-07-13).** 本文是
Branch C 的交付:把"每周期净增长"从父任务 (TASK-1.13.8) 的 "thin-margin, not a leak"
结论重新审到 vLLM cumem 分配器 × PyTorch `expandable_segments` 的**已文档化不兼容**上,
给出可证伪假设 + proxy 复现/判别实验,并**已实测**。

> ## ★ Exp-1 决定性结果 (job 13905350, rc=0, 3:19, <1 GPU-h, 2026-07-13) ★
>
> **载具**:vLLM 0.23.1.dev0+g0fc695fc6 (cw `verl.vllm023.sqsh`, editable @/vllm),
> 小 proxy 模型 Qwen3-0.6B (真权重 dense qwen3),单卡 H100,`enable_sleep_mode=True`,
> **gmu 0.7(铁律未降)**,CUDA graphs ON,idle sleep(level 1)/wake ×20,零 verl/mfsdp。
> 原始 CSV:`examples/verl/cumem_cycle_probe/results/exp1-13905350-{A0,A1}.csv`。
>
> **结果:A0(expandable:True)与 A1(expandable:False)双臂,cycle-2 起逐位恒定,0 MiB/cycle。**
>
> | arm | expandable | awake (MiB) | asleep (MiB) | cycle2→20 drift |
> |-----|-----------|-------------|--------------|-----------------|
> | A0  | **True**  | 57641(恒)  | 2001(恒)    | **0 MiB** (bit-exact) |
> | A1  | **False** | 57599(恒)  | 1959(恒)    | **0 MiB** (bit-exact) |
>
> - cycle-1 awake 偏高(A0=59777/A1=59763)是首轮 torch.compile/CUDA-graph 暖机瞬态,
>   cycle-2 起即落到恒定值——**非累积**。
> - A0−A1 稳态差 = **42 MiB 一次性**(expandable 的 VA 记账小额 overhead),**不逐周期累积**。
>
> **判定(见 §5 判别树"全平"分支)**:在**隔离的 vLLM idle sleep/wake 路径**上,
> 文档化的 cumem×expandable 不兼容**不产生每周期泄漏**——H1/H2/H3/H4/F1 **在此范围内被否证**。
> 真实 32/128 卡 run 的每周期净增长**不来自 vLLM 自身的 sleep/wake cumem 循环**,
> 矛头须转向**训练侧进程(actor/mfsdp)的 expandable×cumem 交互**或 **update_weights 权重传输路径**
> ——与 bayan 07:12 guide③(vLLM worker 内部本就强制 allocator False,矛头对准训练侧)一致。
>
> **对 Branch A 方向冲突的影响**:在 vLLM 侧,`expandable_segments:True` **不是**每周期增长之因
> (双臂皆平,仅 42 MiB 一次性差)。故"expandable:True 是泄漏之因"这一疑虑**在 vLLM 侧被证伪**;
> 但**全局/训练侧 scope 仍未实测**(本探针仅测 vLLM 侧)。是否推翻 Branch A 已批 lever =
> 需一个训练侧 expandable×cumem 探针补测后由 human/moe 终裁(见 §6 下一帧)。

**下一帧(未做)**:训练侧(actor 进程持 Adam/optimizer state + expandable)× vLLM colocate 的
per-cycle 探针,或 update_weights 权重传输路径的显存差分——这才是真实 leak 的所在层。

参考源(fetch 于 2026-07-13):
- vLLM `device_allocator/cumem.py` @ `main` —
  https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/device_allocator/cumem.py
- vLLM Issue #47654 (sleep-mode leaks HBM;untracked allocations 逃逸 `pointer_to_data`)
- vLLM Issue #37860 (sleep mode not releasing GPU memory)
- vllm-ascend PR #9242 (Make expandable segments compatible with CuMemAllocator memory pool)
- PyTorch tracking issue: pytorch/pytorch#147851 (expandable segments × fixed-address memory pool)

> ✅ **新鲜度数据(已落账,Exp-1)**:cw `verl.vllm023.sqsh` 内 vLLM =
> `0.23.1.dev0+g0fc695fc6.d20260616`(editable @/vllm)。这是**新版 pool-context 动态
> toggle**族(非旧版硬 assert)——Exp-1 双臂无 crash 即佐证:expandable:True 下 sleep/wake
> 全程正常,进一步说明新版已在 cumem pool context 内动态处理 expandable,不再是 §2.2 的硬冲突。
> 这也解释了为何 vLLM idle 路径**双臂皆平**:该版本已消化了文档化的不兼容。

---

## 1. 问题重述与"leak vs thin-margin"的分歧

父任务 32 卡 SMOKE (job 13888949, expandable_segments:True + empty_cache ON + gmu 0.7)
结论:actor torch 侧峰值**基本持平**(每分钟 57.6→58.7→58.4→58.1 GiB),torch
`max_memory_reserved` 40→49 GiB 后 plateau,cycle 1–3 绿、cycle 4 在 vLLM `wake_up`
(`cumem_allocator.cpp:139`) OOM,差 ~1 GiB。父任务判为 **"thin margin, 不是 leak"**,
nvidia-smi 总量贴 79 GiB 悬崖,fragmentation/timing 把 cycle 4 顶翻。

bayan 06:12 **重新定性:这是显存泄漏(每周期净增长),根在 cumem × expandable_segments
交互**。分歧的本质是**采样分辨率**:4 个 cycle、per-minute 粗采样,无法区分
"~1 GiB/cycle 的慢速单调 creep(leak)" 与 "均值持平、fragmentation 方差涨到某个
倒霉 cycle 找不到连续空间(margin)"。**两者在 4 cycle 上都长得像"贴悬崖"。**
Branch C 的 proxy(20 cycle、细采样)就是为了把这两族分开。

本文因此把假设分成两族(§3),proxy 判别树(§5)按曲线形状定案。

### 1.5 已有数据的逐周期差分(AC#1,零-GPU,挖 job 13888949 / 13898028 已 distill 的 MEMLOG)

原始 per-cycle 设备级 `nvidia-smi memory.csv` / procmem 探针**在 cw lustre**,未同步回本
worktree(本地遍历确认无);可挖的是父任务 log 已 distill 的离散快照。它们**不足以给出
干净的 GiB/cycle 设备级斜率**,但足以做**关键归因判别**:

**(a) actor torch 侧 = 有界(plateau),泄漏不在这。**(定量)
- job 13898028(0.7 收口炮):actor `alloc` 37.66→45.73→45.73、`reserved` 39.78→49.41→49.42
  GiB(step1→3)。第 2 步 Adam 态首次物化 +~8 GiB 后 **plateau**,step3 与 step2 逐位持平。
- job 13888949(SMOKE):actor per-minute 峰 57.6→58.7→58.4→58.1、torch reserved 40→49
  后 plateau。
- ⇒ **两炮 actor-进程 torch caching allocator 均 plateau**;每周期净增长**不在 actor torch
  视角**(父任务 DoubleBuffer 作用域化 133413497 把这条锁死,单测计数归零、正确、必要)。

**(b) 设备级(nvidia-smi)> torch 视角,且死点周期号对峰值不变 = 累积成分在 torch 视角以下。**
- 死点两炮同为 **cycle-4** 同址 `cumem_allocator.cpp:139`;gmu 0.6→0.7 把设备峰从 ~58 抬到
  78.99 GiB(@05:28:36 gpu1)**只抬峰、不改死点周期号**(bayan 05:41 坐实)。
- 判读:若纯 headroom/thin-margin,改变可用余量应移动死点 cycle;死点固定在 4 ⇒ 存在一个
  **每周期推进、在固定 cycle 数触顶**的累积量,且它**活在 torch caching allocator 视角之下**
  (actor torch 已 plateR 却仍在 cycle-4 顶穿 80)——正是 cumem 物理块 / expandable 段 /
  untracked alloc 这一层(§2、§3),**不是** actor 张量泄漏。

**(c) 分辨率边界(诚实声明)。** 4 个 cycle、per-step/per-minute 粗采样,**无法**把
"~1 GiB/cycle 单调 leak" 与 "均值持平+碎片方差涨" 分开——这正是 §4 proxy(**20 cycle、
awake/asleep/woke 三相细采样**)存在的理由,也是 AC#1 "量化设备级 GiB/cycle" 的**干净数只能
由 proxy 或 cw-side 原始 CSV 给**的数据边界(类比父任务 repo-only recon 边界,不捏造占位数)。

**AC#1 结论**:归因已判别到 **torch 视角以下的 cumem×expandable 层**(排除 actor torch
泄漏);精确 GiB/cycle 由 §4 proxy A0 曲线补齐。

---

## 2. 机制грунт:vLLM cumem sleep/wake 与 expandable_segments 的已文档化冲突

### 2.1 cumem 分配器怎么工作(sleep/wake 的物理内存生命周期)

`CuMemAllocator` 用 CUDA VMM API(`cuMemCreate`/`cuMemMap`/`cuMemUnmap`/`cuMemRelease`,
经 C 扩展 `create_and_map` / `unmap_and_release`)管理**可卸载**显存:

- 只有经 `use_memory_pool_with_allocator(...)` context 里、由 `_python_malloc_callback`
  登记进 **`pointer_to_data`** 的分配,才被 cumem 追踪。
- `sleep(offload_tags)`:对每个被追踪分配,匹配 tag 的**拷到 pinned CPU** 后
  `unmap_and_release(handle)` **释放物理块**;不匹配的直接丢;标 `is_asleep=True`。
  **虚拟地址与 handle 元数据保留**(wake 时按原 handle 重建映射)。
- `wake_up(tags)`:`create_and_map(handle)` 重建虚拟映射,再把 CPU backup 拷回。

**关键**:`sleep()` 末尾的 `torch.cuda.empty_cache()` **只能回收进了追踪池的东西**。
任何在 pool context **之外**分配的 GPU tensor(仍是活 Python 对象)`empty_cache` 动不了
—— 这是 Issue #47654 的泄漏根因(mm encoder 在 pool 外分配 → 每 sleep 不释放 → 每周期
HBM 净留存)。**这条对我们同样成立**:凡不走 vLLM pool 的分配(见 §3 H2)都逃逸 sleep。

### 2.2 expandable_segments 与 cumem pool 的**根本不兼容**(smoking gun)

vLLM `cumem.py` 里对此有**明文**:cumem 的 sleep/wake pool **依赖固定地址分配**
(fixed-address:handle→固定 VA 重映射);而 PyTorch `expandable_segments` **动态扩张/
搬移**内存区。两者同时开会冲突。历史演进:

- **旧版 = 硬 assert**:
  ```python
  assert 'expandable_segments:True' not in conf, (
      'Expandable segments are not compatible with memory pool. '
      'Please track https://github.com/pytorch/pytorch/issues/147851 ...')
  ```
  即:旧版 vLLM 下,若进程设了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  开 sleep mode 直接 **AssertionError crash**(不是慢泄漏)。
- **新版 = 动态 toggle**(vllm-ascend PR #9242 起,已进 main):进入
  `use_memory_pool_with_allocator` 时**临时** `_set_allocator_settings("expandable_segments:False")`,
  退出时恢复 `True`:
  ```python
  conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
  expandable_was_enabled = "expandable_segments:True" in conf
  if expandable_was_enabled:
      torch.cuda.memory._set_allocator_settings("expandable_segments:False")
  # ... yield mem_pool ...
  if expandable_was_enabled:
      torch.cuda.memory._set_allocator_settings("expandable_segments:True")
  ```

**对我们的直接含义(核心洞见):**
父任务 Branch A 把 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 作为
**"primary lever / the proper fix"** 全局施加在 cw harness 上。但按 vLLM 自证:

1. 若 cw 容器 vLLM **旧版(硬 assert)** → 要么 SMOKE 早就 assert-crash(说明该版无 assert
   或被绕过),要么该分支根本没进 sleep 追踪路径 —— **需版本核实**。
2. 若 **新版(toggle)** → expandable_segments 只在 vLLM pool context 内被关,**在
   训练/actor forward/mfsdp export materialize(69 GiB 峰)期间是开的**。于是 torch 侧
   caching allocator 用 expandable 段扩到高水位后,这些段**只在 `empty_cache` 时才交回
   driver**;而 cumem `wake_up` 的 `cuMemCreate` 需要**新的物理页**。torch 的 expandable
   段与 cumem 的 fixed-address 映射**在同一设备 VA 空间里交错**,每 cycle wake 需要连续
   物理页,torch 段却按高水位驻留不缩 → **要么净增长(段不缩,H1),要么 VA 碎片单调恶化
   (fragmentation family)**。

**结论(待 proxy 证伪):expandable_segments:True 很可能是"每周期净增长"的因,而非父任务
认定的"碎片修复解"。bayan 06:12 的重新定性与本机制一致。** 这不否定父任务 Branch B
(scoped release_cached_buffers 丢引用)—— 那修的是 mfsdp export 钉住 slot 的正交 bug;
本文指向的是**再往下一层**的 driver/cumem 层交互。

---

## 3. 可证伪假设(排序;proxy 判别)

### Leak 族(每周期均值单调↑)
- **H1 — expandable 段不缩、cumem 无法回收(torch 侧驻留)。** expandable_segments:True
  下,export materialize(69 GiB)与 activations 把 torch expandable 段推到高水位;段驻留
  不交回 driver,cumem wake 拿不到 → 每 cycle 净留存。**判据**:关 expandable_segments
  后 net-growth 消失。**最可能。**
- **H2 — untracked allocation 逃逸 cumem sleep(Issue #47654 同构)。** 凡不经 vLLM pool
  的分配(`update_weights` 推权重的中转 buffer、CUDA graph pool、任何 side tensor)不进
  `pointer_to_data`,sleep 不 unmap → 每 cycle "睡着"占用 creep。**判据**:每 cycle
  **sleep 后立即**测 vLLM 进程驻留,若 creep → H2;结合 CUDA graph 开/关对照。
- **H3 — cumem handle/物理块未尽释(cumem 归还)。** 追踪到但 handle list 空 → 跳过
  unmap_and_release(ROCm double-free 守卫的 CUDA 侧同构);或 graph-capture 进 pool 每 wake
  增长。**判据**:cumem `pointer_to_data` 条目数 / 累计 `cuMemCreate−cuMemRelease` 差随
  cycle 涨。
- **H4 — mfsdp export 残留(export 残留)。** 父任务 scoped release 丢了引用,但若
  expandable 段仍钉住那 69 GiB 物理背衬(empty_cache 在 expandable 下不真交回),export 段
  每 cycle 累积。**实为 H1 的一个面**;判据同 H1(关 expandable 后是否消失)。

### Fragmentation 族(均值持平,carved 方差/最大空闲块单调恶化)
- **F1 — cumem fixed-address 映射与 torch expandable 扩张在同一 VA 空间交错**,
  `reserved − allocated` 缺口 & 最大连续空闲块随 cycle 恶化,均值持平但某 cycle 找不到
  连续空间 → wake OOM(父任务"thin margin"的机制化版本)。**判据**:net 均值≈0 但
  `reserved−allocated` 或 largest-free-block 单调恶化。

**注**:父任务已**证伪** DS4 21:05 "7 兄弟 rank ×2.5 GiB 可见性泄漏"(每卡仅 1 actor+1
vLLM,`device_count()==1`)—— 该杠杆不适用,proxy 不必复验此项。

---

## 4. proxy 复现 & 判别 harness 设计

**目标**:几分钟一轮,20 cycle,拿到**干净的 per-cycle 显存曲线**,判 leak vs fragmentation
并定位到 H1–H4/F1,再决定修法。**泄漏机制与模型无关(bayan)** → 用最小模型。

### 载具
- 模型:Qwen3-0.6B(或 1.7B/4B 备选),复用 `examples/verl` DAPO/GRPO 最小配方
  (README §"Run GRPO with the MLite actor and vLLM rollout")。
- 拓扑:**1 节点**,1~2 GPU,colocated actor(mlite,backend 任意——泄漏在 vLLM/driver 层,
  mfsdp 非必需,可先用最省事 backend 缩小变量)+ vLLM rollout,`enable_sleep_mode=True`。
- **gmu 固定 0.7**(bayan 守 0.7:不靠缩 KV 掩盖泄漏,要看真实 net-growth)。
- 环境走 cw 容器(GPU 铁律:Slurm,禁登录节点跑),复用 kernel_rollout `docs/runs/` 的
  sbatch/容器写法(pytorch sqsh + job 内 venv),**先查先例再新搭**。

### 循环与埋点(每 cycle 记一行 CSV)
```
for cycle in 1..20:
    generate (rollout)            # vLLM awake
    llm.sleep(level=1)            # 卸权重、丢 KV
    [measure @ ASLEEP]            # ← 关键:睡着态驻留是否 creep (H2/H3)
    train_step + update_weights   # actor 前反向 + 推权重回 vLLM
    llm.wake_up()                 # cumem 重映射（OOM 现场）
    [measure @ AWAKE]             # 峰值/驻留
每次 measure 记录:
  - nvidia-smi total device MiB（进程级 --query-compute-apps）
  - torch.cuda.memory_reserved / memory_allocated / max_memory_reserved
  - torch.cuda.memory_stats(): reserved_bytes.all.current, num segments,
    inactive_split / (reserved-allocated) 缺口, largest free block(碎片指标)
  - vLLM cumem: len(allocator.pointer_to_data)、累计 create/release 计数(能拿则拿)
```

### A/B 判别臂(每臂 20 cycle,几分钟级)
| 臂 | 变量 | 判什么 |
|---|---|---|
| **A0 baseline** | expandable_segments:**True**, gmu 0.7 | 复现 net-growth(对齐父任务 32 卡) |
| **A1 关 expandable** | expandable_segments:**False** | net-growth 是否消失 ⇒ 判 H1/H4/F1(**最关键臂**) |
| **A2 sleep level** | level 1 vs 2 | 卸载 vs 丢弃对 creep 的影响 |
| **A3 无 update_weights** | 只 sleep/wake,不推权重 | creep 是否跟权重推送(H2 update 路径) |
| **A4 CUDA graph 开/关** | enforce_eager on/off | graph pool 是否随 wake 增长(H2/H3) |

先跑 A0 确认能复现;A0 复现后 A1 是**决定性实验**。若 A0 在 0.6B 上复现不出(可能 8 GiB
模型太小、峰值撑不到悬崖),放大到 4B 或人为抬 gmu/seqlen 制造压力,再固定回 0.7 观察斜率。

---

## 5. 判别树(曲线形状 → 机制 → 修法方向)

```
A0 net-growth 斜率 > 噪声带?
├─ 全平(awake 与 asleep 皆逐位恒定,无碎片恶化)  ⟵ ★ Exp-1 实测落此叶 ★
│     ⇒ vLLM idle sleep/wake 路径**无泄漏、无碎片**(此版本已消化 cumem×expandable 冲突)。
│        判定:每周期净增长**不在 vLLM 自身**;转查【训练侧 actor/mfsdp 进程的 expandable×cumem】
│        与【update_weights 权重传输路径】。expandable:True 在 vLLM 侧非泄漏之因(仅 42MiB 一次性)。
├─ 否但碎片恶化(均值持平,largest-free-block/reserved-allocated 单调恶化)
│     ⇒ Fragmentation 族 (F1)。修法:碎片治理——但 expandable_segments 本身与 cumem
│        fixed-address 冲突,方向是【收口 run 关 expandable_segments,靠 cumem toggle +
│        empty_cache 时序】而非全局开 expandable。父任务"thin margin"成立但可控。
└─ 是(单调净增长)⇒ Leak 族,按 A1 分叉:
   ├─ A1(关 expandable)后斜率≈0  ⇒ H1/H4：expandable 段不缩/cumem 无法回收。
   │     修法:【收口 run 不全局设 expandable_segments:True】(与 Branch A 相反!),
   │        或仅在无 sleep 的纯训练段开、colocation 段关;export 后 empty_cache 归还 driver。
   │        ⚠ 这会**推翻父任务 Branch A "primary lever" 的方向**,需 moe 门 + bayan 裁。
   └─ A1 后仍单调↑（torch reserved 平但 nvidia-smi 涨,或 sleep 后驻留 creep）
         ⇒ H2/H3：untracked allocation / cumem handle 未尽释。
         修法:确保 vLLM 全部分配走 pool(update_weights 中转 buffer、CUDA graph);
            必要时上游 issue（#47654 家族);A4 若 graph 是元凶则 enforce_eager 或修 graph pool。
```

---

## 6. 交付物 / 本帧边界 / 下一帧闸

**本帧交付(零-GPU,全部 commit 进交付仓):**
- 本分析文档(§1–5):机制грунт + AC#1 已有数据差分 + 可证伪假设 + proxy 判别设计/树。
- **便宜实验载具已实现**(AC#3 的下一帧变成纯执行,不用现搭):
  `examples/verl/cumem_cycle_probe/` —
  - `cumem_cycle_probe.py`:vLLM-only sleep/wake ×N,per-cycle awake/asleep/woke 三相采
    nvidia-smi 设备驻留 + torch memory_stats(reserved/allocated/碎片/segments)。stdlib+vLLM,
    `--help` 无需 CUDA(py_compile/help 已过)。
  - `run_cumem_cycle_probe.sbatch`:复用 kernel_rollout 已验证 vllm023 容器配方,**单卡**
    背靠背跑 **A0(expandable:True)/A1(:False)** 决定性对照,gmu 固定 0.7,预算 <1 GPU-h。
  - `README.md`:判别读法(A0 爬升+A1 平 ⇒ H1;均值平+碎片涨 ⇒ F1;asleep 爬 ⇒ H2)。
- **层级化便宜实验计划**(便宜先行,按需升级):Exp-1 idle sleep/wake A0/A1(最便宜、最决定
  性)→ 若 A0 也平则 Exp-2 `--reload-weights`(近似 update_weights,仍 vLLM-only)→ 若仍平
  则 Exp-3 全 colocated actor+vLLM proxy(需 verl/mfsdp,较重)。

**Exp-1 已执行(2026-07-13,job 13905350,rc=0,bayan 07:12 解冻批准):**
- 两道前置闸均已穿越:①预算解冻(bayan 07:12 批 <1 GPU-h);②新鲜度落账(vLLM 0.23.1.dev0)。
- 结果见文首 ★ 决定性结果 ★:**双臂全平,0 MiB/cycle**,落 §5 判别树"全平"叶。
- 原始曲线 commit 进交付仓:`examples/verl/cumem_cycle_probe/results/exp1-13905350-{A0,A1}.csv`。

**方向冲突裁决(仍需 human/moe 门,数据先行已就绪):**
- Exp-1 证伪了"vLLM 侧 expandable:True 是每周期泄漏之因"(双臂皆平)。但**全局/训练侧 scope 未测**——
  Branch A 的 lever 是**全局**施加,训练侧 actor 进程(持 Adam optimizer state,~35B×fp32 首步物化)
  的 expandable×cumem 交互仍是未测变量。**是否推翻 Branch A 已批 lever = 需训练侧探针补测后 bayan 终裁。**

**下一帧(未做,新 frame)= 训练侧 per-cycle 探针:**
- 目标层:actor/mfsdp 进程 × vLLM colocate 的 per-cycle 设备驻留差分,或 update_weights 权重
  传输路径的显存差分。这才是真实 32/128 卡 leak 的所在层(Exp-1 已排除 vLLM 自身)。
- 需 verl/mfsdp(较重,非纯 vLLM),建议作为独立子任务,由 bayan 定预算与是否走小 proxy actor。

**收口**:训练侧探针定位 → 修法方向(守 gmu 0.7 铁律、零 backend 分支)→ 同载具验证 → 回 32/128 收口。
