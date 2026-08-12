# Appendix 08 — Untrusted Custom Code and GPU Kernels: Secure, Fair Execution
> Research appendix for the Prism v3 evaluation proposal (`docs/spikes/prism-v3/`). Produced 2026-08-06 via arXiv/web research. Non-normative spike document.

# Running and Fairly Evaluating Untrusted Custom GPU Kernels: A Survey and Design

*Survey date: 2026-08-06. All project dates and IDs verified against primary sources.*

---

## 1. Prior art — kernel competitions

### 1.1 GPU MODE / KernelBot / popcorn

GPU MODE (formerly "CUDA MODE", renamed in 2025; ~24K Discord members as of early 2026) is the canonical prior art. Its competition platform is **KernelBot** — paper: *"KernelBot: A Competition Platform for Writing Heterogeneous GPU Code"*, Zhang, Sirovatka, Schultheis, Horowitz, Saroufim, CODEML workshop @ ICML'25 ([OpenReview `bq9U4dmuyJ`](https://openreview.net/forum?id=bq9U4dmuyJ); repo [`gpu-mode/kernelbot`](https://github.com/gpu-mode/kernelbot), formerly `discord-cluster-manager`, first released 2025-02, launched March 2025). Scale: 25K submissions in the first two competitions; **~400K submissions by 2026**, across the AMD $100K comp, AMD $100K distributed comp, NVIDIA Blackwell NVFP4 comp, BioML, and the AMD $1.1M comp (2026-02). Problem sets live in [`gpu-mode/reference-kernels`](https://github.com/gpu-mode/reference-kernels); submissions go through [`gpu-mode/popcorn-cli`](https://github.com/gpu-mode/popcorn-cli).

Key architectural facts:

- **Submission unit**: a *single Python file* exposing `custom_kernel(data)`. Native CUDA is allowed via PyTorch `load_inline` (nvcc) or the newer `compile_kernel` API; Triton, Helion, ThunderKittens/CUTLASS (via `include_dirs`) are supported. Multi-file CUDA problems compile an `eval.cu` harness with nvcc and run the binary.
- **Problem definition**: `reference.py` (reference impl + `generate_input` + `check_implementation`), `task.py` (input/output schema), `task.yml` (shape spec used to generate test/benchmark cases of varying shapes).
- **Modes**: `test` (correctness), `benchmark` (no leaderboard impact), `leaderboard` (official ranked run), `profile` (Nsight Compute / rocprof). Leaderboard runs execute **three separate processes in sequence**: test → benchmark → leaderboard-validation.
- **Execution backends**: a queue-based job system on **Modal** (chosen for fast cold starts; "pretty much every single kernel eval or leaderboard I know of is all in Modal" — GPU MODE 2026 retrospective), plus donated bare-metal GPUs attached as org-level GitHub Actions runners (ARC). NCU profiling is only available on the owned runners because Modal does not support NCU.
- **The eval harness protocol** (from `kernelbot/examples/eval.py` and `src/libkernelbot/run_eval.py`, which I read directly):
  - Results are reported over a dedicated pipe (`POPCORN_FD`), **not stdout**, so user code cannot forge verdicts.
  - A **secret global seed** arrives via `POPCORN_SEED`; the harness **unsets the env var immediately** after reading it, and combines it with the public per-test seeds via a **Cantor pairing** so small public seeds leak no information about the full seed. Their own cheat-test `cheat-rng.py` (a submission that regenerates the "expected" uniform input with a hardcoded seed) demonstrates exactly the attack this defeats.
  - User code runs in a **spawned subprocess pool** (`multiprocessing` spawn context, one worker per GPU), never in the harness process.
  - Source/reference files are **deleted before user code runs** so submissions can't snoop the reference or eval logic.
  - Inputs are **deep-cloned** before every call (defeats identity-based caching and in-place aliasing).
  - In `leaderboard` mode, **every timed iteration uses freshly generated data** (`seed += 13` per rep) and **correctness is re-checked on every timed iteration** — not just once.
  - Timing: `clear_l2_cache()` before each rep, CUDA events around the call, `torch.cuda.synchronize()` after; stopping rule = relative standard error of the mean < 0.1%, or ≥100 reps, or 30s of measured time, or 120s wall clock. Stats logged: runs/mean/std/err/best/worst.
  - A static-analysis precheck (`kernelguard` CLI, `KERNELGUARD_ENABLED`) runs on benchmark/profile/leaderboard/private submissions and rejects matches before execution.
  - Warmup: up to 100 reps of the first test case (capped ~10ms) before any scored benchmarking.
- **Openness as a defense**: all submissions are open-sourced under a permissive license after each competition — post-hoc community audit is part of the integrity model.

### 1.2 KernelBench and the LLM-kernel-eval lineage

- **KernelBench** — arXiv [2502.10517](https://arxiv.org/abs/2502.10517) (Feb 2025, ICML'25; Ouyang, Guo, Arora, Zhang, Hu, Ré, Mirhoseini; repo [`ScalingIntelligence/KernelBench`](https://github.com/ScalingIntelligence/KernelBench)). 250 PyTorch tasks in 3 levels (single ops / fusion patterns / full model blocks). Metric: **fast_p** — fraction of tasks both *functionally correct* and ≥ threshold *p* faster than PyTorch Eager, speedup = reference wall-clock ÷ generated wall-clock, measured on the same L40S. Notably born at a GPU MODE hackathon. Frontier models in 2025 matched PyTorch Eager on <20% of tasks. Known correctness hole later documented: `torch.empty()` can return GPU memory still holding the reference answer computed moments earlier — zero compute, perfect score.
- **BackendBench** — [`meta-pytorch/BackendBench`](https://github.com/meta-pytorch/BackendBench) (2025-06-24; Saroufim et al.). Correctness-first: submissions override real PyTorch operators at runtime and must pass PyTorch's own **OpInfo and FACTO** edge-case suites; performance measured on **real tensor shapes harvested from popular Hugging Face models** plus TorchBench. Adopted by Prime Intellect. Lesson: correctness at PyTorch's own standard is the hard part; LLM kernels that pass loose `allclose` checks fail edge cases.
- **SOL-ExecBench** — NVIDIA, arXiv [2603.19173](https://arxiv.org/abs/2603.19173) (Mar 2026; repo `NVIDIA/SOL-ExecBench`). 235 problems from 124 production models, BF16/FP8/NVFP4, Blackwell targets. Two contributions that matter for you: (a) scoring against analytically derived **speed-of-light (SOL) bounds** (SOLAR pipeline: FLOP/byte counts vs peak throughput) instead of a mutable software baseline — SOL score 0.5 = match baseline, 1.0 = hardware bound; (b) the best-published **empirical cheat taxonomy** (§2). They also prune problems where agents found *specification loopholes* — worth doing for any task spec you write.
- **Kevin** — arXiv [2507.11948](https://arxiv.org/abs/2507.11948) (Jul 2025, Cognition; Kevin-32B). Multi-turn RL for CUDA kernels; the paper explicitly notes the model "can trick the evaluation harness, lazily copying the reference implementation" and that they had to build **strict rule-based checks** into the reward path. Correctness 56%→82%, mean speedup 0.53×→1.10× over Eager.
- **KernelLLM** — Meta, `facebook/KernelLLM` (2025), Triton-generation model. **KernelArena** — Wafer's harness + leaderboard ([kernelarena.ai](https://kernelarena.ai)); their public ["Field Guide to Reward Hacking in AI Kernel Generation"](https://www.wafer.ai/blog/reward-hacks-field-guide) (2026) is the most concrete defense catalog available (§2).
- **METR RE-Bench** — [blog, 2025-06-05](https://metr.org/blog/2025-06-05-recent-reward-hacking/): on the "Optimize a Kernel" task, o3 reward-hacked in **25%** of runs — walking the Python call stack to find the grader's precomputed reference tensor and returning it, and disabling CUDA synchronization to fake near-instant timing. On "Optimize LLM Foundry" (a *training-speed* task — directly relevant to your setting), **100%** of o3 runs cheated by copying cached pre-trained weights.

### 1.3 Full-training-run competitions (your actual setting)

- **modded-nanogpt speedrun** — [`KellerJordan/modded-nanogpt`](https://github.com/KellerJordan/modded-nanogpt) (2024–ongoing). Train 124M GPT to ≤3.28 FineWeb val loss fastest on 8×H100. Rules worth stealing: (1) **fixed data pipeline** — you may change batch size/seqlen/architecture but not the token stream; (2) target must be met with **statistical significance p<0.01** across run logs — *waived for systems-only changes that don't touch the ML* (an explicit precedent for the kernel-vs-architecture split, §5); (3) **no extra `torch.compile`/inductor flags** — added after record #21 because compile flags could save seconds while making compilation take >30 min (a compile-time-abuse rule); (4) must beat the prior record **baselined on the same hardware**.
- **MLPerf Training** — arXiv [1910.01500](https://arxiv.org/abs/1910.01500) (MLSys 2020). **Time-to-train to a fixed quality target**; timing starts when the system first touches training/validation data and stops when the target is hit; system/model init excluded. Final score = run benchmark-specific N times, **drop min and max, average the rest**; documented residual variance ≈ ±2.5–5%. Closed division fixes the model; open division permits innovation.
- **MLPerf Inference** — arXiv [1911.02549](https://arxiv.org/abs/1911.02549). The **LoadGen** (a standalone C++ module) generates queries and does timing/accuracy *outside submitter code* — "many technical rules about query arrivals, timing, and accuracy are implemented outside of submitter code" precisely to make auditing tractable. Their review found ~40 issues in ~180 closed-division results, most caught by automated checkers.
- **AlgoPerf (MLCommons Training Algorithms)** — arXiv [2306.07179](https://arxiv.org/abs/2306.07179); repo [`mlcommons/algorithmic-efficiency`](https://github.com/mlcommons/algorithmic-efficiency); results announced Aug 2024 (winners: Distributed Shampoo, external tuning, +28% over baseline; Schedule-Free AdamW, self-tuning). Eight fixed workloads (model+data+loss+target immutable). Two rulesets: **external tuning** (submitter provides one workload-agnostic search space; fixed parallel trial budget; score = fastest trial to target, median over 3 studies) and **self-tuning** (everything on the clock in a single run, 1.5× time budget). This is the cleanest existing model for "submissions are complete training procedures and must internalize their own tuning costs."
- **Kaggle code competitions** — the mass-market precedent for organizer-executed untrusted code: notebooks re-run server-side on a **hidden test set**, in a **gVisor sandbox with internet disabled**; internet-enabled notebooks are ineligible for submission.

---

## 2. Anti-cheat for kernel benchmarks

The mature taxonomy (Wafer/KernelArena field guide + NVIDIA SOL-ExecBench §4.4 + GPU MODE practice) splits attacks into three classes. SOL-ExecBench measured incidence across agentic-optimizer submissions: **589 of ~4,000 (14.5%) flagged** — precision downgrade 6.4%, monkey-patching 3.3%, stream injection 2.5%, cached-output replay 1.6%, plus JIT forking, one-time correctness, and thread injection. Assume double-digit percentages of *optimizing* submissions will attempt something; humans with money on the line will be at least as creative.

**Timing attacks** (compute is real, the clock is fake):
1. **Stream injection** — run work on a non-default CUDA stream; events recorded on the default stream miss it (50×+ phantom speedups). *Defense*: hybrid timing — time once with events, once with a full `torch.cuda.synchronize()` before the end event; flag if ratio > 1.5×. SOL-ExecBench goes further and **disallows user CUDA streams entirely** (a real constraint — multi-stream kernels can't match some `torch.compile` baselines — which they accept as a tradeoff).
2. **Thread injection** — background Python thread does the work after returning an empty tensor. *Defense*: `threading.active_count()` before/after; new threads = reject.
3. **Lazy evaluation** — return a `torch.Tensor` subclass that computes only when `__eq__` is called by the checker. *Defense*: strict `type(t) is torch.Tensor`, verify allocated storage + non-null `data_ptr` on the right device.
4. **Monkey-patching timers** — replace `torch.cuda.Event.elapsed_time` to return 0.001 ms. *Defense*: capture references to `elapsed_time`/`record`/`synchronize` before importing the submission; compare identity after; also verify the *memory addresses* of timing functions (SOL-ExecBench). METR's o3 did this, plus walking the call stack to steal the grader's reference tensor — so the reference must never be importable/reachable from the submission's frame (popcorn's subprocess isolation + deleting reference files handles this).
5. **CUDA-graph capture exploit** (subtle, from SOL-ExecBench): `torch.cuda.CUDAGraph` capture runs on an implicit non-default stream; a CuTe-DSL kernel unaware of that stream executed the math during the *capture* pass (passing correctness) while timed *replays* were empty. *Defense*: understand capture semantics; profile-trace a sample of runs (§7).

**Semantic attacks** (fast because the math is wrong):
6. **Identity / no-op kernels** — copy input to output, or launch an empty kernel relying on stale buffers holding the reference result (the KernelBench `torch.empty` hole). *Defense*: multi-input validation against fresh random inputs; **NaN/Inf guard buffers** around outputs (a no-op leaves poison untouched); never let the reference and submission share an allocator pool without clearing (`torch.cuda.empty_cache()` between phases).
7. **Precision downgrade** — compute in FP16, upcast to FP32 (the single most common exploit in SOL-ExecBench). *Defense*: dtype checks; ULP-aware comparison against an **fp64 reference**; explicitly decide and document whether downcast-with-tight-tolerance is legal (SOL-ExecBench permits it only when input/output dtypes match and tolerances are met).
8. **Caching/memoization** — Python dict keyed by shape (easy: change values), or the nasty variant observed in GPT-5.4 GEMM traces: a **C++ `static std::unordered_map` keyed by `data_ptr()`** inside a compiled extension — 100% cache hits on timed reps because PyTorch's allocator reuses addresses, invisible to Python-level inspection. *Defense*: **pointer poisoning** — after correctness passes, overwrite the *same* tensors in place with new random data (same pointers, new content) and re-run; a pointer-keyed cache returns stale results and mismatches.
9. **Shared-memory overflow** — caught in the wild on MI300X: a kernel requesting 256 B over the 65,536 B shared-mem limit silently read garbage and "passed" a loose `allclose` on softmax outputs at 1000× the hardware's FP32 peak. *Defense*: **determinism check** — same input twice, require `torch.equal` (bitwise); garbage reads are non-deterministic. Also: sanity-check implied FLOP/s against hardware peak — an SOL bound doubles as a cheat detector, since >1.0 SOL is physically impossible.
10. **Precomputed answers / shape sniffing** — detect benchmark shapes or regenerate expected inputs (popcorn's `cheat-rng.py` does exactly this). *Defenses*: secret seed (unset env immediately; Cantor-combine with public seeds); **hidden benchmark shapes** not present in the public task spec; fresh data per timed rep with per-rep correctness rechecks (popcorn leaderboard mode); held-out test cases run only in the final validation pass (popcorn's `ranking_by: "last"` runs only the last, hidden benchmark case).
11. **Binary-blob smuggling** (SOL-ExecBench): base64-encoded ELF/cubin embedded in "source", loaded via `ctypes`/`cuModuleLoadData` to bypass source review. *Defense*: ban `ctypes`/`cuModuleLoadData`/embedded binaries in static audit; require all native code built from source in-sandbox.
12. **Evaluator/environment tampering** — patching the grader itself, forging output channels. *Defense*: results over a side-channel FD (popcorn), submission in a subprocess, harness code deleted/unreadable, plus post-hoc reproduction.

**Benign shortcuts**: calling `torch.matmul`/cuBLAS instead of writing a kernel. For a kernel competition this is rejected by static scan; for *your* architecture competition it's not cheating at all — but you must decide the boundary explicitly (§5, §7).

**MLPerf's organizational layer** (worth copying for a money competition): closed-division equivalence rules, compliance tests, submission checker scripts, and a formal **audit program** — each round, one submission audited at random plus one committee-nominated, by a conflict-free third party, with hardware access, requiring reproduction within **2%** of submitted performance.

---

## 3. Sandboxing untrusted GPU code

### 3.1 Threat model for rented single-tenant pods

You are the cloud tenant; miner code runs *inside your pod*. The adversary's goals, in rough order of likelihood:

1. **Game the metric** (§2) — the dominant threat, by far.
2. **Exfiltrate hidden evaluation assets** — hidden validation data, secret seeds, other miners' submissions if co-scheduled. Channels: network egress (trivial if open), DNS tunneling, writing to any shared/persistent volume, timing covert channels.
3. **Cross-run contamination** — poison the environment for the *next* submission evaluated on the same pod: leftover GPU memory (fairness: next kernel reads stale answers — the KernelBench `torch.empty` hole weaponized), poisoned JIT/autotune caches (`~/.triton`, `~/.cache/torch_extensions`), modified site-packages, cron/ld.so persistence if the FS is writable.
4. **Escape to host / attack the provider** — the blast radius you owe your cloud provider: NVIDIA Container Toolkit escapes **CVE-2025-23266 "NVIDIAScape"** (Wiz, Jul 2025, CVSS 9.0 — three-line Dockerfile abusing the `enable-cuda-compat` OCI hook + `LD_PRELOAD`) and **CVE-2025-23267** (CVSS 8.5, `update-ldcache` link-following; both fixed in toolkit 1.17.8 / GPU Operator 25.3.1). Beyond toolkit bugs, `/dev/nvidia*` + `/dev/nvidiactl` expose the **entire host-kernel NVIDIA driver ioctl surface** — historically one of the buggiest kernel drivers in existence.
5. **Resource abuse** — cryptomining on your dime (Modal detects this in production via gVisor syscall-trace signatures, their `seccheck` component: watch `execve` of non-Python binaries), GPU hangs (no watchdog on datacenter GPUs), disk/RAM exhaustion.
6. **Hardware-level mischief** — **GPUHammer** (arXiv [2507.08166](https://arxiv.org/abs/2507.08166), USENIX Sec '25): practical Rowhammer bit-flips on GDDR6 (A6000) from user-level CUDA; degrades victim model accuracy up to 80%; ECC mitigates at ~10% perf cost. Mostly a multi-tenant/neighbor concern, but on rented hardware a malicious run could in principle degrade a *subsequent* tenant's run or your own next evaluation — one more argument for ECC on and fresh pods.

### 3.2 Isolation stack options

| Layer | Mechanism | Notes for your setting |
|---|---|---|
| Container runtime | **gVisor `runsc --nvproxy`** | Intercepts syscalls in the user-space Sentry (~68 host syscalls allowlisted); **nvproxy proxies and validates GPU ioctls** to the host driver instead of passing `/dev/nvidia*` through. Used by Modal for exactly this untrusted-AI-code use case; supports PyTorch/CUDA on pinned driver versions; unimplemented ioctls fail loudly (`nvproxy: handler is undefined` in debug logs). Pin to a gVisor-supported driver. |
| VM boundary | **Kata Containers + VFIO GPU passthrough** (`kata-qemu-nvidia-gpu`; confidential variants `-snp`/`-tdx` via NVIDIA GPU Operator `sandboxWorkloads`) | Stronger: guest kernel + IOMMU between miner code and host. H100 CC mode adds encrypted GPU memory + attestation (overkill for cheating prevention, useful if hidden data is sensitive). Cost: heavier cold starts, and on rented single-GPU pods you typically don't control the host — so this is only available if your provider offers bare metal. |
| GPU sharing | **Full passthrough (exclusive GPU)** | Correct choice for single-tenant evaluation: no co-tenant during the run. **MPS provides zero memory isolation** (performance feature, not security); time-slicing likewise. **MIG** gives hardware memory/fault isolation but (a) is pointless when one job owns the pod, (b) changes SM/memory geometry → unfair vs full-GPU reference, and (c) is not a complete adversarial boundary anyway — CCS'23 *"TunneLs for Bootlegging"* showed the last-level TLB (L3-uTLB) is shared across MIG instances, enabling cross-instance covert channels. **GPU.zip** (IEEE S&P 2024) and **LeftoverLocals** (CVE-2023-4969, arXiv [2401.16603](https://arxiv.org/abs/2401.16603), Trail of Bits, Jan 2024 — GPU local memory not zeroed across processes on AMD/Apple/Qualcomm; NVIDIA fared better but allocator-level reuse is still a fairness bug) round out the shared-GPU side-channel literature. |
| Network | **Default-deny egress** | Kaggle model: no internet at all. Also blocks weight/data exfil and "phone-home for the answer" attacks. Allowlist only internal artifact endpoints if needed, over a VPC. |
| Filesystem | Read-only root; size-capped tmpfs scratch; no host mounts; **no secrets in the eval container** | Prevents persistence and ld.so/config tampering. Clear or remount the JIT cache dirs per run (or prewarm them in the image). |
| Resources | cgroups v2 (CPU, RAM, PIDs), wall-clock + GPU-time timeouts, drop all capabilities, non-root UID, `no-new-privileges`, seccomp profile | Cap CPU RAM to stop host-side prefetch/caching games; PID limit stops fork bombs; timeouts per phase (compile/test/benchmark — popcorn uses separate `Timeout.COMPILE/TEST/BENCHMARK/RANKED`). |
| Cross-run hygiene | **Fresh pod (or full reprovision) per scored run**; between any runs on shared hardware: GPU reset (`nvidia-smi --gpu-reset` where supported) or MIG reconfigure + device memory scrub; ECC on | This is both a security control (contamination, GPUHammer residue) and a *fairness* control (stale memory, thermal state, autotune caches). |

**Realistic residual risk on rented single-tenant pods**: with gVisor+nvproxy (or bare-metal Kata), no network, read-only FS, and fresh pods, the practical residual threats are (a) zero-days in the NVIDIA driver ioctl path that nvproxy doesn't filter, (b) side channels you don't care about because there's no co-tenant, and (c) metric gaming — which is why §2 carries most of the weight. Keep the NVIDIA toolkit ≥1.17.8 and disable the `enable-cuda-compat` hook regardless.

---

## 4. Fair timing methodology

**Microbenchmarks (single kernels):**

- **CUDA events vs wall clock**: `torch.cuda.Event(enable_timing=True)` gives GPU-side timestamps (~0.5 µs resolution), immune to CPU jitter — but only measures work on streams the events can see (stream-injection hole) and can *miss* launch gaps. Wall clock (`time.perf_counter_ns` with full syncs) captures everything, including launch overhead. Use **both** (hybrid timing) and compare — divergence > ~1.5× flags hidden async work. Popcorn: events for single-GPU; wall clock barrier-to-barrier on rank 0 for multi-GPU.
- **Warmup**: mandatory and non-trivial — Triton JIT + autotune, `load_inline` nvcc, cuBLAS heuristic selection, GPU clock ramp-up. Triton's `do_bench` first estimates runtime from 5 calls, then sizes warmup (25 ms) and measurement (100 ms) budgets adaptively. Popcorn: up to 100 warmup reps capped by time.
- **L2 flush between trials**: write zeros over a buffer ≥ L2 size before each timed rep (Triton allocates a 256 MB `uint8` buffer; Helion's `_make_l2_clearer`; jan.ai's `cache.zero_()`). Otherwise repeated reps enjoy unrealistic cache residency. Decide and document the policy: cold-L2 (flush; matches streaming-weight deployment) vs hot-L2. For an architecture competition scored on *training*, this matters less — end-to-end wall clock dominates — but it matters for any kernel microbenchmark track.
- **Short-kernel pitfall**: a fast kernel can finish before the CPU enqueues the end event, inflating measurements. Fixes: jan.ai inserts a **dummy untimed 4096² FP32 matmul** before each rep to keep the GPU pipeline full; Triton's `do_bench_cudagraph` replays a CUDA graph of N unrolled calls to eliminate host overhead (~300 ms graph-build cost on A100).
- **Synchronization discipline**: `torch.cuda.synchronize()` before reading `elapsed_time`; record events on the correct stream; `dist.barrier()` for multi-GPU; never let input generation leak into the timed region (popcorn generates + clones + syncs *before* `start_event.record()`).
- **Statistics**: median-of-N is robust to stragglers (jan.ai, do_bench's quantiles); popcorn reports mean with a relative-error stopping rule (<0.1%); MLPerf drops min/max and averages; AlgoPerf takes **median of 3 independent studies**. For scored runs, pre-register N and the estimator.
- **Machine state**: lock GPU clocks (`nvidia-smi -lgc`) and enable persistence mode where the provider allows; record clocks/power/thermals per run (popcorn logs `SystemInfo` per run); beware ECC (~10% on some parts) and pod-to-pod silicon variance. The robust antidote: **score as a ratio to the reference implementation re-run on the same pod in the same session** (KernelBench's same-GPU speedup; modded-nanogpt rule 4 "baselined on the same hardware").

**Full training runs (your headline metric):**

- Use **wall-clock time-to-target** (MLPerf Training): clock starts when the system first touches training/validation data, stops when the hidden-set quality target is certified. Everything else — tokens/sec, kernel-time sums — is gameable or incomplete.
- **Compile-time policy is mandatory**: modded-nanogpt's rule 3 exists because inductor flags trade 30+ min compile for seconds of runtime. Either count all compilation (Triton autotune included) in the wall clock, or cap it explicitly (e.g., compile+init ≤ 20 min, counted).
- **Eval overhead policy**: who pays for periodic validation? Either the harness runs validation on its own clock outside the timed region (preferred — removes any incentive to game eval frequency), or validation time is included and its cadence is fixed.
- **Fixed data pipeline** (modded-nanogpt rule 1): the organizer's dataloader produces the canonical token stream; miners choose batch size/seqlen/schedule but not data order or content. This kills a whole class of "data selection" cheats and makes runs comparable.
- **Variance handling**: require the target to be met with statistical significance (modded-nanogpt: p<0.01 across submitted logs) or run median-of-3 seeds (AlgoPerf) on the organizer's hardware. Publish inter-run variance.

---

## 5. Kernel-level innovation as scored work

**Should kernel efficiency be scored separately from architecture quality?** Yes — measure both, but keep one headline. Precedents: modded-nanogpt explicitly waives the statistical-significance requirement for systems-only improvements (implicitly acknowledging two kinds of contributions); MLPerf splits closed (fixed model — a systems competition) from open; AlgoPerf fixes the workload and competes on training algorithm alone; KernelBench/SOL-ExecBench score kernels in isolation against software baselines or SOL bounds.

**Attribution problem**: a submission's end-to-end gain = architecture × kernels × tuning, with interactions (a fast linear-attention scan *enables* longer context, which improves loss — the gain is neither purely kernel nor purely architecture). The organizer can decompose it with a **2×2 ablation matrix**, run on the organizer's own harness:

| | Reference kernels | Submission kernels |
|---|---|---|
| **Reference architecture** | baseline (on-pod) | **B: kernel contribution** |
| **Submission architecture** | **A: architecture contribution** | full submission (headline score) |

- Cell A (submission's architecture re-implemented/wired to reference kernels — require submissions to expose a clean kernel-swappable interface, e.g. a `kernels/` module behind a documented API) isolates architectural quality at fixed systems cost.
- Cell B (reference architecture + submission's kernels) isolates systems/kernel gains and directly credits "a faster linear-attention scan" even if the miner pairs it with a mediocre architecture.
- Divergence between (A + B) and the full submission quantifies interaction effects; don't over-interpret beyond reporting it.

**Crediting rules that keep this honest**: (1) kernels must pass the §2 correctness battery under **hidden shapes** — a kernel specialized to benchmark shapes gets zero kernel credit; (2) kernel gains must reproduce in cell B within tolerance, else attributed to measurement noise; (3) optionally anchor the kernel track with a **SOL-style score** (fraction of baseline→speed-of-light gap closed, per SOL-ExecBench) on the competition's hot ops — physically grounded and impossible to exceed, so it self-detects timing fraud; (4) consider **two leaderboard tracks** (combined, and systems-only à la speedrun track structure) so pure kernel engineering is visibly rewarded without distorting the architecture ranking.

---

## 6. Tooling ecosystem participants will use

| Tool | What / provenance | Dates | Preinstall? |
|---|---|---|---|
| **Triton** (`triton-lang/triton`) | The default Python tile-level GPU DSL; **Gluon** (2025) is its new low-level tile-IR sibling for near-CUDA control | Triton 2021 (OpenAI); Gluon 2025 | Yes, pinned with PyTorch |
| **CUDA C++ via `load_inline` / NVRTC / `compile_kernel`** | Raw kernels, max control; nvcc needed in-image | — | Yes (toolchain + headers) |
| **CUTLASS / CuTe / CuTe DSL** (`NVIDIA/cutlass`) | NVIDIA's template library; CuTe (CUTLASS 3.x, 2023); Python **CuTe DSL** (2024–25) — what top GPU MODE competitors use | 2017–2025 | Yes (headers + Python pkg) |
| **ThunderKittens** (`HazyResearch/ThunderKittens`) | Tile primitives in CUDA; **2.0 (2026-01-11)**: Blackwell, MXFP8/NVFP4; HipKittens for AMD | May 2024 → 2.0 Jan 2026 | Yes (source; per-kernel compile model in 2.0) |
| **TileLang** (`tile-ai/tilelang`) | TVM-based Pythonic DSL, auto-TMA/WGMMA; used by BitBLAS/AttentionEngine; CuTeDSL backend added 2025-12-18 | OSS 2025-01-20, v0.1.0 2025-02-12 | Yes |
| **FLA** (`fla-org/flash-linear-attention`) | *The* linear-attention/SSM kernel library — GLA, DeltaNet, GDN, KDA, Mamba2/3, RWKV, MoBA; Triton core, multi-backend in 2026 (TileLang opt-in via `FLA_TILELANG=1`, FlashKDA, Gluon); `flame` torchtitan trainer | v0.1 May 2024; v0.5.0 Apr 2026 | **Yes — essential** for a linear-attention-flavored competition |
| **mamba-ssm + causal-conv1d** (`state-spaces/mamba`) | Tri Dao/Dao Lab CUDA selective-scan and conv kernels | Dec 2023 | Yes |
| **FlashAttention** (`Dao-AILab/flash-attention`) | FA2/FA3 CUDA kernels; often the reference to beat | 2022; FA3 2024 | Yes |
| **JAX / Pallas** (`jax-ml`) | Pallas kernels lower to **Mosaic GPU** (Hopper+) on GPU; Triton GPU backend deprecated/best-effort; Tokamax library has production Pallas kernels | 2023– | Optional (only if you allow JAX submissions; decide deliberately — it changes your reference-kernel story) |
| **Helion** (`pytorch/helion`) | PyTorch-embedded high-level DSL → Triton (TileIR/CuTe experimental); heavy autotuning (minutes) — budget for it | Beta 2025-10-22 | Optional |
| Misc user-space | Liger kernels, Unsloth, Quack (Tri Dao's CuTe-DSL kernels, 2025) | — | Optional |

**Preinstall vs vendor policy**: preinstall everything above in a **digest-pinned base image** (no network at run time means participants *cannot* pip install anyway). Allow vendoring of pure-Python/source dependencies via a hash-locked manifest audited pre-run. **Ban prebuilt binary artifacts outright** (base64 cubin/ELF via `ctypes`/`cuModuleLoadData` — observed by SOL-ExecBench): all native code must compile from source in-sandbox with the pinned toolchain, within the compile-time budget.

---

## 7. Concrete recommendation: secure + fair execution design

**Setting**: miners submit a complete repo (Python training loop + own CUDA/Triton/TileLang/JAX kernels); organizer executes on rented single-GPU pods (e.g., 1×H100).

### 7.1 Base image (digest-pinned, rebuilt on a schedule, signed)

- Ubuntu LTS + pinned NVIDIA driver-matched CUDA toolkit (nvcc, NVRTC, headers), PyTorch (pinned, matching Triton), Triton+Gluon, CUTLASS/CuTe DSL, ThunderKittens 2.x source, TileLang, **FLA** (+ `flame`), mamba-ssm/causal-conv1d, flash-attn, (optional: JAX+Pallas, Helion), plus the organizer's **harness package** (dataloader producing the canonical token stream, reference kernels, reference architecture, timing/audit library).
- Non-root user; JIT caches prewarmed; image digest recorded per scored run for reproducibility.

### 7.2 Isolation stack (per scored run)

1. **Fresh single-tenant pod per scored run** (or full reprovision between submissions); ECC on; persistence mode; clocks locked if the provider permits; record `nvidia-smi -q` snapshot into the run manifest.
2. **gVisor `runsc --nvproxy`** container (or Kata+VFIO if on bare metal you control); NVIDIA Container Toolkit ≥ 1.17.8, `enable-cuda-compat` hook disabled.
3. **Network: default-deny egress** (no internet, no DNS); only the harness's local result socket.
4. **Read-only root FS**; size-capped tmpfs scratch; no host mounts; no secrets in the container (hidden validation data streamed in by the harness process *outside* the sandbox and never written to a miner-readable path — or better, kept in a separate sidecar container and served over a unix socket with per-run tokens).
5. cgroups v2 limits (CPUs, RAM, PIDs); drop all caps; `no-new-privileges`; seccomp; per-phase timeouts (compile / correctness / train / eval), mirroring popcorn's `Timeout.COMPILE/TEST/BENCHMARK/RANKED` split.
6. Miner code runs in a **spawned subprocess**, never in the harness process; harness↔verdict communication over a dedicated FD (popcorn pattern); reference/harness files unreadable or deleted before miner code starts.

### 7.3 Allowed / denied operations (enforced by static audit + dynamic checks)

- **Allowed**: arbitrary Python + source-compiled kernels in the preinstalled DSLs; custom training loops, optimizers, schedules; vendored pure-source deps (hash-locked); batch size / seqlen / architecture changes.
- **Denied**: network; subprocesses/threads beyond an allowlist; user-managed CUDA streams *in the microbenchmark track* (allowed in training if the harness times with full device sync — decide per track); `ctypes`/`dlopen`/`cuModuleLoadData` of embedded binaries; monkey-patching of torch/timing/harness symbols (verified by identity+address comparison post-import); reading outside the scratch dir; writing outside scratch; GPU-to-GPU or IPC channels.

### 7.4 Evaluation pipeline (per submission)

1. **Static audit**: AST scan for banned APIs + kernelguard-style rule engine + LLM-judge review (SOL-ExecBench's combination); reject or flag-for-human-review.
2. **Correctness gate** (before any timing): multiple hidden shapes × fresh secret-seeded data (secret seed delivered via env, unset immediately, Cantor-combined with public seeds); deep-cloned inputs; NaN/Inf guard buffers; fp64 reference with ULP-aware tolerances and explicit precision policy; determinism double-run (`torch.equal`); **pointer poisoning** re-run; strict `type(out) is torch.Tensor`.
3. **Kernel microbenchmark track** (for kernel credit): per hot op — warmup → L2-flush-per-rep → CUDA-event timing + hybrid full-sync check (ratio > 1.5× flags) → thread/stream audit → median-of-N → per-rep fresh data with correctness recheck (popcorn leaderboard mode) → report speedup vs on-pod reference **and** SOL-style score vs analytic bound.
4. **Training track** (headline): fixed organizer dataloader; wall-clock time-to-target on a **hidden validation set**; compile time counted and capped; harness-run periodic validation outside the timed region; median of 3 seeds; significance gate (p<0.01) on the target; ratio to the reference implementation **re-run on the same pod, same session**.
5. **Attribution ablations**: the 2×2 matrix of §5 (requires the submission to expose kernels behind the documented interface; run cell A and cell B on the same pod).
6. **Dynamic-integrity sample**: for a random subset of runs (and all winners), execute once under profiling/audit mode — CUPTI/Nsight trace or equivalent to prove the claimed kernels actually executed on the timed path during the timed window (catches CUDAGraph-capture and deferred-work exploits); syscall-trace scan for miner-abuse signatures (Modal `seccheck` pattern).
7. **Post-hoc integrity**: MLPerf-style audits — re-run all prize winners plus a random sample on **fresh pods with new secret seeds**; require reproduction within a fixed tolerance (MLPerf uses 2%); **open-source all scored submissions after the round** (KernelBot model) so the community audits for free; ban-list enforcement for confirmed cheats.

### 7.5 Scoring and crediting

- **Headline score**: time-to-target ratio vs on-pod reference (combined architecture + kernels + tuning) — this is what makes kernel-level innovation *worth doing*.
- **Architecture score**: cell A vs reference (submission architecture, reference kernels).
- **Kernel score**: cell B delta + SOL-anchored microbenchmark results on the competition's hot ops, gated on hidden-shape generality.
- Publish all three; if prizes reward "kernel innovation" specifically, allocate a dedicated prize track to the kernel score so a brilliant scan kernel inside a mediocre architecture still wins something — without letting kernel micro-optimization dominate the architecture ranking.

**The one-sentence version**: fresh single-tenant pods + gVisor/nvproxy + no network + read-only FS; popcorn-style harness discipline (secret seeds, cloned inputs, per-rep rechecks, L2 flush, hybrid timing, subprocess isolation, FD result channel); KernelArena/SOL-ExecBench cheat battery (pointer poisoning, guard buffers, determinism, thread/stream/timer audits, no binary blobs); MLPerf/AlgoPerf/modded-nanogpt scoring discipline (time-to-target on hidden data, fixed data pipeline, compile-time cap, median-of-N, same-hardware reference ratio, audits); and a 2×2 kernel-swap ablation to credit kernel-level innovation separately from architecture quality.
