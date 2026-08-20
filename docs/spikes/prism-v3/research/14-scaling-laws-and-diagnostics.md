# Appendix 14 — Scaling Laws and Automated Diagnostics under 6 h / 4×5090 / 1 B
> Research appendix for the Prism v3 evaluation proposal (`docs/spikes/prism-v3/`). Produced 2026-08-16 via arXiv/web research. Non-normative spike document.

**Numbering note:** filed as appendix **14**, not 11 — `11-sample-efficiency-scaling.md`
already exists and is cited from `harness/eval/g6_curve.py`. This report covers
adjacent ground (scaling-law fitting, G6/G8 diagnostics) but does not supersede it.

**Authority:** non-normative. Per [`docs/AGENTS.md`](../../../AGENTS.md), when a
spike conflicts with a frozen spec ([`docs/PRISM.md`](../../../PRISM.md),
[`docs/BUNDLE_SPEC.md`](../../../BUNDLE_SPEC.md), the pre-registered anchor sets)
**the normative doc wins**. Nothing below is a scoring contract.

## Errata — this report inspected the wrong checkout

The report was produced against `/root/gbase` (anchors v0, 350 M cap, 1×RTX 5090)
and says so honestly in §1.1, but the target of its recommendations was the
`prism-v2.1-scoring` worktree (anchors **v2**, **1 B** cap, **4×RTX 5090**). Its
§1.1 table ("`anchors/v2.json` absent", "`org.g8.mup_scaling_slope` absent from
the entire repo", "1× RTX 5090") is therefore **wrong for this branch** — all
three exist here. Each bug claim was independently re-verified against this
worktree before anything was changed; the outcome:

| Claim (report §4.1) | Verdict on this branch | Fix |
|---|---|---|
| Bug 1 — `org.g6.auc_log_tokens` inverted + inert | **Confirmed** (v2 inherited the v0/v1 anchor verbatim) | `anchors/v2.json` re-anchored lower-better; v0/v1 stay byte-frozen |
| Bug 2 — `org.g6.tokens_to_threshold` rewards censored runs | **Confirmed** | `eval/g6_curve.py` fail-closes to `CENSORED_TOKENS` |
| Bug 3 — G1/G2 single bootstrap cluster | **Confirmed** | per-doc / per-row cluster ids |
| Bug 4 — eval budget over-subscription | **Confirmed, arithmetic wrong** — the report's ~4.75 h double-counts G5: `g5_ruler` 1200 / `g5_babilong` 900 / `natural` 900 are *shares* of G5's 3600 s (`g5_longctx._BUDGET_SHARE`), not additions to it. True ceiling sum is **14 100 s ≈ 3.92 h**. The pod over-subscription is *worse* than reported (~9.78 h, not ~9.3 h) | one global battery budget with fractional group shares; pod cap raised to fit |

Two further report recommendations were **not** adopted, deliberately:

- **Renaming to `org.g6.auc_log_bytes`** (report §4.1, §4.2). The probe curve
  carries `{step, tokens_seen, wall_s, probe_loss}` and no byte counts, so a
  bits/byte form cannot be computed without changing the miner-visible probe
  contract. Renaming without changing the computation would make the key lie.
  The direction bug is fixed in place; the bits/byte form stays a v3 item.
- **De-weighting Winogrande / BoolQ / ARC-challenge / OpenBookQA** (report §4.4
  conclusion 2). Group and metric weights are a governance decision. Those
  tasks keep their weights and their 200-item cap; only the four discriminative
  tasks got a raised cap.

---

## Résumé exécutif (FR)

1. **Non, on ne peut pas « démontrer » une loi d'échelle dans ce budget** — et ce n'est pas le bon objectif. La littérature récente (Choshen et al., ICML 2025 ; « Small-Scale Experiments: Are We There Yet? », 2026) exige ≥ 3 tailles de modèle, idéalement 4–5, avec une recherche d'hyperparamètres lourde (64–256 configs/échelle pour une extrapolation fiable). Prism dispose de ~0,5 h utile : hors de portée.
2. **Ce qui EST mesurable et discriminant:** l'ordonnancement d'architectures à budget identique (niveau de perte à FLOPs fixés), la qualité du transfert de LR (µP), la forme de la courbe d'apprentissage, et un **exposant local différentiel** contre une architecture de référence épinglée.
3. **La métrique `mup_scaling_slope` à 2 points est statistiquement non fondée.** Le biais dominant n'est pas le bruit de graine mais le **terme de perte irréductible E**: la pente mesurée vaut `α·(1 − E/L)`, soit seulement **30–56 %** de α selon L (30–49 % avec le E corrigé de Besiroglu). Un modèle simplement meilleur en niveau paraît « mieux scaler ». C'est un confondant, pas un signal.
4. **Le passage du plafond de 350 M à 1 B est contre-productif** à ce budget. À 4×5090 / ~5 h / MFU 30 %, C ≈ 4,5e18 FLOPs ⇒ l'optimum est **N ≈ 140–280 M** selon le ratio tokens/paramètre retenu. Argument sans constantes: un modèle de 1 B ne serait optimal que si le vrai ratio D/N valait **≈ 0,75 token/paramètre** — or aucun ajustement publié ne descend sous 1 (Chinchilla ≈ 20). À 1 B, D/N tombe à 0,75 et la perte se dégrade de ~+0,23 nats. Le plafond 1 B invite les mineurs à s'auto-saboter.
5. **Quatre bugs vivants trouvés dans le code de scoring** (détails § 4.1), dont trois critiques: l'ancre G6 `auc_log_tokens` est **inversée et inerte** (toute valeur plausible sature à 1,0), et `tokens_to_threshold` **récompense les runs censurés** (un modèle qui n'atteint jamais le seuil marque 1,0).
6. **Bug systémique du bootstrap:** G1 et G2 n'émettent **qu'un seul cluster par métrique** ⇒ variance nulle sur **40 % du poids** du composite. Le SE est sous-estimé, la borne LCB trop haute, et la barrière « CI half-width » est vide de sens sur ces axes.
7. **G2 est largement du bruit à cette échelle — et ancré sur des FLOPs équivalents, pas estimé à la main.** Cerebras-GPT-111M a été entraîné à C = 2,6e18, soit exactement le budget de Prism ; par interpolation en log-FLOPs, seuls **LAMBADA, ARC-easy et PIQA** ont une marge exploitable. Winogrande, ARC-challenge, OpenBookQA et BoolQ sont **au niveau du hasard ou en dessous**, et **trois de ces huit termes se normalisent à 0 pour *toutes* les soumissions** — poids mort constant qui plafonne G2 vers 0,625 pour tout le monde. HellaSwag n'a qu'une marge de **2 points** contre un seuil de détection de 8,7 points à n=200: inutilisable au plafond actuel.
8. **Pour l'A/B « boucle vs transformeur »:** attendre le signal sur G1 bits/byte, G3 (rappel associatif), G4 (raisonnement algorithmique) et G7 (coût d'inférence), **pas** sur G2. Une architecture récurrente en profondeur gagne en paramètres, perd en FLOPs/token — et G7 le pénalisera.
9. **Recommandation budgétaire:** rééquilibrer à 4,0 h train / 1,2 h batterie / 0,55 h échelle / 0,25 h marge. Aujourd'hui les plafonds par groupe somment à **~4,75 h** contre un `PRISM_EVAL_TIMEOUT_S` de 3 h, et train+eval peut atteindre **~9,3 h** contre un plafond de vie du pod de 7 h: sur-souscrit, avec troncature silencieuse de la batterie.
10. **Deux correctifs à très fort levier, ~1 h de travail:** (a) définir le *warmup* comme une **fraction du nombre total de pas**, jamais un nombre de pas fixe — Porian et al. montrent qu'un warmup à pas constant pénalise mécaniquement les petits modèles, ce qui saborde le point de base d'une échelle sans qu'aucune ligne de code ne paraisse tricher ; (b) figer/vérifier la version Triton/PyTorch de l'image, car l'écart Triton 3.3→3.7 vaut **~17 % de débit gratuit** sur sm_120.
11. **Justification la plus solide du design Prism:** la correspondance perplexité→capacité résiste aux changements d'échelle, d'hyperparamètres, d'architecture et **même de tokenizer**, mais **casse quand les données de pré-entraînement changent**. Le shard fineweb-edu épinglé est précisément la condition qui rend le classement par perte légitime. Garder G1 dominant ; ne jamais laisser les mineurs choisir leurs données.
12. **Décision demandée:** corriger les bugs d'ancrage/bootstrap **avant** toute bascule `composite` (ils faussent le classement), publier l'échelle à 4 points en **télémétrie observée seulement**, et ne la scorer qu'en v3 après calibration sur les baselines.

---

## 1. Périmètre, méthode, et écart entre la demande et le dépôt

### 1.1 Écart important à signaler

La commande décrit `crates/prism-recipe/anchors/v2.json`, des poids de groupe « v2 », et une métrique `org.g8.mup_scaling_slope` « nouvelle en v2.1 ». **Aucun de ces éléments n'existe dans le checkout `/root/gbase`.** Ce que j'ai vérifié:

| Élément demandé | État réel dans `/root/gbase` |
|---|---|
| `anchors/v2.json` | Absent. Seul `anchors/v0.json` existe ; `LATEST_ANCHOR_VERSION = 0` ; `AnchorSet::load` n'accepte que `0`. |
| `org.g8.mup_scaling_slope` | **Absent du dépôt entier** (`rg` sur tout l'arbre: 0 occurrence). G8 n'expose que `loss_spike_score` et `mup_lr_stability`. |
| Plafond 1 B paramètres | Absent. `MAX_PARAMS = 350_000_000` (`prism-recipe/src/lib.rs`), et `gates.max_params = 350000000` dans v0.json. |
| 4× RTX 5090 | Le dépôt épingle **1× RTX 5090** (`PRISM.md`: « Hard pin: 1× RTX 5090 (non-5090 / multi-GPU rejected at rent) »). |
| Poids de groupe G1..G8 | Présents et cohérents: 0,25 / 0,15 / 0,10 / 0,15 / 0,15 / 0,075 / 0,075 / 0,05. |

**Conséquence méthodologique.** J'ai traité 4×5090, 1 B et v2.1 comme la **cible future** demandée, et v0/350 M/1×5090 comme l'**état vérifié** du code. Toutes mes recommandations précisent laquelle des deux bases elles visent. Les bugs du § 4.1 sont sur le code **réellement présent** et sont donc actionnables immédiatement. Si `v2.json` existe dans le worktree en cours de rebase, mes conclusions sur les bugs G6/G2 doivent être revérifiées contre ce fichier — la logique de normalisation (`composite.rs`) est en revanche partagée et inchangée.

### 1.2 Ce que j'ai lu

`AGENTS.md`, `docs/PRISM.md`, `docs/PRISM_RECIPE.md`, `docs/AGENTS.md` ; `crates/prism-recipe/anchors/v0.json` ; `crates/prism-recipe/src/anchors.rs`, `src/lib.rs` ; `crates/prism-pipeline/src/composite.rs` ; toute la batterie `crates/prism-recipe/harness/eval/` (`common.py`, `g1_intrinsic.py`, `g2_downstream.py`, `g3_recall.py`, `g4_reasoning.py`, `g5_*.py`, `g6_curve.py`, `g7_inference.py`, `g8_stability.py`, `rollup.py`, `natural_docs.py`) ; `harness/prismlib/` (`probes.py`, `telemetry.py`, `train_v3.py`, `eval_v3.py`, `main.py`).

---

## 2. Question 1 — Peut-on démontrer une loi d'échelle dans ce budget ?

### 2.1 What the literature actually requires to *fit* a scaling law

| Requirement | Evidence | Number |
|---|---|---|
| Minimum model count to fit at all | Choshen et al., *A Hitchhiker's Guide to Scaling Law Estimation* (ICML 2025), 485 models / >1000 fitted laws — [arXiv:2410.11840](https://arxiv.org/abs/2410.11840) | **≥ 3** sizes; 4–5 strongly preferred |
| Best achievable extrapolation error | same | **ARE ≈ 4 %** typical floor; up to 20 % still ranks design choices |
| Seed/restart noise | same (cites MultiBERTs) | up to **3.5 % relative** on loss |
| Checkpoint reuse | same | use intermediate checkpoints; discard first ~10 % of training, need ≥ 30–40 % |
| HP search needed at small scale | *Small-Scale Experiments: Are We There Yet?* (2026) — [arXiv:2608.11859](https://arxiv.org/abs/2608.11859) | 4 cfg/scale → **no law**; 16 → unreliable; 64 → weak; **256 → accurate** |
| Fit procedure | Hoffmann et al. 2022; Besiroglu et al. replication — [arXiv:2404.10102](https://arxiv.org/abs/2404.10102) | Huber δ=1e-3 on log-log, LSE parameterization, grid init, **BFGS** (not L-BFGS-B) |
| Reference exponents | Besiroglu re-fit of Chinchilla | α = 0.3478 ± 0.02, β = 0.3658 ± 0.02 (Hoffmann: 0.34 / 0.28) |
| Point count for tight CIs | Besiroglu | Hoffmann's published CIs would need **> 600 000 runs**; they ran < 500 |
| Good-practice grid | *Farseer* (NeurIPS 2025) — [arXiv:2506.10972](https://arxiv.org/abs/2506.10972) | ~1000 LLMs, **3 M H100-hours**, √2-spaced (N,D) grid → 0.50 % extrapolation error |

**Where the literature disagrees — do not paper over this:**

- **Kaplan vs Chinchilla.** Reconciled by Porian et al. ([arXiv:2406.19146](https://arxiv.org/abs/2406.19146)) via three artifacts: last-layer FLOP counting, warmup too long for small models, and scale-dependent optimizer tuning. LR decay was *not* the cause. So a "wrong" exponent is often a **tuning artifact**, not an architectural property — directly relevant to a competition where miners tune.
- **The Chinchilla parametric fit was not reproducible.** Besiroglu et al. found the original fit's confidence intervals implausibly tight and traced it to Huber loss being averaged rather than summed, causing premature L-BFGS-B termination. Treat published E/A/B as convenient, not authoritative.
- **The irreducible term E is unstable.** The 2026 small-scale study found E varies wildly across independent samples of runs, and reports a case study (pre-norm vs post-norm) where **the conclusion flips depending on whether E is tied across architectures.** This is the single most important caution for Prism.
- **µP does not transfer across depth.** Tensor Programs VI shows fundamental limitations for multi-layer blocks ([arXiv:2310.02244](https://arxiv.org/abs/2310.02244)); Bordelon et al. show a 1/√L residual scale is required ([arXiv:2309.16620](https://arxiv.org/abs/2309.16620)); Everett et al. (10k+ models, to 26.8B) find a per-layer *standard*-parameterization prescription can **beat** µP ([arXiv:2407.05872](https://arxiv.org/abs/2407.05872)); u-µP notes µP transfer often fails in practice ([arXiv:2407.17465](https://arxiv.org/abs/2407.17465)). **Prism's G8 must not treat µP-conformance as a proxy for architectural quality.**
- **Downstream scaling laws are unreliable.** "Scaling Laws Are Unreliable for Downstream Tasks" finds a close linear fit in only **39 %** of cases ([arXiv:2507.00885](https://arxiv.org/abs/2507.00885)). This is why G2 must not be the scaling signal.

### 2.2 Compute budget on 4× RTX 5090 (hard numbers)

**Hardware facts** (cited): RTX 5090 = 21 760 CUDA cores, 170 SM, 32 GB GDDR7, **1792 GB/s**, 575 W, PCIe, **no NVLink** ([NVIDIA](https://www.nvidia.com/en-gb/geforce/graphics-cards/50-series/rtx-5090/)). Dense tensor throughput: **BF16/FP16 with FP32 accumulate = 209.5 TFLOPS**; FP16 with FP16 accumulate = 419; FP8 = 419/838; FP4 = 1676 dense. The FP32-accumulate halving is a deliberate GeForce restriction and the 5090 lacks tcgen05/TMEM ([NVIDIA devforum](https://forums.developer.nvidia.com/t/rtx-5090-peak-bf16-tensor-tflops/350543)).

**Use 209.5 TFLOPS/GPU as the training denominator** — 4 GPUs = **838 TFLOPS** peak bf16.

**Software risk on sm_120 (material, cite before assuming MFU):** FlashAttention 2/3 C++ cubins are **absent for sm_120**, and FA3's techniques are WGMMA-dependent so they *cannot* be back-ported; the FA4 CuTeDSL path exists but upstream PRs were still open ([FA PR #2634](https://github.com/Dao-AILab/flash-attention/pull/2634), [CUTLASS PR #3030](https://github.com/NVIDIA/cutlass/pull/3030)). The 5090 uses an Ampere-style `mma.sync` model — architecturally "Blackwell" in name more than in programming model. Triton 3.3 (PyTorch 2.7 stable) crashes matmul autotuning on sm_120 and falls back to Ampere CUTLASS for **~17 % throughput loss**. Transformer Engine on SM120 lags SM100: **MXFP8 forward works but backward does not** (missing cuBLAS non-TN GEMM layouts), confirmed by NVIDIA ([TE issue #2668](https://github.com/NVIDIA/TransformerEngine/issues/2668)). Consumer GPUs also cannot P2P over PCIe, so a memcpy backend is required ([LLMQ, arXiv:2512.15306](https://arxiv.org/pdf/2512.15306)).

**Fairness consequence — pin the container image.** The Triton 3.3 → 3.7 gap is worth **~17 % throughput** for free to any miner who knows to install PyTorch nightly. That is a first-order fairness problem in a one-shot competition, not a footnote. The recipe already pins an image (`daturaai/pytorch:...cuda13.0.2...`, template v9 per `PRISM.md`), so the action is to **assert the Triton/PyTorch version in the harness manifest** and treat a mismatch as an infra fault rather than a silent advantage.

**One asymmetry worth tracking:** the FP32-accumulate halving that caps bf16 at 209.5 TFLOPS **does not apply to FP4**, which runs at the full 1676 TFLOPS dense. So FP4 carries an unusually large prize on this specific GPU (~8× bf16 on paper). That makes it a plausible late-game differentiator — but see the verdict below; it is not something to *assume* a miner can land in one shot.

**Verdict on low precision: do not plan a one-shot 6 h scored run on NVFP4/FP8 training.** NVFP4 pretraining does work at scale (12 B hybrid, 10 T tokens, MMLU-Pro 62.58 vs FP8 62.62 — [arXiv:2509.25149](https://arxiv.org/abs/2509.25149)) but only with random Hadamard transforms, 2D block scaling, stochastic rounding on gradients, and selective BF16 layers. With TE backward broken on SM120, this is a one-shot failure risk, not an upside. Plan **bf16**; treat FP8/FP4 as a miner-side gamble.

**MFU assumption (labelled ESTIMATE, and the central value is contested):** I use **20–35 %**, and quote tables at 25/30/35 %. An independent review of the same evidence argued for a **25 % centre** rather than 30 %, on the grounds that small models have poorer arithmetic intensity and the sm_120 stack is immature. I have kept both in the tables so the conclusion can be checked either way — and it holds at every value in the band. Justification: a published sm_120 measurement reports 73–85 % MFU for *inference prefill* on a 270 M model against the same 209.5 TF denominator, but training adds backward, optimizer, and PCIe all-reduce; LLMQ measures 51–54 % MFU for 4×4090 inference on 14–32 B. For small-model training on consumer Blackwell with no FA3 and possible Triton fallback, 25–35 % is the honest band. **This is the single assumption most worth measuring on your own baseline before trusting any of the tables below.**

**Achievable compute (5.0 h of training):**

| MFU | C = 838e12 × MFU × 18 000 s |
|---|---|
| 25 % | 3.77e18 FLOPs |
| 30 % | 4.53e18 FLOPs |
| 35 % | 5.28e18 FLOPs |

**PCIe all-reduce is not the bottleneck** at these batch sizes (ESTIMATE): bf16 ring all-reduce at ~25 GB/s effective gives ~14 ms/step for N = 1.2e8 and ~120 ms/step for N = 1e9, against ~1.8 s and ~15 s of compute per 0.5 M-token step — **≈ 1 % overhead**.

### 2.3 The decisive finding: the 1 B cap is a trap at this budget

Using the Chinchilla parametric form `L(N,D) = E + A/N^α + B/D^β` with Hoffmann's constants (E=1.69, A=406.4, B=410.7, α=0.34, β=0.28) and `C = 6ND`:

| N | D (MFU 30 %, 5 h) | D/N | L (nats) |
|---|---|---|---|
| 50 M | 15.1 B | 302 | 3.250 |
| 100 M | 7.54 B | 75 | 3.169 |
| **159 M (optimum)** | **4.73 B** | **29.7** | **3.154** |
| 250 M | 3.02 B | 12.1 | 3.168 |
| 350 M (current cap) | 2.16 B | 6.2 | 3.196 |
| 1 B (proposed cap) | 0.75 B | **0.75** | **3.386** |

**Raising the cap from 350 M to 1 B costs ≈ +0.19 nats vs the 350 M cap and +0.23 nats vs the optimum.** The optimum sits at ~160 M across the whole MFU band (133 M @20 %, 147 M @25 %, 159 M @30 %, 171 M @35 %) and D/N ≈ 29–30 — i.e. essentially Chinchilla-optimal, and *below the existing 350 M cap*.

**A constants-free version of the same argument (more robust — read this one).** The table above inherits Hoffmann's E/A/B, which Besiroglu showed were not reproducible. But the conclusion does not depend on them. From `C = 6ND` alone, if the compute-optimal token/param ratio is `r = D/N`, then `N* = √(C/6r)`:

| r = D/N | N* @ C=4.53e18 (MFU 30 %) |
|---|---|
| 40 | 137 M |
| 30 | 159 M |
| **20 (Chinchilla)** | **194 M** |
| 10 | 275 M |
| 5 | 388 M |
| 2 (Kaplan-implied) | 614 M |
| 1 | 868 M |

**Inverting: a 1 B model is compute-optimal at this budget only if the true ratio is ≈ 0.75 tokens/param** (0.50 @20 % MFU, 0.88 @35 %). **No published fit is anywhere near below 1** — Chinchilla is ~20, Besiroglu's corrected `a = 0.5126` is the same order, and even Kaplan's low implied ratio (~2) was traced by Porian et al. to warmup and FLOP-counting artifacts. So across *any* plausible exponent set, the optimum at C ≈ 3–5e18 lands at **140–280 M**, and 1 B is **4–7× past it.**

*Methodological note, stated because I got this wrong first:* I initially tried to sanity-check this by substituting Besiroglu's corrected α/β into Hoffmann's A/B. That is invalid — the amplitudes are jointly fitted with the exponents, so perturbing one without refitting the other produces a meaningless optimum (it moved to 1.2 B with D/N = 0.4, which contradicts Besiroglu's own reported `a`). The constants-free argument above is the one to rely on.

**Recommendation:** keep the cap at 350 M, or if raising it to 1 B for architectural headroom, **state explicitly in miner docs that 1 B is not the optimum** and publish this table. Otherwise the cap change silently rewards whoever ignores it. A cap increase does not become useful until C grows ~30× (i.e. ~150 h on this pod, or a much larger pod).

**Independent confirmation.** Cerebras-GPT trained exactly this regime (Pile, 20 tokens/param):

| Params | Tokens | Pile xent | HellaSwag | PIQA | Winogrande | LAMBADA | ARC-e | ARC-c | OBQA |
|---|---|---|---|---|---|---|---|---|---|
| 111 M | 2.2 B | 2.566 | .268 | .594 | .488 | .194 | .380 | .166 | .118 |
| 256 M | 5.1 B | 2.299 | .274 | .613 | .511 | .293 | .410 | .170 | .158 |
| 590 M | 11.8 B | 2.184 | .291 | .627 | .498 | .366 | .464 | .190 | .158 |
| 1.3 B | 26.3 B | 1.996 | .325 | .664 | .521 | .462 | .508 | .224 | .166 |

The Cerebras 256 M / 5.1 B row is almost exactly Prism's computed optimum (159 M / 4.7 B), so **its benchmark column is the best available prior for what a good Prism submission will score.** Note ARC-c (.170) and OBQA (.158) are *below* the 0.25 chance floor — see § 4.4.

### 2.4 Is the 2-point `mup_scaling_slope` statistically sound? No — and the reason is not noise

The proposed metric is `slope = (ln L_base − ln L_wide)/(ln N_wide − ln N_base)`.

**Error source 1 (small, usually overstated): seed noise.** For a 2-point finite difference,
`SE(slope) = √2 · σ_lnL / ln(N_wide/N_base)`, with `σ_lnL = σ_L / L`.
Note that 4× *width* is ≈ **16× body params**, so the denominator is `ln 16 = 2.773`, not `ln 4`.

| σ_L (nats) | L | N ratio | SE(slope) | as % of α=0.34 |
|---|---|---|---|---|
| 0.01 | 3.2 | 16× | 0.0016 | 0.5 % |
| 0.02 | 3.2 | 16× | 0.0032 | 0.9 % |
| 0.05 | 3.2 | 16× | 0.0080 | 2.3 % |
| 0.10 | 3.2 | 16× | 0.0159 | 4.7 % |
| 0.20 | 3.2 | 16× | 0.0319 | 9.4 % |

Cross-seed loss variability from the literature: restart variance up to 3.5 % relative (Hitchhiker's); PolyPythias (45 runs, 14 M–410 M, 10 seeds) found only 2 of 50 runs beyond 2 sd ([arXiv:2503.09543](https://arxiv.org/abs/2503.09543)). So σ_L ≈ 0.01–0.08 nats is plausible for a *properly trained* run ⇒ SE(slope) ≈ 0.002–0.013. That looks acceptable.

**But the current G8 sweep does not produce a properly trained run.** Reading `g8_stability.py`: `steps = 10` (4 under tiny caps), and the score is `min` loss over those 10 steps, with a plain `AdamW` and no schedule, on a `d_model=128, n_layer=4` probe. At 10 steps the loss is still dominated by initialization and warmup and sits near `ln V`. Realistic σ there is **O(0.1–0.5 nats)**, giving SE(slope) ≈ 0.009–0.043, i.e. **3–13 % of α** — before the far larger problem below.

**Error source 2 (dominant, and irreducible by more seeds): the E-term confound.**

For `L(N) = E + A/N^α`, the *local* log-log slope is
```
d ln L / d ln N  =  −α · (A/N^α)/(E + A/N^α)  =  −α · (1 − E/L)
```
So the measured slope is **not α** — it is `α · (1 − E/L)`:

| E | L | attenuation (1 − E/L) | measured slope as % of α |
|---|---|---|---|
| 1.69 (Hoffmann) | 2.6 | 0.350 | **35 %** |
| 1.69 | 3.0 | 0.437 | **44 %** |
| 1.69 | 3.4 | 0.503 | **50 %** |
| 1.69 | 3.8 | 0.555 | **56 %** |
| **1.82 (Besiroglu re-fit)** | 2.6 | 0.300 | **30 %** |
| **1.82** | 3.0 | 0.393 | **39 %** |
| **1.82** | 3.6 | 0.494 | **49 %** |

Note that the *corrected* E (1.82 rather than 1.69) makes the confound **worse**, not better: attenuation drops to 30–49 %. Unlike the § 2.3 optimum, this table depends only on E and L — not on A or B — so it is unaffected by the amplitude/exponent coupling problem.

**Interpretation, and why this is disqualifying for a scored metric:** two architectures with *identical* α but different loss levels produce different measured slopes. An architecture that is merely **better in level** (lower L) shows a *smaller* |slope| and looks like it scales *worse*. Conversely a deliberately handicapped base point inflates the slope. Combined with the 2026 finding that conclusions **flip** depending on whether E is tied across architectures, a 2-point slope is a confound with a scaling-shaped name. It should not be scored.

**Gaming surface (all cheap for a miner):**

| Attack | Mechanism | Hardening |
|---|---|---|
| Sabotage the base point | Any choice that hurts small width more (e.g. a fixed head_dim that is proportionally large at d=128) inflates the slope | Organizer-fixed probe geometry **and** organizer-fixed per-width LR grid; report per-rung losses as telemetry so an anomalous base is visible |
| Tune only for small width | Init/scale tricks that help at 1× only | Score the **level at the top rung**, not the slope, so degrading the base point cannot help |
| Warmup/schedule exploitation | 10 steps is pure warmup; a fast-warmup recipe wins regardless of scaling | Train ≥ 300–500 steps per rung with an organizer-fixed cosine schedule |
| Vocabulary/tokenizer inflation | Loss in bits/token shrinks with vocab | Already correct in G1: score tokenizer-neutral **bits/byte**. Do the same for every rung of the ladder |
| **Warmup sandbagging** (highest-risk, invisible in review) | Porian et al. showed a **constant-step** warmup is too long for small models and inflates their loss. A fixed step count therefore sandbags the base rung mechanically, with no code that looks like cheating | **Specify warmup as a fraction of total steps, not a step count.** One-line change, closes the mechanism at its root |
| **Embedding-fraction gaming** | 4× width ≠ 4× params when embeddings dominate; inflating the embedding share shrinks the ln-N denominator | Count **body params only** (see below) |
| Non-monotone curve exploitation | Pick rungs where the curve happens to be steep | Fit over ≥ 4 rungs and publish `r²`; gate on fit quality |
| Same-code-path evasion | Different code paths for base and wide | Build both from one `build_model` call with only the width knob changed (already done via `prism_width_multiplier`) |

### 2.5 Cheapest credible upgrade: a 4-rung width ladder, scored on level + differential exponent

**Statistical gain from more rungs.** For OLS on `m` equally log-spaced rungs spanning ratio `R`, `SE(slope) = σ_lnL/√Sxx`:

| m | span | Sxx | SE(slope) @σ_L=0.02 | @σ_L=0.05 |
|---|---|---|---|---|
| 2 | 16× | 3.844 | 0.0032 | 0.0080 |
| 4 | 16× | 4.271 | 0.0030 | 0.0076 |
| **4** | **64×** | **9.609** | **0.0020** | **0.0050** |
| 5 | 64× | 10.810 | 0.0019 | 0.0048 |

**Key insight: extra interior rungs buy almost nothing in variance (~10 %).** What they buy is **curvature detection** — the ability to see that the curve is *not* a straight line, which is exactly the diagnostic that exposes the E-confound and the gaming attempts. Widening the span from 16× to 64× params buys more (SE −38 %) than adding rungs.

Minimum detectable slope gap between two submissions (2 × 1.96 × SE): **0.0125** for 2 rungs/16× at σ=0.02, **0.0079** for 4 rungs/64×. Against a plausible true architectural difference in α of 0.02–0.05, 4 rungs over 64× is adequate *for the slope's statistical error* — the E-confound remains the binding limitation.

**Proposed ladder and its cost** (4×5090, bf16, MFU 30 %, `C = 6ND`):

| Rung | d_model | n_layer | N (body) | D tokens | Wall-clock |
|---|---|---|---|---|---|
| R1 | 256 | 8 | 2.0e7 | 4.0e8 | 3.2 min |
| R2 | 384 | 8 | 4.3e7 | 4.0e8 | 6.8 min |
| R3 | 512 | 8 | 7.4e7 | 4.0e8 | 11.8 min |
| R4 | 768 | 8 | 1.6e8 | 4.0e8 | 25.5 min |
| **Total, 1 LR/rung** | | | | | **47.3 min** |

**The embedding trap — count body params only.** `N` in the table above is **body** parameters. This matters more than it looks. With V = 32768, L = 8, ffn = 4×:

| d_model | body | embeddings (tied) | total | embedding share |
|---|---|---|---|---|
| 256 | 6.29e6 | 8.39e6 | 1.47e7 | **57.1 %** |
| 384 | 1.42e7 | 1.26e7 | 2.67e7 | 47.1 % |
| 512 | 2.52e7 | 1.68e7 | 4.19e7 | 40.0 % |
| 768 | 5.66e7 | 2.52e7 | 8.18e7 | 30.8 % |

At the smallest rung, embeddings are **the majority of parameters**. Over the d = 256 → 768 ladder, the param ratio is **9.00×** on body but only **5.57×** on total, so `ln N_ratio` drops from 2.197 to 1.718 — **using total params inflates the measured |slope| by ~28 %, and makes vocabulary size a slope lever.** Prism's `count_params.py` / `MAX_PARAMS` semantics count *all* params (correctly, for a cap), so the ladder must use a separate body-only count. Also note `n_params` dedupes tied embeddings per `PRISM.md`, which further changes the arithmetic — worth asserting explicitly in the implementation.

At D = 1.5e8 tokens/rung the total drops to **17.7 min** (1 LR) or **53 min** (3 LRs). Recommended operating point: **D = 2.5e8, 1 organizer-fixed LR per rung from a µP-scaled prescription, 4 rungs ⇒ ~30 min**, plus one repeated smallest rung for a seed-noise estimate (+3 min). That is **~0.55 h**, i.e. **9 % of the 6 h budget** — affordable *only if* the eval battery is rebalanced (§ 5.3).

**Three ladder designs and what each licenses:**

| Design | Cost | Statistical claim it supports | Claim it does NOT support |
|---|---|---|---|
| (a) Width ladder, fixed D | ~30 min | Local exponent in N at fixed data; LR-transfer quality; monotonicity/fit quality | α itself (E-confounded); anything about data scaling |
| (b) 3–4 point IsoFLOP mini-profile | ~48 min | Existence and location of a loss-vs-N minimum at fixed C — the most decision-relevant shape | E, A, B; cross-OOM extrapolation |
| (c) Fixed-N token ladder | free (reuse probe curve) | Data-efficiency ordering, β-like local slope in D | N-scaling |

**(c) is free** — it is the existing G6 probe curve, which already samples loss vs tokens. Fix G6 (§ 4.1) before adding anything new.

### 2.6 Honest ledger: what is and is not demonstrable

**NOT demonstrable at 6 h / 4×5090:**
- The irreducible-loss term **E** (unstable even with 128 runs per scale in the 2026 study).
- Absolute α and β (needs ≥ 64 HP configs/scale for accuracy; Prism affords ~1).
- **Cross-order-of-magnitude extrapolation.** Farseer needed ~1000 LLMs / 3 M H100-hours for 0.5 % error at >1 OOM. Prism has ~1e-6 of that.
- Emergent-ability claims — and they are contested as measurement artifacts anyway (Schaeffer et al., which the harness already cites as a design rule in `common.py`).
- Any downstream-benchmark scaling law (39 % fit rate).

**IS demonstrable:**
- **Loss level at fixed budget** — the cleanest, least gameable architecture ranking available. This is what G1 bits/byte already does.
- **Ordering of local exponents *between* architectures on an organizer-fixed ladder**, provided the level is reported alongside so the E-confound is visible.
- **LR-transfer quality** (µP), if trained long enough to be meaningful (not 10 steps).
- **Data-efficiency ordering** from the learning curve (G6, once fixed).
- **Fit quality / monotonicity** — a cheap, strong cheat-detector.
- **Inference-cost Pareto** (G7), which is where looped architectures will separate.

**The strongest available justification for Prism's whole design.** The load-bearing assumption behind ranking architectures by loss is that "better loss ⇒ better capability". Recent work (Mayilvahanan et al. 2025, as surveyed by Lourie et al.) finds this perplexity→capability correspondence **survives changes in scale, hyperparameters, architecture, and even the tokenizer — but breaks when the pretraining data changes.** Prism pins the fineweb-edu shard, which is *exactly* the condition under which loss-ranking is legitimate. This is a strong argument for keeping G1 bits/byte as the dominant weight, and for resisting any proposal to let miners choose their own data.

---

## 3. Question 2 — Automated diagnostics: does this model generalize, what are its defects?

Ground rules I applied: no human, no LLM judge on the scored path, must run in minutes on a ≤1 B checkpoint inside the harness, and must be hard to game. I classify every diagnostic as **must add**, **nice to have**, or **trap — do not score**.

### 3.1 Generalization vs memorization

| Diagnostic | Defect detected | Cost | Gameable? | Verdict |
|---|---|---|---|---|
| Per-domain bits/byte, tokenizer-neutral | Narrow fit; tokenizer gaming | already in G1, ~2–5 min | Low — bits/byte is the right invariant | **Already correct. Keep.** |
| Fresh-crawl bits/byte (post-cutoff text) | Contamination of the pinned shard | ~1 min | Very low | **Already present** (`g1.bits_per_byte.fresh`) — **raise its weight** |
| Train/val/held-out gap | Overfit to the shard | ~1 min | Low | **Must add** as observed: `org.g1.bits_per_byte_val_train_gap` |
| Public/private mirror gap | Benchmark contamination | already in `rollup.build_mirrors` | Low | **Already correct**, but see § 4.1 bug 3 |
| n-gram (13-gram) overlap vs eval assets | Direct leakage | ~30 s CPU | Low | **Nice to have**, observed-only |
| Verbatim-continuation / extraction probe | Rote memorization | ~1–2 min | Medium | **Nice to have**, observed-only |
| Membership-inference (Min-K %, Min-K %++, zlib ratio) | Training-set membership | ~2 min | — | **Trap — do not score.** Duan et al. 2024 ([arXiv:2402.07841](https://arxiv.org/abs/2402.07841)) show MIA on LLMs performs **near chance** and that apparent success is usually a distribution shift artifact between members and non-members. |

**Note on the mirror gap:** the design is sound but in the `public_dev` tier the run is **its own mirror** (gap ≡ 0 by construction — `rollup.py` copies the public series). The code labels this honestly, but it means the anti-contamination penalty is **inert unless the private tier is staged.** Operators must not read "mirror penalty 0" as evidence of no contamination.

### 3.2 Calibration and uncertainty

| Diagnostic | Defect | Cost | Verdict |
|---|---|---|---|
| NLL / gold-answer nll on MCQ | Miscalibrated likelihood | free (`score_choices` already returns it) | **Must add as observed**: `org.g2.mean_gold_nll` |
| Brier score on MCQ | Calibration, smooth at low accuracy | ~free from existing logprobs | **Must add as observed** — smooth where accuracy is at chance |
| Predictive entropy distribution | Collapse to a constant answer / degenerate confidence | ~free | **Must add as observed** |
| ECE | Calibration | ~free | **Nice to have, observed only.** ECE is **biased by binning** and not comparable across models with different confidence spreads; use adaptive binning and never score it. |
| Temperature sensitivity | Sharpness pathology | ~1 min | Nice to have |

**Why this matters at Prism's scale:** when accuracy is pinned at chance (§ 3.4), **NLL and Brier still move.** They are the correct high-resolution substitutes for accuracy on tasks whose accuracy is uninformative — this is the single highest-value cheap addition in this section.

### 3.3 Robustness / invariance

| Diagnostic | Defect | Cost | Gameable? | Verdict |
|---|---|---|---|---|
| MCQ choice-order / cyclic-permutation consistency | Position bias, not comprehension | ~k× the MCQ cost (k = #permutations) | Low | **Must add as observed**: `org.g2.choice_order_consistency`. Zheng et al., *LLMs Are Not Robust Multiple Choice Selectors* ([arXiv:2309.03882](https://arxiv.org/abs/2309.03882)) show large selection bias toward specific option positions |
| Prompt-format sensitivity | Brittleness to surface form | ~2–3× MCQ cost | Low | **Nice to have.** Sclar et al., FormatSpread ([arXiv:2310.11324](https://arxiv.org/abs/2310.11324)) report accuracy spreads up to ~76 points across semantically equivalent formats |
| Paraphrase invariance | Memorized surface patterns | ~2× | Low | Nice to have |
| Distractor sensitivity | Shallow heuristics | ~2× | Low | Nice to have |
| Length/position generalization beyond train context | RoPE/positional failure | already in G5 (`lstar`) | Low | **Already present. Keep.** |

**Position-bias caution specific to Prism:** the harness scores `acc_norm` as `sum_logprob / len(choice_in_characters)` (`common.score_choices_detail`). Character-length normalization is a defensible choice (it is what makes metrics smooth at 100–350 M, per the module docstring), but it is **not** the same as lm-eval's `acc_norm` (which normalizes by byte length) nor `acc`. **Prism numbers are therefore not directly comparable to published Pythia/Cerebras numbers** — a subtle trap when reading § 4.4's ranges. Document this explicitly in miner docs.

### 3.4 Representation quality / internal health

| Diagnostic | Defect | Cost | Verdict |
|---|---|---|---|
| Loss-spike count, grad-norm history, NaN fraction | Instability | free (telemetry) | **Already in G8. Keep.** |
| Attention-entropy collapse | Training pathology | ~1 min | **Nice to have.** Zhai et al. ([arXiv:2303.06296](https://arxiv.org/abs/2303.06296)) link entropy collapse to instability |
| Effective rank / RankMe of hidden states | Representation collapse | ~1–2 min | **Must add as observed**: `org.diag.effective_rank_mid`. Cheap, architecture-agnostic, and a genuine collapse detector |
| Massive activations / outlier features | Quantization fragility, attention-sink reliance | ~1 min | Nice to have (Sun et al., [arXiv:2402.17762](https://arxiv.org/abs/2402.17762)) |
| Dead/saturated unit fraction | Wasted capacity | ~1 min | Nice to have |
| Layer-wise linear probe quality | Where representations degrade | ~3–5 min | Nice to have, observed-only |
| Tokenizer pathologies (unreachable/glitch tokens, fertility, bytes-per-token) | Vocab gaming, broken tokenizer | ~30 s CPU | **Must add as observed**: `org.diag.tokenizer_fertility`, `org.diag.unreachable_token_frac` |
| **WeightWatcher / heavy-tailed power-law α** | claimed quality predictor | ~2 min | **Trap — do not score.** The HT-SR α claim is contested, sensitive to fitting choices, and has no reliable validation at 100 M–1 B scale. Observed-only at most. |

### 3.5 Data efficiency and stability (Prism's G6/G8)

Both groups have the right *intent* and broken or weak *implementation*. See § 4.1 for the G6 anchor bugs and § 2.4 for the G8 sweep being 10 steps of warmup. Concretely:

- **G6 must move to bits/byte, not CE**, so the AUC and threshold metrics are tokenizer-neutral like G1 already is. As written, `g6.tokens_to_ce4.0` is a **per-token CE** threshold, so a large-vocab tokenizer reaches "CE 4.0" on fewer tokens for free.
- **The G6 x-axis is miner-controlled.** `probes.py` fires every `PRISM_PROBE_EVERY`-th *telemetry report* (default 25), and the report cadence is the miner's `training.py` choice. A miner who reports rarely early and often late shrinks the log-token span and lowers the mean-loss AUC. Fix by sampling probes on **organizer-chosen token milestones** (e.g. fixed decades: 1e7, 3e7, 1e8, 3e8, 1e9), not on report counts.
- **Gradient-noise scale** (McCandlish et al., [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)) is theoretically attractive for batch-size diagnosis but expensive and noisy; **nice to have, observed-only.**

### 3.6 Would these diagnostics distinguish a looped / recurrent-depth architecture?

**Expected profile of weight-tied recurrent depth** (Universal Transformer; *Looped Transformers are Better at Learning Learning Algorithms*; Geiping et al. 2025 recurrent-depth / Huginn-3.5B, [arXiv:2502.05171](https://arxiv.org/abs/2502.05171); Mixture-of-Recursions 2025):

| Axis | Expected effect vs plain transformer | Prism group |
|---|---|---|
| **Params at fixed quality** | **Better** — weight tying is the core parameter-efficiency claim | G1 (under a param cap, this is where looping should win) |
| **FLOPs/token at fixed quality** | **Worse** — r loop iterations cost r× compute for one set of weights | Implicitly G1 under the wall-clock cap; explicitly G7 |
| **Algorithmic / recall tasks** | **Better** — the strongest, most reproducible claim in the looped literature | **G3, G4** |
| **Length generalization** | **Better** | G5 `lstar` |
| **Inference latency (TTFT/TPOT)** | **Worse** (r× depth serially) | **G7 — will penalize looping** |
| **KV/state per token** | Neutral to better | G7 `state_bytes_per_token` |
| **Loss-vs-N local exponent** | Genuinely unclear — **evidence is thin.** No reliable published measurement of α for looped vs plain at 100 M–1 B under matched compute | proposed ladder |

**The critical confound for Prism's A/B:** the parameter cap rewards looping (tied weights → more effective depth per parameter), while the **wall-clock cap punishes it** (r× FLOPs per token → fewer tokens in 6 h). These pull in opposite directions, so **the A/B result will be determined by which cap binds.** Under the numbers in § 2.3, at a 350 M cap and 5 h the binding constraint is *compute*, not parameters — which **disadvantages looping**. A fair A/B must therefore report both matched-params and matched-FLOPs comparisons, or the answer is an artifact of the cap choice.

---

## 4. Question 3 — Concrete proposal for Prism

### 4.1 Live bugs found in the current scoring path (fix before any `composite` flip)

These are on the code actually present in `/root/gbase`. All three change rankings.

**Bug 1 — `org.g6.auc_log_tokens` is inverted and inert. (Critical.)**

`anchors/v0.json` declares `{"kind": "efficiency_log_ratio", "reference": 0.5, "cap": 0.95}` with the note *"higher-better"*. But `g6_curve.py` computes `g6.auc.log_tokens` as the trapezoid integral of **probe loss** over log10(tokens), divided by the log span — i.e. a **mean cross-entropy**, which is **lower-better** and whose plausible range is **3.0–5.0 nats**.

Normalization is `clip01(ln(x/reference)/ln(cap/reference))`. With reference 0.5 and cap 0.95:

| measured AUC | normalized |
|---|---|
| 0.4 | 0.000 |
| 0.95 | 1.000 |
| 3.0 | **1.000** |
| 5.0 | **1.000** |

**Every plausible submission scores exactly 1.0.** The metric is (a) direction-inverted relative to its own note and (b) fully saturated, so it contributes nothing and silently inflates G6. Since G6 has only two metrics, this means **half of G6 is a constant**.

*Fix:* re-anchor as lower-better in **bits/byte**, e.g. `{"kind": "efficiency_log_ratio", "reference": 1.30, "cap": 0.95}` (cap < reference encodes lower-better, per `composite.rs`), after measuring both baselines. Requires an anchor-version bump.

**Bug 2 — `org.g6.tokens_to_threshold` rewards censored (failed) runs. (Critical.)**

`g6_curve.py` returns `(last_tokens_seen, censored=True)` when the loss never reaches CE 4.0, and separately emits `g6.tokens_to_ce4.0.censored`. But `rollup.py` maps `org.g6.tokens_to_threshold → g6.tokens_to_ce4.0` **unconditionally and ignores the censored flag.** With `reference=2e9, cap=5e8`:

| `tokens_to_ce4.0` | normalized |
|---|---|
| 1e8 (censored — never reached the threshold) | **1.000** |
| 3e8 | 1.000 |
| 1e9 | 0.500 |
| 2e9 | 0.000 |

A model that trains briefly and **never reaches CE 4.0** reports a small `tokens_seen` and receives a **perfect** sample-efficiency score, beating a genuinely efficient model that reached the threshold at 6e8 tokens. This is directly exploitable: train fewer tokens, score higher.

*Fix:* when `censored` is true, emit the fail-closed worst value (or omit the key so the completeness gate fires) — mirroring the pattern already used correctly for `org.g8.mup_lr_stability`, which deliberately emits 0.0 rather than omitting after a real sweep.

**Bug 3 — G1 and G2 collapse to a single bootstrap cluster, voiding the CI gate. (Critical, systemic.)**

The clustered bootstrap in `composite.rs` resamples `series.clusters` with replacement. Resampling a single value always returns that value ⇒ **zero variance**.

- `g2_downstream.py` records every item with the **constant** cluster `f"g2/{task}"`.
- `g1_intrinsic.py` records every doc with the constant `tag` for that call (`"val"`, `f"domain/{name}"`, `"fresh"`).

So each G1/G2 metric has exactly **one** cluster. G3/G4 (per-item `it["cluster"]`) and G5 (`f"{probe}@{length}"`, `f"qa{qa}@{length}"`) do have real cluster variety.

Consequences: **G1 (0.25) + G2 (0.15) = 40 % of composite weight contributes no bootstrap variance.** `SE(C)` is understated, the LCB `C − 1.645·SE` is too high, and the `ci_half_width_delta = 0.05` gate is **vacuous** on those two axes — precisely the axes carrying the most weight. Interestingly `rollup.build_mirrors` **does** use per-row clusters (`f"{tag}#{i}"`), so the mirror path is correct while the main path is not — good evidence this is an oversight rather than a design choice.

*Fix:* use per-item cluster ids in G1/G2 (e.g. `f"g2/{task}#{i}"`, `f"domain/{name}#{i}"`), matching the mirror path. This is a one-line change per site and **no anchor bump is needed** — but it will *lower* scores by revealing real variance, so it must land before anchors are calibrated, or calibration will bake in the bug.

**Bug 4 — budget over-subscription. (Operational, not a ranking bug.)**

Summing the per-group defaults in the battery: G1 1800 + G2 1800 + G3 1800 + G4 1800 + G5 (ruler 1200 + babilong 900 + natural 900 + longctx 3600) + G7 2400 + G8 sweep 300 + mirror 600 ≈ **17 100 s ≈ 4.75 h** of *ceilings*, against `PRISM_EVAL_TIMEOUT_S = 3 h`. And `TRAIN_HOURS_CAP (6 h) + EVAL_TIMEOUT_S (3 h) + build/checkpoint` can reach **~9.3 h** against `POD_LIFETIME_HOURS_CAP = 7 h`. The per-group budgets are independent ceilings, so in practice the battery truncates (`g*.partial` flags) rather than overrunning — but truncation is *silent partial scoring*, which distorts comparisons between submissions. **Any scaling-ladder addition must come with an explicit global eval budget.**

### 4.2 Proposed metrics table

Norm kinds are from the anchor schema in `anchors.rs`: `accuracy`, `bpb_log_ratio`, `efficiency_log_ratio`, `stability_bounded`. Anchor values are **placeholders to be measured on the two E6 baselines**, exactly as v0 requires.

| `org.*` metric | Group | Norm kind | chance / ref / cap (placeholder) | Harness file | Cost | Zone |
|---|---|---|---|---|---|---|
| `org.g8.ladder_slope` | G8 | `stability_bounded` (after mapping) | map \|slope\| into [0,1] vs ref slope 0.15 | new `eval/g8_ladder.py` | ~30 min | **B / observed first** |
| `org.g8.ladder_fit_r2` | G8 | `stability_bounded` | ref 0.90 | `eval/g8_ladder.py` | free w/ ladder | **A, v3** |
| `org.g8.ladder_top_bpb` | G8→G1 | `bpb_log_ratio` | chance 3.6 / ref TBD | `eval/g8_ladder.py` | free w/ ladder | **A, v3** |
| `org.g8.mup_lr_stability` | G8 | `stability_bounded` | existing | `eval/g8_stability.py` | ~5 min (raise to ≥300 steps) | **A (exists)** |
| `org.g6.auc_log_bytes` | G6 | `efficiency_log_ratio` | ref 1.30 / cap 0.95 (lower-better) | `eval/g6_curve.py` | free | **A, v3 (replaces buggy key)** |
| `org.g6.bytes_to_threshold` | G6 | `efficiency_log_ratio` | ref 2e9 / cap 5e8 + censor fail-closed | `eval/g6_curve.py` | free | **A, v3** |
| `org.g1.bits_per_byte_val_train_gap` | G1 | `efficiency_log_ratio` | ref 0.15 / cap 0.02 | `eval/g1_intrinsic.py` | ~1 min | **B → A later** |
| `org.g2.mean_gold_nll` | G2 | `efficiency_log_ratio` | ref 4.0 / cap 2.5 (lower-better) | `eval/g2_downstream.py` | free | **A, v3** |
| `org.g2.brier` | G2 | `efficiency_log_ratio` | ref 0.75 / cap 0.55 | `eval/g2_downstream.py` | free | **B first** |
| `org.g2.choice_order_consistency` | G2 | `accuracy` (chance 0.25) | — | `eval/g2_downstream.py` | ~2× G2 | **B first** |
| `org.diag.effective_rank_mid` | new G9 or G8 | `efficiency_log_ratio` | ref 0.25 / cap 0.75 (frac of d_model) | new `eval/g9_health.py` | ~2 min | **B** |
| `org.diag.tokenizer_fertility` | G1 | observed | — | `eval/toklen.py` (exists) | ~30 s | **B** |
| `org.diag.unreachable_token_frac` | G1 | observed | — | `eval/toklen.py` | ~30 s | **B** |
| `org.diag.attn_entropy_min` | G8 | `stability_bounded` | — | `eval/g9_health.py` | ~1 min | **B** |

**Anchor-version guidance:** anything in the **A** column needs a **v3 anchor set** (new keys, new normalization, and the G6 re-anchoring), pre-registered and hash-committed per `anchors.rs`. Everything in the **B** column can ship **today** as Zone B / observed telemetry with no governance action — and *should*, so that v3 anchors can be calibrated on real measured distributions instead of guesses. Note Zone B is `miner.*`-namespaced and participant-reported by contract; organizer-measured-but-unscored metrics should therefore be emitted as `org.*` keys **absent from the anchor set** (the composite already ignores undeclared keys — see the `unknown_metrics_are_ignored` test), rather than as Zone B.

### 4.3 Recommendation for the looped-vs-transformer A/B

**Metrics where I expect the difference to show, ranked by expected signal-to-noise:**

1. **`org.g1.bits_per_byte_*` (G1, weight 0.25)** — the primary signal. Highest weight, lowest variance, tokenizer-neutral, hardest to game. If looping helps at a fixed param cap, it shows here first.
2. **G3 recall + G4 reasoning (0.10 + 0.15)** — the strongest published looped-transformer claim (algorithmic/recall tasks). Procedural and memorization-proof, so a real difference is credible.
3. **G7 inference efficiency (0.075)** — expect looping to **lose** here (r× serial depth → worse TTFT/TPOT). Report it prominently; a "win" that ignores latency is not a win.
4. **G6 learning-curve shape (0.075)** — informative *after* the bugs are fixed; meaningless before.
5. **G5 `lstar` (0.15)** — plausible looped advantage in length generalization.
6. **G2 (0.15) — do not use for this A/B.** See below.

**Practical protocol:** run the A/B at **matched FLOPs** *and* **matched params**, and report both. As shown in § 3.6, the param cap favours looping and the wall-clock cap punishes it, so a single-condition A/B measures the cap, not the architecture.

### 4.4 Expected G2 accuracy ranges — so noise is not misread as signal

**Prism's current cap is 200 items/task** (`eval_asset_cap(200, 8, env_key="PRISM_EVAL_G2_CAP")`), not the full validation sets. That dominates the noise floor.

**These are no longer hand-estimated.** Cerebras-GPT publishes **training FLOPs** alongside its downstream table, and Cerebras-111M was trained at **C = 2.6e18** — essentially identical to Prism's own budget (3.0–5.3e18). So Prism's expected G2 is a **log-FLOP interpolation between Cerebras-111M (2.6e18) and Cerebras-256M (1.3e19)**, which is a genuinely FLOP-matched anchor rather than an extrapolation. Numbers verified against both [the paper](https://ar5iv.labs.arxiv.org/html/2304.03208) and the HF model cards. Chance floors are from `anchors/v0.json`.

At Prism's central budget (C = 4.53e18, i.e. 34.5 % of the way from 111M to 256M in log-FLOPs), the implied Pile-style cross-entropy is **≈ 2.47 nats**, and:

| Task | Chance | FLOP-matched expected `acc` | Margin over chance | SE @n=200 | MDD @n=200 (2 subs, 95 %) | Verdict |
|---|---|---|---|---|---|---|
| LAMBADA | 0.00 | **0.20–0.24** | +0.228 | 2.97 pp | 8.2 pp | **usable (best G2 signal)** |
| ARC-easy | 0.25 | **0.38–0.39** | +0.140 | 3.45 pp | 9.6 pp | **usable** |
| PIQA | 0.50 | **0.60** | +0.101 | 3.46 pp | 9.6 pp | **usable** |
| HellaSwag | 0.25 | **0.27** | +0.020 | 3.14 pp | 8.7 pp | **NOISE — margin 2 pp vs MDD 8.7 pp** |
| Winogrande | 0.50 | **0.49–0.50** | −0.004 | 3.54 pp | 9.8 pp | **DEAD — at/below chance** |
| ARC-challenge | 0.25 | **0.17** (`acc`) | −0.083 | 2.64 pp | 7.3 pp | **DEAD — below chance** |
| OpenBookQA | 0.25 | **0.13** (`acc`) | −0.118 | 2.39 pp | 6.6 pp | **DEAD — below chance** |
| BoolQ | 0.62 (majority) | **0.45–0.62** | −0.085 | 3.53 pp | 9.8 pp | **DEAD — below majority** (not in the Cerebras suite; estimate retained) |

**Normalization caveat, and it cuts in a specific direction.** Cerebras reports lm-eval `acc`; Prism scores **character-normalized `acc_norm`**, which typically *raises* small-model MCQ scores by removing length bias (e.g. Pythia-160M OBQA `acc` ≈ 0.18 vs `acc_norm` ≈ 0.28). So Prism's ARC-c and OBQA will land **nearer chance (~0.20–0.30) rather than far below it**. That changes the sign of the gap but not the conclusion: both remain statistically indistinguishable from chance.

**The consequence nobody has priced in.** G2 uses an equal-weight arithmetic mean of `accuracy`-normalized terms, and `accuracy` normalization is `clip01((x − chance)/(1 − chance))`. For any submission at this scale:

| Task | expected `acc` | normalized contribution |
|---|---|---|
| ARC-challenge | 0.167 | **0.000** |
| OpenBookQA | 0.132 | **0.000** |
| Winogrande | 0.496 | **0.000** |

**Three of eight G2 terms normalize to ~0 for *every* submission**, good and bad alike. They are constant dead weight, which caps the achievable G2 point estimate near **5/8 = 0.625** for the entire field while contributing zero discriminative power. Under `scoring_version 4` (equal-weight mean of *available* accuracies as the live leaf) the same three tasks dilute every submission's score identically. This is a pure loss of dynamic range.

**Items needed for a task to separate two submissions at all** (n such that MDD < margin):

| Task | Required n | Full set size | Reachable? |
|---|---|---|---|
| LAMBADA | ~30 | 5153 | yes |
| ARC-easy | ~90 | 2376 | yes |
| PIQA | ~140 | 1838 | yes |
| HellaSwag | ~3 800 (at the 2 pp FLOP-matched margin) | 10042 | yes — **but needs ~19× the current cap** |
| Winogrande | ≫ set size | 1267 | **no** |
| OpenBookQA | ≫ set size | 500 | **no** |
| ARC-challenge | ∞ (expected ≤ chance) | 1172 | **no** |
| BoolQ | ∞ (expected ≤ majority) | 3270 | **no** |

Note HellaSwag got materially *worse* under the FLOP-matched anchor: my earlier hand-estimate assumed a 4.5 pp margin (needing ~790 items), but the interpolation gives only **2.0 pp**, which needs ~3 800 items — nearly the whole validation set. HellaSwag is effectively **not usable** at Prism's budget either, which was not obvious before FLOP-matching.

**Actionable conclusions for the A/B:**

1. **Raise `PRISM_EVAL_G2_CAP` from 200 to ≥ 1000** for LAMBADA, PIQA and ARC-easy — the three tasks with real margin. HellaSwag needs ~3 800 items to be worth scoring; either give it the full set or treat it as observed-only.
2. **Drop or de-weight Winogrande, BoolQ, ARC-challenge and OpenBookQA** at this scale. They are structurally incapable of separating two 6 h submissions, and three of them normalize to a **constant 0** for the entire field (§ above) — pure dead weight in both the v3 composite and the live v4 leaf.
3. **A 2–3 pp delta on any G2 task is noise at n=200.** Require ≥ 9 pp before treating a G2 difference as real, or raise n. Note the FLOP-matched HellaSwag margin (2.0 pp) is *itself* below the n=200 MDD (8.7 pp) — so a HellaSwag "win" at the current cap is almost certainly noise.
4. **Prefer `mean_gold_nll` / Brier over accuracy** on the near-chance tasks: they move continuously where accuracy is pinned. This is the cheapest way to recover discriminative power from benchmarks that are otherwise dead at this scale.
5. Remember Prism's `acc_norm` is **character**-normalized, not byte-normalized like lm-eval, so expect systematic offsets vs published tables (§ 3.3).

---

## 5. Prioritized plan

### 5.1 Order of work (highest value first)

| # | Action | Why | Anchor bump? | Effort |
|---|---|---|---|---|
| 1 | Fix G1/G2 bootstrap clustering (per-item cluster ids) | 40 % of composite weight currently has zero variance; LCB and CI gate are wrong | No | ~1 h |
| 2 | Fix `org.g6.tokens_to_threshold` censoring | Directly exploitable: train less, score higher | No (logic fix) | ~1 h |
| 3 | Re-anchor `org.g6.auc_log_tokens` (inverted + saturated) → bits/byte | Half of G6 is a constant 1.0 today | **Yes (v3)** | ~2 h + baseline runs |
| 4 | Add global eval budget; reconcile 4.75 h of ceilings vs 3 h timeout vs 7 h pod | Silent partial scoring distorts comparisons | No | ~2 h |
| 5 | Raise `PRISM_EVAL_G2_CAP` to ≥1000 for LAMBADA/ARC-e/PIQA; de-weight or drop Winogrande / ARC-c / OBQA / BoolQ | 3 of 8 G2 terms normalize to a **constant 0** for the whole field; HellaSwag's real margin (2 pp) is below its own n=200 MDD (8.7 pp) | Yes for weights | ~1 h |
| 6 | Emit calibration/robustness telemetry (`mean_gold_nll`, Brier, choice-order consistency, effective rank, tokenizer fertility) as unscored `org.*` | Enables v3 anchors to be calibrated on real distributions | No | ~1 day |
| 6b | **Make warmup a fraction of total steps, not a step count**; assert Triton/PyTorch version in the manifest | Closes the Porian sandbagging mechanism; removes a ~17 % free throughput edge | No | **~1 h — best value/effort in the list** |
| 7 | Lengthen the µP sweep from 10 steps to ≥300 with an organizer-fixed schedule | 10 steps measures warmup, not transfer | No | ~0.5 day |
| 7b | Count **body-only** params for any ladder slope; hold depth fixed, vary width only | Total-param counting inflates slope ~28 % and makes vocab a lever; µP is known not to transfer across depth | No | ~2 h |
| 8 | Implement the 4-rung ladder as observed-only telemetry | Builds the evidence base to decide whether to score it | No | ~2 days |
| 9 | Only after (8) has data on both baselines: consider scoring `ladder_fit_r2` + `ladder_top_bpb` | Level and fit quality are defensible; raw slope is not | Yes (v3) | later |
| 10 | Publish the § 2.3 optimum table in miner docs; reconsider the 1 B cap | 1 B costs +0.23 nats; the cap change is an unforced error | No | ~1 h |

### 5.2 What I would NOT do

- **Do not score a 2-point (or even 4-point) scaling slope.** The E-confound makes it a level-in-disguise metric; the 2026 literature shows conclusions flip on how E is handled. Score the **level at the top rung** and the **fit quality**; publish the slope as telemetry.
- **Do not add membership-inference scoring.** Near-chance on LLMs per Duan et al.; false confidence is worse than no signal.
- **Do not score ECE or WeightWatcher α.** Binning-biased and contested respectively.
- **Do not raise the param cap to 1 B expecting better models** at this compute. It is a ~0.2 nat regression unless compute grows ~30×.
- **Do not treat mirror-gap = 0 as evidence of no contamination** in the `public_dev` tier — it is zero by construction.

### 5.3 Suggested budget rebalance (6 h wall clock)

| Phase | Now (ceilings) | Proposed |
|---|---|---|
| Train | up to 6.0 h | **4.0 h** |
| Eval battery G1–G7 | ~4.75 h of ceilings / 3 h timeout | **1.2 h (global budget, enforced)** |
| µP sweep + 4-rung ladder (G8) | 300 s | **0.55 h** |
| Slack / checkpoint / staging | — | **0.25 h** |
| **Total** | up to ~9.3 h (> 7 h pod cap) | **6.0 h** |

At 4.0 h of training and MFU 30 %, C ≈ 3.6e18 ⇒ optimum shifts slightly to N ≈ 145 M, D ≈ 4.1 B. The conclusion in § 2.3 is unchanged: **the optimum stays far below both the 350 M and 1 B caps.**

### 5.4 Risks and where my analysis is weakest

- **MFU is assumed, not measured.** Every FLOP/token number scales linearly with it. If real MFU on 4×5090 with no FA3 is 15 %, all token budgets halve and the optimum drops to ~120 M. **Measure this first.**
- **Chinchilla constants are borrowed.** They were fitted on a different tokenizer and corpus, and the fit itself was shown non-reproducible. The *shape* of the conclusion (optimum ≈ 150–200 M, 1 B is worse) is robust to reasonable perturbation of E/A/B; the exact loss values are not.
- **Expected G2 ranges are now FLOP-matched** (Cerebras-111M at C=2.6e18 vs Prism's 3.0–5.3e18), which is the strongest part of § 4.4. The residual uncertainty is the `acc` → character-normalized `acc_norm` conversion, which I could not resolve from published tables — it shifts ARC-c/OBQA upward toward chance without changing the "cannot discriminate" conclusion. **Treat absolute values as ±3 pp; treat the which-tasks-are-dead ordering as reliable.**
- **§ 3 (the diagnostics battery) rests on lighter evidence than § 2 and § 4.** The workstream researching it completed its searches but failed to return its written report, so § 3's citations are ones I verified individually while its recovered notes supplied the Cerebras table and the MDD arithmetic (both since independently re-verified). The diagnostic *classifications* (must-add / nice-to-have / trap) are my judgement calls informed by that literature, not consensus findings. The three "trap" verdicts (MIA, ECE, WeightWatcher α) are the ones I hold most confidently; the cost estimates for the representation-health probes are the softest numbers in the report and should be measured before scheduling them into a budget.
- **The v2/v2.1 gap (§ 1.1).** If a v2 anchor set exists in the rebasing worktree, bugs 1–3 must be re-verified against it. The normalization logic in `composite.rs` is shared and unchanged, so the *mechanism* of each bug holds regardless; only the specific anchor numbers could differ.
- **Looped-vs-plain scaling exponents: evidence is genuinely thin.** I found no reliable published measurement of α for recurrent-depth vs plain transformers at 100 M–1 B under matched compute. Anyone claiming a scaling-exponent advantage either way at this scale is over-reading their data — including us, if we score a slope.

---

## Sources

**Scaling-law methodology**
- Kaplan et al., *Scaling Laws for Neural Language Models* — https://arxiv.org/abs/2001.08361
- Hoffmann et al., *Training Compute-Optimal LLMs* (Chinchilla) — https://arxiv.org/abs/2203.15556
- Besiroglu et al., *Chinchilla Scaling: A Replication Attempt* — https://arxiv.org/abs/2404.10102
- Porian et al., *Resolving Discrepancies in Compute-Optimal Scaling* — https://arxiv.org/abs/2406.19146
- Choshen et al., *A Hitchhiker's Guide to Scaling Law Estimation* (ICML 2025) — https://arxiv.org/abs/2410.11840
- *Farseer: A Refined Scaling Law* (NeurIPS 2025) — https://arxiv.org/abs/2506.10972
- *Predictable Scale I / Step Law* — https://arxiv.org/abs/2503.04715
- *Small-Scale Experiments: Are We There Yet?* (2026) — https://arxiv.org/abs/2608.11859
- *Spend Less, Fit Better* (2026) — https://arxiv.org/abs/2604.22753
- *Unraveling the Mystery of Scaling Laws, Part I* — https://arxiv.org/abs/2403.06563
- Ruan et al., *Observational Scaling Laws* — https://arxiv.org/abs/2405.10938
- Ivgi et al., *Scaling Laws Under the Microscope* — https://aclanthology.org/2022.findings-emnlp.544/
- *Scaling Laws Are Unreliable for Downstream Tasks* — https://arxiv.org/abs/2507.00885
- Arnal et al., *Scaling Laws with Hidden Structure* — https://arxiv.org/abs/2411.01375
- *Scaling Law with Learning Rate Annealing* — https://arxiv.org/abs/2408.11029

**Parameterization / µP**
- Yang & Hu et al., *Tensor Programs V* (µTransfer) — https://arxiv.org/abs/2203.03466
- *Tensor Programs VI* (depth-µP) — https://arxiv.org/abs/2310.02244
- Bordelon et al., *Depthwise Hyperparameter Transfer* — https://arxiv.org/abs/2309.16620
- Everett et al., *Scaling Exponents Across Parameterizations and Optimizers* — https://arxiv.org/abs/2407.05872
- *u-µP: The Unit-Scaled Maximal Update Parametrization* — https://arxiv.org/abs/2407.17465

**Seed / run-to-run variance**
- *PolyPythias: Stability and Outliers across 50 LM Pre-Training Runs* — https://arxiv.org/abs/2503.09543
- Epoch AI, *Chinchilla Scaling: A Replication Attempt* (analysis) — https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt

**Loss-to-capability correspondence**
- Mayilvahanan et al. 2025 on perplexity→capability robustness (surveyed in [arXiv:2608.11859](https://arxiv.org/abs/2608.11859)) — correspondence survives scale/HP/architecture/tokenizer changes but **breaks on data changes**
- Bits-per-byte definition and pitfalls — https://dipkumar.dev/posts/llm/bits-per-byte/

**Hardware / systems (sm_120)**
- NVIDIA RTX 5090 product page — https://www.nvidia.com/en-gb/geforce/graphics-cards/50-series/rtx-5090/
- RTX 5090 peak BF16 tensor TFLOPS (NVIDIA devforum) — https://forums.developer.nvidia.com/t/rtx-5090-peak-bf16-tensor-tflops/350543
- FlashAttention sm_120 support PR — https://github.com/Dao-AILab/flash-attention/pull/2634
- CUTLASS sm_120 PR — https://github.com/NVIDIA/cutlass/pull/3030
- Transformer Engine SM120 MXFP8 backward gap — https://github.com/NVIDIA/TransformerEngine/issues/2668
- *LLMQ* (consumer-GPU MFU, PCIe P2P limits) — https://arxiv.org/pdf/2512.15306

**Low-precision pretraining**
- *Pretraining LLMs with NVFP4* — https://arxiv.org/abs/2509.25149
- NVIDIA, *Using NVFP4 for low-precision training* — https://developer.nvidia.com/blog/using-nvfp4-low-precision-model-training-for-higher-throughput-without-losing-accuracy/

**Diagnostics / evaluation**
- Duan et al., *Do Membership Inference Attacks Work on LLMs?* — https://arxiv.org/abs/2402.07841
- Zheng et al., *LLMs Are Not Robust Multiple Choice Selectors* — https://arxiv.org/abs/2309.03882
- Sclar et al., *Quantifying Sensitivity to Prompt Formatting* (FormatSpread) — https://arxiv.org/abs/2310.11324
- Zhai et al., *Stabilizing Transformer Training by Preventing Attention Entropy Collapse* — https://arxiv.org/abs/2303.06296
- Sun et al., *Massive Activations in LLMs* — https://arxiv.org/abs/2402.17762
- Garrido et al., *RankMe* — https://arxiv.org/abs/2210.02885
- McCandlish et al., *An Empirical Model of Large-Batch Training* — https://arxiv.org/abs/1812.06162
- Schaeffer et al., *Are Emergent Abilities of LLMs a Mirage?* — https://arxiv.org/abs/2304.15004
- Carlini et al., *Quantifying Memorization Across Neural Language Models* — https://arxiv.org/abs/2202.07646

**Architectures (looped / recurrent depth)**
- Dehghani et al., *Universal Transformers* — https://arxiv.org/abs/1807.03819
- Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning* (recurrent depth) — https://arxiv.org/abs/2502.05171
- *Looped Transformers are Better at Learning Learning Algorithms* — https://arxiv.org/abs/2311.12424
- *Mixture-of-Recursions* (2025) — https://arxiv.org/abs/2507.10524

**Model baselines used for expected ranges**
- Biderman et al., *Pythia* — https://arxiv.org/abs/2304.01373
- *Cerebras-GPT* (20 tok/param family) — https://arxiv.org/abs/2304.03208

**Repo files verified** (read-only): `docs/PRISM.md`, `docs/PRISM_RECIPE.md`, `crates/prism-recipe/anchors/v0.json`, `crates/prism-recipe/src/anchors.rs`, `crates/prism-recipe/src/lib.rs`, `crates/prism-pipeline/src/composite.rs`, `crates/prism-recipe/harness/eval/*.py`, `crates/prism-recipe/harness/prismlib/*.py`.

---

## Annex — reproducible computations

All numeric tables in §§ 2.2–2.5 and 4.4 were generated by the scripts listed
below, with combined output in `NUMBERS.txt`. **Those scripts are not vendored
into this repo** — they are throwaway analysis, and `docs/spikes/` is evidence,
not product code. Every table states its own inputs, so the arithmetic is
reproducible from the text alone. The G2 item-cap cost model in § 4.4 was
re-measured against this worktree and is now asserted in
`crates/prism-recipe/harness/tests/test_eval_budget.py`.

| Script | Produces |
|---|---|
| `calc.py` | FLOP budget, token budget, naive slope SE, G2 MDD table, G6 anchor bug demonstration |
| `calc2.py` | E-term attenuation, honest slope SE, OLS multi-rung SE, ladder wall-clock costs |
| `calc3.py` | Compute-optimal N sweep, expected G2 ranges, items-needed-to-separate |
| `calc4.py` | Optimum sensitivity (incl. the invalid mixed-constants attempt, kept as a caution), E-confound with corrected E, embedding-fraction table |
| `calc5.py` | **Constants-free optimum argument** (§ 2.3) — the version to rely on |
| `calc6.py` | **FLOP-matched G2 anchor** (§ 4.4) — log-FLOP interpolation between Cerebras-111M/256M, plus the constant-zero normalization demonstration |

Assumptions were set at the top of each script and are restated inline in the sections above: 5090 dense bf16 = 209.5e12 FLOP/s; `C = 6ND`; Chinchilla E/A/B/α/β where used, with the caveat in § 2.3.

