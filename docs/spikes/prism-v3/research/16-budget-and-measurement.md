# Appendix 16 — Budget and Measurement Protocol
> Research appendix for the Prism v3 evaluation proposal (`docs/spikes/prism-v3/`). Produced 2026-08-16 via web research + read-only repository inspection. Non-normative spike document.

**Authority:** non-normative. Per [`docs/AGENTS.md`](../../../AGENTS.md), when a
spike conflicts with a frozen spec ([`docs/PRISM.md`](../../../PRISM.md),
[`docs/BUNDLE_SPEC.md`](../../../BUNDLE_SPEC.md), the pre-registered anchor sets)
**the normative doc wins**. Nothing below is a budget or scoring contract.

**Status of these recommendations:** the compute-currency and `σ_seed`
measurement programme described here is **not** implemented by the emission
work in [appendix 15](15-incentives-and-landscape.md). It is the companion
proposal, and its `σ_seed` measurement is the explicit prerequisite for
enabling the significance-gated emission mode (`PRISM_EMISSION_MODE=sig`) —
see [`docs/PRISM.md`](../../../PRISM.md) § Significance-gated emission.

**Original front matter follows.**

**Objet:** remplacer le budget wall-clock fixe par une monnaie de calcul défendable, et remplacer la
métrique de pente d'échelle confondue par un programme de mesure crédible.
**Statut:** document de conception. Aucune modification de code. Arithmétique reproductible en annexe.
**Entrées:** `/tmp/prism-scaling-research/REPORT.md` (552 l.), `/root/gbase` (lecture seule),
`/root/gbase-v21` (PR #166, lecture seule).

---

## Résumé exécutif (FR)

1. **Le diagnostic de l'utilisateur est juste, sa solution ne l'est pas.** Le wall-clock fixe distord
   bien la compétition — mais en **paliers de taille** (tiers) la distorsion empire au lieu de disparaître.
2. **Pourquoi les paliers échouent:** si le budget de calcul `C_k` croît avec la taille du palier, la perte
   décroît de façon **monotone** en palier. Il n'existe plus d'optimum intérieur: tout le monde déclare le
   palier le plus haut, et la frontière de palier devient le nouveau plafond. On a déplacé le problème.
3. **Aggravant:** Prism émet en **winner-take-all** (un seul hotkey positif par époque). Des paliers
   exigeraient soit de casser le WTA, soit de renormaliser vers un classement global unique — ce qui
   annule l'intérêt des paliers tout en ajoutant le *tier-shopping* (viser le palier au champ le plus faible).
4. **Recommandation: budget iso-FLOPs à double plafond.** La monnaie devient les **FLOPs attestés**
   (`C_MAX = 3.0e18`), le wall-clock (`5.0 h`) redevient un simple garde-fou anti-DoS, et le mineur choisit
   librement `N` et `D`. C'est l'« hybride » que l'utilisateur cherche, mais obtenu **par la mesure des
   FLOPs/token du graphe** et non par des catégories administratives: une architecture bouclée à `r=4`
   paie ~3,3× par token et reçoit donc moins de tokens, un MoE ne paie que ses experts actifs. Le budget
   s'adapte automatiquement à la classe d'architecture.
5. **L'attestation est faisable et bon marché.** `torch.utils.flop_counter.FlopCounterMode` sur un
   forward+backward **piloté par le harness** sur des lots à index secrets, multiplié par les tokens
   comptés par le flux du harness. Le mineur ne signe rien. Vérification physique: `C_attesté ≤ pic × t`.
6. **Correction importante au rapport de recherche:** `C = 6ND` est **faux à cette échelle**. À `d=512`, la
   tête `lm_head` vaut **36 %** des FLOPs/token et l'attention quadratique 9 % — `6·N_body` n'en capture
   que 55 %. Toute comptabilité iso-FLOPs doit inclure `6·d·V` et `12·L·d·S`.
7. **Le plafond de paramètres n'est pas le vrai sujet.** À `C_MAX`, l'optimum est à `N_body ≈ 143 M`
   (`N_total ≈ 177 M`), et le **plateau à 0,02 nats s'étend de 88 M à 236 M** (2,7×). Un plafond à 350 M
   comme à 1 B est **non contraignant**: c'est le calcul qui lie, pas les paramètres. Le débat 350 M / 1 B
   est donc largement vide — je le tranche pour la mémoire et le checkpoint, pas pour la science.
8. **La pente d'échelle ne doit pas être scorée.** Le confondant `E` la réduit à 30–56 % de `α`, et même
   la variante « croissance d'avantage » (qui annule `E`) a une **différence minimale détectable de 0,023**
   avec 3 graines et une référence fixe — contre un signal architectural plausible de 0,013–0,065. C'est
   à la limite, donc: télémétrie observée, jamais scorée.
9. **Remplacement: un mini-profil IsoFLOP à 5 barreaux** (`C_probe = 2,0e17` par barreau, `L=12` fixe,
   largeur seule). On score le **niveau au minimum du profil** (bien déterminé, SE ≈ 0,01–0,03 nats) et la
   **qualité d'ajustement/convexité**. On **ne score pas** l'argmin: il n'est déterminé qu'à ±17–47 % en `N`.
10. **Tournoi à deux étages.** Criblage iso-FLOPs pour tous (financé par le mineur, inchangé), puis
    confirmation IsoFLOP pour le **top-5 seulement** (financée par l'opérateur, **~13 $/candidat**,
    **~64 $/époque**). Cela représente **77 %** du coût de criblage à 10 soumissions mais **8 %** à 100:
    il faut donc conditionner `k` à la taille du champ (`k = min(5, ⌈n/4⌉)`), pas le lancer inconditionnellement.
11. **Le scoring normalisé « compute-optimal » (option 5) est viable en version restreinte seulement.**
    La courbe de référence `L_ref(C)` est **mesurable** (~47–94 $ une fois) par interpolation sur l'intervalle
    de la compétition. Mais elle ne doit **pas** entrer dans le score comme bonus: un mineur pourrait
    dépenser moins pour être jugé contre une référence plus faible. Utilisation: ancres, correction de
    troncature **bornée à ≤ 0**, et diagnostic publié.
12. **Décision demandée:** (a) émettre la télémétrie FLOPs en Zone A non ancrée dès maintenant, sous le
    wall-clock actuel; (b) recipe 2.1.0 avec double plafond une fois la télémétrie calibrée; (c) ancres v3
    pré-enregistrées après mesure sur les deux baselines E6; (d) ne jamais scorer une pente.

---

# Detailed analysis (EN)

## 0. What the current protocol actually does

Verified against `/root/gbase` (and `/root/gbase-v21` for the newer state). The facts that constrain every
option below:

| Mechanism | Current state | Consequence |
|---|---|---|
| Train budget | `TRAIN_HOURS_CAP = 6.0` h wall-clock; `ctx["guard"]` closure the miner must call itself | **Cooperative.** Backstop is a parent SIGKILL at `cap + 120 s`, which loses the entire run |
| Step budget | `ctx["max_train_steps"] = 20000` | **Never enforced.** No harness-side check exists |
| Token budget | *none* | There is no token-budget key in `ctx` at all |
| Param cap | `MAX_PARAMS` 350 M (`main`) / 1 B (v21); counted as `sum(p.numel() for p in model.parameters())` | Dedupes tied embeddings *implicitly*; **no body/embedding split**, **no active-param notion** |
| Token accounting | `stream.tokens_seen`, harness-owned; degrades to a row count when `tokens_seen_source == "legacy"` | Trustworthy **only** when the miner uses the harness stream |
| FLOPs / MFU | **nothing** — no FLOPs, TFLOPS, MFU, MoE or loop-factor concept anywhere in the harness | The compute a submission actually spent is currently unknown |
| Eval budget | independent per-group ceilings summing to ≈ 3.9 h, under a single 3 h `PRISM_EVAL_TIMEOUT_S` hard kill | Over-subscribed; a hard kill **fails the run** rather than truncating |
| G6 probe x-axis | fires on `state["reports"] % PRISM_PROBE_EVERY`, i.e. a **miner-controlled counter**; curve is in **nats/token** | Probe density and therefore AUC are miner-gameable; tokenizer-dependent |
| G8 sweep | `d_model=128, n_layer=4`, **10 steps**, plain `AdamW`, **no schedule**, LRs `[3e-4,1e-3,3e-3]`, widths `(1.0, 4.0)` | 10 steps measures the init transient, not LR transfer |
| Anchors | `main`: only `v0` (`LATEST_ANCHOR_VERSION = 0`). v21: adds `v1`, `v2`; `DEFAULT_ANCHOR_VERSION = 0` | Live scoring is `PRISM_SCORING_MODE=benchmarks` (v4 G2 lattice), not the composite |

Two structural facts about the anchor system govern the whole migration path:

- A metric **in metrics.json but absent from the anchor set is silently ignored** (test
  `unknown_metrics_are_ignored`). New harness keys are therefore **inert** until declared.
- A metric **in the anchor set but absent from metrics.json is a hard `MissingMetric` completeness
  failure** → `Ineligible` → lattice 0. Declaring a key makes it **mandatory**.

⇒ **Emit first, declare later.** This ordering is forced, not stylistic.

## 1. Framing: what is the budget *for*?

A budget in an architecture competition has three jobs, and they conflict:

1. **Fairness** — two submissions should differ in score because of architecture, not because of an
   accident of the cap.
2. **Decision relevance** — the winner should be the architecture you would actually want to scale.
3. **Operational safety** — bounded cost, bounded pod lifetime, no DoS, predictable scheduling.

The choice of budget *currency* decides which of the three you get. The candidates are wall-clock time,
FLOPs, tokens, or a per-class allowance. Before comparing them, one correction to the research report,
because every downstream number depends on it.

### 1.1 Correction: `C = 6ND` is not usable at Prism's scale

The report (and the standard Chinchilla shorthand) uses `C = 6ND`. That counts only matmuls in the
transformer *body*. At Prism's model sizes the neglected terms are not a rounding error. With `V = 32768`,
`S = 512`, fwd+bwd factor 3:

```
F_tok  =  6·N_body·r_eff   +   6·d·V        +   12·L·d·S
          (body matmuls)       (lm_head)        (attention quadratic)
```

| `d` | `L` | `N_body` | body | lm_head | attention |
|---|---|---|---|---|---|
| 512 | 8 | 25.2 M | 54.5 % | **36.4 %** | 9.1 % |
| 768 | 8 | 56.6 M | 64.3 % | 28.6 % | 7.1 % |
| 1024 | 12 | 151.0 M | 76.6 % | 17.0 % | 6.4 % |
| 1536 | 24 | 679.5 M | 88.5 % | 6.6 % | 4.9 % |
| 2048 | 24 | 1208.0 M | 91.1 % | 5.1 % | 3.8 % |

At `d = 1024, L = 12`, `6·N_body` overstates the affordable token count by **1.31×**. At `d = 512` it
overstates it by **1.83×**. Any iso-FLOP accounting that uses `6ND` hands a free 30–80 % compute bonus to
whoever picks a small `d` with a large vocabulary. This also means the `lm_head` term **self-taxes vocab
inflation** — a property I rely on in §3.3.

### 1.2 The optimum is a plateau, not a point

Re-deriving the compute optimum with the corrected `F_tok`, at `C = 3.0e18`, using Hoffmann's
`E/A/B/α/β` **as an illustration only** (Besiroglu showed that fit is not reproducible; the *shape* is what
matters, not the constants):

| choice | `N_body` | `N_total` | `D` | `L` (illustrative) | vs optimum |
|---|---|---|---|---|---|
| `d=832, L=18` | 149.5 M | 176.8 M | 2.60 B | 3.3143 | +0.0015 |
| `d=1024, L=12` | 151.0 M | 184.5 M | 2.54 B | 3.3189 | +0.0061 |
| `d=1280, L=16` | 314.6 M | 356.5 M | 1.32 B | 3.3609 | +0.0481 |
| `d=1536, L=20` | 566.2 M | 616.6 M | 0.77 B | 3.4532 | +0.1404 |
| `d=2048, L=24` | 1208.0 M | 1275.1 M | 0.38 B | 3.6515 | +0.3387 |

Optimum `N_body ≈ 143 M`. **Plateau width** — the range of `N_body` within a given loss tolerance of the
optimum:

| tolerance | `N_body` range | span |
|---|---|---|
| 0.005 nats | 117 M – 184 M | 1.6× |
| 0.010 nats | 107 M – 200 M | 1.9× |
| 0.020 nats | 88 M – 236 M | **2.7×** |
| 0.050 nats | 69 M – 321 M | 4.7× |
| 0.100 nats | 48 M – 455 M | 9.5× |

Two conclusions the report does not draw:

- **Any param cap above ~350 M total is non-binding**, at either 350 M or 1 B. Compute binds. The
  350 M-vs-1 B debate is therefore *mostly empty* as a scientific matter — it matters for VRAM, optimizer
  state and the checkpoint budget (`n_params × 12` bytes), not for the achievable loss.
- **The plateau is 2.7× wide at the noise floor.** So a tier scheme that separates 100 M from 200 M is
  separating models whose achievable loss differs by less than seed noise. The tier boundary would be
  measuring nothing.

I **agree** with the report that raising the cap to 1 B does not buy better models. I **disagree** that it is
a trap worth much attention: at a non-binding cap, a miner who moves to 1 B is simply choosing a worse
point on a curve they can see. Publishing the table is the fix; the cap number is close to irrelevant.

## 2. The five options

### Option 1 — Fixed wall-clock (status quo)

**Mechanism.** Every submission gets the same `T = 6 h`. Compute spent is whatever the implementation
achieves: `C = peak × MFU_achieved × T`.

**Why it distorts.** It makes MFU a scored quantity. Two identical architectures differ in score by their
kernel maturity. Three specific mechanisms, all live on this hardware:

- **sm_120 kernel lottery.** FlashAttention 2/3 have no sm_120 cubins; FA3's WGMMA techniques cannot be
  back-ported. An architecture whose op mix happens to have a fused kernel wins over one that does not,
  independent of merit.
- **Triton version rent.** The report's Triton 3.3 → 3.7 finding (~17 % throughput) means a miner who
  knows to install PyTorch nightly gets a free 17 % compute bonus. Under v21's new miner-installable
  dependency path (`prismlib/deps.py`, `requirements.txt` wins), this is now *easier* to exploit, not harder.
- **Architecture-class penalty.** A looped model at `r = 4` pays ~3.3× FLOPs/token (not 4×: head and
  attention do not loop) and therefore sees ~3.3× fewer tokens. Under wall-clock this is charged to the
  architecture as if it were a quality defect.

**What it is nonetheless good for, and this is not trivial:**

- **Cost predictability.** Pod cost is `T × $/h`, known exactly in advance. Miners fund their own pods via
  `X-Lium-Api-Key`; a budget they cannot bound is a budget they cannot pay for.
- **Anti-DoS.** A wall-clock cap is the only thing that bounds a pathologically slow submission. A pure
  FLOPs cap does not: a miner with a 2 % MFU implementation would hold a pod for days.
- **Enforceability without trust.** Wall-clock needs no attestation. It is measured by the operator's own
  clock. Every other currency needs a measurement the miner could try to corrupt.
- **It is the deployment-relevant currency for inference.** Time, not FLOPs, is what a user waits.

**Verdict.** Keep wall-clock as a **safety bound**, discard it as the **scoring currency**. The distinction is
the core of my recommendation.

### Option 2 — Fixed FLOPs (iso-FLOP)

**Mechanism.** `C_MAX` FLOPs per submission. Miner chooses `N`, `D`, batch, schedule freely.

**How to measure or attest FLOPs on a pod you do not fully trust.** Three approaches; I recommend a
specific hybrid.

*(a) Analytic accounting from the model graph.* Compute `F_tok` from the declared architecture:
`6·N_body·r_eff + 6·d·V + 12·L·d·S`. **Rejected as primary.** It requires the harness to know `r_eff`, MoE
routing sparsity, and every non-standard op — i.e. it requires *understanding a novel architecture*, which
is precisely what Prism cannot assume. It is the one thing a miner submitting novel code can most easily
misrepresent. Keep it as a **cross-check**, not the meter.

*(b) Hardware counters.* CUPTI/DCGM SM-activity or `nvidia-smi` energy. **Rejected as primary.** DCGM
`PROF_PIPE_TENSOR_ACTIVE` needs profiling permissions often unavailable in a container; energy is a proxy
whose FLOPs-per-joule ratio varies by op mix, so it penalizes memory-bound architectures. G7 already shows
the fragility: its joules metric is a 2-sample `nvidia-smi` power delta, not integrated energy. Useful as a
**plausibility bound**, not a meter.

*(c) Dispatch-level measurement — recommended.* `torch.utils.flop_counter.FlopCounterMode` intercepts
`__torch_dispatch__` and counts matmul / conv / SDPA FLOPs of *ops that actually executed*. Verified
available in torch ≥ 2.0 (I confirmed the import on torch 2.13; the pod image ships 2.12). This gets MoE
sparsity, looping, and the attention quadratic term **for free and correctly**, because it measures the
realized computation rather than modelling it.

The attestation protocol I recommend is **harness-owned end to end**, so the miner never reports a number:

```
F_tok_probe  = median over 8 harness-driven fwd+bwd passes on batches drawn from the
               harness stream at SECRET indices (seeded from PRISM_EVAL_SECRET_SEED),
               measured under FlopCounterMode, divided by tokens in the batch
C_attested   = F_tok_probe × stream.tokens_seen         # both harness-owned
```

`stream.tokens_seen` is already harness-owned and already has a trust discriminator
(`tokens_seen_source == "train_stream"`, which the Rust side already refuses to treat as authoritative
otherwise). So the entire product is harness-measured.

**Cheat surface and hardening.**

| Attack | Mechanism | Hardening |
|---|---|---|
| Under-report FLOPs to buy compute | Miner cannot: it never reports them. The residual attack is on `F_tok_probe` | Probe on **real training batches at secret indices**; take **max** across probes, not mean, when the spread exceeds a threshold |
| Input-dependent cost | MoE that routes to fewer experts on probe-shaped inputs; early-exit on probe inputs | Probes are indistinguishable from training batches (same stream, same shapes). Emit `flops_probe_cv`; a high coefficient of variation is a flag |
| Bypass the harness stream | Own dataloader ⇒ `tokens_seen` is a row count | Already detected (`tokens_seen_source`). Make it **fail-closed** for iso-FLOP eligibility, not just non-authoritative |
| Exceed the budget silently | Train past `C_MAX` | Enforce **inside the stream**: refuse to yield batches once `F_tok_probe × tokens_seen ≥ C_MAX`. This is a hard stop, unlike today's cooperative `guard` |
| Physically impossible claim | Any `C_attested > peak × n_gpu × t_wall` | Assert it; on violation emit `inconsistent_metrics` (the taxonomy already has this code) |

**Determinism and reproducibility.** `F_tok_probe` is deterministic given the model, the batch indices and
the dtype. It is *not* invariant to dtype or to autocast regions, which is correct: an fp8 matmul really is
cheaper. It must be recorded alongside the dtype so two runs are comparable. Re-running the probe on the
parked checkpoint reproduces `F_tok_probe` to within kernel-selection noise (zero, for a dispatch counter).

**Weakness.** `FlopCounterMode` counts what the dispatcher sees. A custom CUDA/Triton kernel registered as
a single opaque op is **not counted**. This is a real hole: v21 permits miner dependencies, so a miner could
ship a fused kernel that under-counts. Mitigation: cross-check (a) against (c) and flag a gap > 25 %; and
treat an uncounted-op fraction above a threshold as `suspicious` for the agentic reviewer. **I flag this as
the single largest residual risk in the recommendation.**

### Option 3 — Fixed token budget `D` (iso-data)

**Mechanism.** Every submission trains on exactly `D` tokens.

**Merits.** Trivially enforceable — the harness stream already counts tokens and can simply stop. No
attestation problem at all. Immune to the kernel lottery.

**Why I reject it.** It rewards throughput-optimized models. At fixed `D`, a small fast model and a large
slow model are charged the same, so the optimal play is to be as small as still-competitive and spend the
saved time on nothing. It inverts the wall-clock bias rather than removing it, and it makes the compute
spent — the thing an operator actually pays for — unbounded.

**The tokenizer interaction, quantified.** G1 is scored in **bits/byte**, which is tokenizer-neutral *as a
metric*. But under iso-data the tokenizer changes how much **text** the budget buys:

```
bytes seen = D_tokens × bytes_per_token
```

| bytes/token | text seen at `D = 4e9` | vs 4.3 B/tok baseline |
|---|---|---|
| 3.0 | 12.00 GB | −30.2 % |
| 4.0 | 16.00 GB | −7.0 % |
| 4.3 | 17.20 GB | ±0 |
| 4.8 | 19.20 GB | +11.6 % |
| 5.5 | 22.00 GB | **+27.9 %** |

Converting to loss with `dL/dln D = −β·B/D^β ≈ −0.236` nats per e-fold at `D = 4e9`:

| bytes/token | equivalent `Δln D` | `ΔL` (nats) |
|---|---|---|
| 3.0 | −0.360 | **+0.085** |
| 5.5 | +0.246 | **−0.058** |

So a miner who pushes compression from 4.3 to 5.5 bytes/token gains ≈ **0.058 nats** — larger than the
entire 350 M-vs-1 B cap effect at the optimum (0.043 nats), and larger than most genuine architectural
deltas. Under iso-data, **tokenizer compression becomes a primary scoring axis**, which is not what the
subnet is for. v21's tokenizer anti-cheat card (`vocab_multiword_frac`, `probe_tokens_per_byte`) detects the
extreme form, but a 4.3 → 5.5 shift is a legitimate BPE choice, not a cheat — it would not be flagged, and
it should not be, which is exactly why iso-data is the wrong currency.

Note this interaction **disappears under iso-FLOPs**: a bigger vocabulary raises `6·d·V` in `F_tok`, so
compression is paid for in the budget. At `d = 1024`, moving `V` from 32 768 to 131 072 raises the head
term from 22 % to 89 % of the body cost. Iso-FLOPs prices the tokenizer automatically.

### Option 4 — Size-class tiers (the user's hybrid intuition, made rigorous)

**The steel-manned version.** Discrete tiers by active-parameter count, each with its own compute
allowance and its own anchor set, so architectures compete within a class; cross-class comparison via a
normalized score. Boundaries from the Chinchilla arithmetic rather than round numbers:

| tier | `C_k` | `N*_k` at `r=20` | wall @ MFU 25 % | pod cost (est.) |
|---|---|---|---|---|
| T-small | 1.0e18 | 91 M | 1.33 h | $2.12 |
| T-mid | 3.0e18 | 158 M | 3.98 h | $6.36 |
| T-large | 9.0e18 | 274 M | 11.93 h | $19.09 |

**I recommend against this, and the reasons are structural rather than practical.**

1. **It destroys the interior optimum.** Under a *single* compute budget, `L(N)` at fixed `C` is U-shaped:
   there is a best `N`, and finding it is part of the architectural skill being measured. If `C_k` grows with
   the tier, then `L` is **monotone decreasing** across tiers — bigger tier is always better. The optimum
   moves to the tier boundary and the boundary becomes the new cap. This is the same pathology the report
   identifies in the current design, relocated one level up.
2. **Cross-tier normalization is not establishable.** To compare a T-small winner against a T-large winner
   you need `L_ref` per tier, measured to a precision finer than the between-submission spread, at every
   tier, for every anchor version. That is 3× the calibration cost (§5.4) and it must be redone on every
   reference-architecture, data, tokenizer, or GPU-stack change. And the normalized comparison is exactly
   as arbitrary as the choice of `L_ref` — you have replaced a defensible physical budget with a
   judgement call.
3. **Winner-take-all makes tiers incoherent.** `prism_registry::apply_wta` emits **one** positive Score leaf
   per epoch (argmax over positive credits, lexicographically smallest hotkey on ties). Tiers therefore
   force one of: (a) split emissions per tier — which requires changing the WTA contract and the emission
   share math, a governance action with chain-facing consequences; or (b) normalize to a single global
   ranking — which reintroduces (2) and leaves tiers doing no work except adding a gaming surface.
4. **Tier-shopping is unpreventable and rational.** Under WTA, the payoff is winning your tier. A miner
   should therefore enter the tier with the weakest field, not the tier that suits their architecture. This
   is not a hardening problem; it is the incentive the design creates. Declared-tier verification (active vs
   total params for MoE, effective FLOPs/token for looped models) is *technically* solvable with the same
   `FlopCounterMode` machinery as Option 2 — but solving it does not address the incentive.
5. **The plateau makes fine tiers meaningless.** §1.2: `N_body` from 88 M to 236 M is within 0.02 nats of
   optimal. A tier boundary inside that range separates models that are computationally equivalent.

**What the tier intuition is *right* about, and how to keep it.** The user's underlying diagnosis is
correct: a single fixed budget should not make whole architecture classes un-winnable. But the mechanism
that fixes it is not administrative classes — it is **choosing a currency that already accounts for class**.
Under iso-FLOPs measured from the realized graph:

- a looped model at `r = 4` pays 3.3× per token, so it gets 3.3× fewer tokens — and if the recurrence is
  genuinely worth it, it still wins, because it is competing on loss at equal compute;
- an MoE pays only for **active** experts, because the dispatch counter only sees the experts that ran;
- a model with a big vocabulary pays for its head.

The budget becomes class-adaptive **automatically**, by measurement rather than by declaration — and there
is no tier to shop for, no boundary to game, and no per-tier reference to calibrate. That is the hybrid,
and it does not need tiers to exist.

### Option 5 — Compute-optimal-normalized scoring

**Mechanism.** One compute cap; miner picks `N` and `D` freely; score *relative to the compute-optimal
frontier* for the FLOPs actually spent — reward beating `L*(C_spent)`.

This is the most architecture-neutral formulation on paper, and the report says absolute laws are not
fittable. So: **is the reference frontier honestly establishable at this scale?** Working it through
carefully, because this is the crux.

**What is and is not needed.** The report's "not fittable" finding is about **extrapolation** — recovering
`α, β, E` and predicting across orders of magnitude (Farseer: ~1000 LLMs, 3 M H100-hours, for 0.5 % error
at >1 OOM). That is genuinely out of reach. But Option 5 does not need it. It needs an **interpolation
table** of `L_ref(C)` over the narrow interval the competition actually spans — roughly
`0.3·C_MAX … C_MAX`, i.e. **half an order of magnitude**. Over that interval, a local log-linear fit through
measured points is interpolation, not extrapolation, and it makes no claim about `E`.

**Cost of establishing it** (est., $0.40/GPU-h, anchored on the repo's `$2.5/h` per-pod guard and the
$0.97/3-run evidence wave):

| artifact | runs | pod-h | cost |
|---|---|---|---|
| Full IsoFLOP surface (5 sizes × 4 compute levels) | 20 | 63.9 | **$102** (2 seeds: $205) |
| Single IsoFLOP slice at `C_MAX` (5 sizes) → `N*` and `L*(C_MAX)` | 5 | 29.5 | **$47** (2 seeds: $94) |
| Truncation curve `L_ref(C)` at fixed `N = N*` (4 levels) | 4 | 12.4 | **$20** |

So the frontier is **affordable** — $67–115 one-time for the slice plus the truncation curve. It is a
one-time cost per (reference architecture, data pin, tokenizer, GPU stack) tuple, and it must be re-measured
whenever any of those change.

**So why do I not recommend scoring against it?** Two reasons, one fatal.

*Reason 1 (fatal): it creates a deliberate-underspend attack.* If the score is `L_ref(C_spent) − L_sub`,
the miner chooses `C_spent`. The frontier's slope is roughly −0.05 to −0.10 nats per e-fold of compute. A
miner whose architecture improves **slower** than that with more training gains by stopping early — they
are scored against a weaker reference. This is not exotic: any architecture that saturates early has this
property, and stopping early is free. The attack is *structural*, not an implementation gap.

*Reason 2: it does not actually deliver architecture-neutrality.* A frontier measured on one reference
family (Transformer++) means "beat Transformer++ at this compute". That is what a plain reference-relative
bpb anchor already gives you. The frontier's genuinely distinct contribution is narrower than it looks: it
lets you compare submissions that spent **different** compute. Which is a real need — but only because
wall-clock truncation makes `C_spent` vary.

**Verdict: viable in a restricted role, not as the scored surface.** Specifically:

- **Do** measure the frontier and publish it (miner docs, the `(N,D)` menu of §3.2). It is the honest way to
  tell miners where the optimum is instead of letting them discover the cap is a trap.
- **Do** use `L_ref` values as the `reference` fields in the `bpb_log_ratio` anchors. That is already how
  v0 works; the frontier just makes those numbers measured rather than placeholder.
- **Do** use it for a **truncation correction bounded at ≤ 0** — if wall-clock cut a submission off at
  `C_spent < C_MAX`, correct *toward* the full-budget equivalent but never award a bonus. Bounding at ≤ 0
  kills the underspend attack: underspending can never gain points.
- **Do not** make `L_ref(C_spent) − L_sub` a scored metric. Emit it as unscored `org.*` telemetry
  (`org.diag.frontier_delta`) so the evidence accumulates for a future decision.

## 3. Comparison and recommendation

### 3.1 Comparison table

| | 1. Wall-clock | 2. Iso-FLOP | 3. Iso-data | 4. Tiers | 5. Frontier-normalized |
|---|---|---|---|---|---|
| **Enforceable without trust** | ✅ perfect | ⚠️ needs dispatch counter | ✅ perfect | ⚠️ + tier verification | ⚠️ inherits 2 |
| **Cost predictable** | ✅ exact | ⚠️ bounded by wall-clock backstop | ❌ unbounded compute | ✅ per tier | ⚠️ inherits 2 |
| **Anti-DoS** | ✅ | ❌ alone; ✅ with wall backstop | ❌ | ✅ | ❌ alone |
| **Neutral to kernel maturity** | ❌ MFU is scored | ✅ | ✅ | ❌ within tier | ✅ |
| **Neutral to architecture class** | ❌ looping punished | ✅ automatic via realized graph | ❌ favours small/fast | ⚠️ by declaration, gameable | ✅ |
| **Tokenizer-neutral budget** | ⚠️ partly | ✅ head cost priced in | ❌ 0.058 nats swing | ⚠️ | ✅ |
| **Preserves an interior optimum** | ✅ | ✅ | ❌ | ❌ **monotone in tier** | ✅ |
| **Compatible with WTA emission** | ✅ | ✅ | ✅ | ❌ needs contract change | ✅ |
| **New gaming surface** | kernel/Triton rent | uncounted custom kernels | tokenizer compression | tier-shopping | **deliberate underspend** |
| **Calibration cost** | $0 | $0 | $0 | 3× per-tier refs | $67–115 one-time |
| **Verdict** | keep as **bound** | **RECOMMEND as currency** | reject | reject | restricted role only |

### 3.2 Recommendation: dual-capped iso-FLOP with free `(N, D)`

**The budget is a FLOPs cap, with wall-clock demoted to a safety bound. The miner chooses `N` and `D`
freely under it. Whichever cap binds first stops the run, and the score records which one it was.**

Exact constants:

| constant | value | role |
|---|---|---|
| `TRAIN_FLOPS_CAP` | **`3.0e18`** attested FLOPs (fwd+bwd+recompute) | **the currency** |
| `TRAIN_HOURS_CAP` | **`5.0`** h | safety bound / anti-DoS only, **not** the currency |
| `POD_LIFETIME_HOURS_CAP` | `7.0` h (unchanged) | pod lifetime |
| `EVAL_GLOBAL_BUDGET_S` | **`4320`** (1.2 h), newly *global* | replaces per-group ceilings summing past the kill timer |
| `MAX_PARAMS` | **`1_000_000_000`** total (v21's value, kept) | VRAM / checkpoint guard, **non-binding** on the science |
| `MIN_SPEND_FRACTION` | **`0.5`** of `TRAIN_FLOPS_CAP` | eligibility floor; below this the run is `Ineligible`, not merely scaled |
| `FLOPS_PROBE_SAMPLES` | `8` | dispatch-counter probes at secret stream indices |
| `FLOPS_PROBE_CV_MAX` | `0.15` | above ⇒ `flops_probe_unstable` flag |
| `FLOPS_ANALYTIC_GAP_MAX` | `0.25` | analytic vs measured mismatch ⇒ agentic evidence |

**Why `C_MAX = 3.0e18`.** It must be reachable inside the wall-clock bound by a *competent but not
heroic* implementation, so that FLOPs — not time — is what binds for most submissions:

| `C_MAX` | wall @ MFU 20 % | 25 % | 30 % | 35 % |
|---|---|---|---|---|
| 2.5e18 | 4.14 h | 3.31 h | 2.76 h | 2.37 h |
| **3.0e18** | **4.97 h** | **3.98 h** | **3.31 h** | **2.84 h** |
| 3.5e18 | 5.80 h | 4.64 h | 3.87 h | 3.31 h |
| 4.0e18 | 6.63 h | 5.30 h | 4.42 h | 3.79 h |

At `C_MAX = 3.0e18` and a 5.0 h bound, **any implementation at ≥ 20 % MFU is FLOPs-bound**, i.e. the
kernel lottery stops mattering for essentially the whole field. At 3.5e18 the 20 % case is already
wall-bound; at 2.5e18 we leave compute on the table. 3.0e18 is the right choice, and it is deliberately
*below* the 6 h/30 % figure the report uses, because the binding constraint should be the one we can
attest.

**The `(N, D)` menu this creates** — published to miners, computed with the corrected `F_tok`:

| `d` | `L` | `N_body` | `N_total` | `F_tok` | `D` | `D/N_total` |
|---|---|---|---|---|---|---|
| 512 | 8 | 25.2 M | 41.9 M | 2.77e8 | 10.84 B | 258 |
| 768 | 12 | 84.9 M | 110.1 M | 7.17e8 | 4.18 B | 38.0 |
| **1024** | **12** | **151.0 M** | **184.5 M** | **1.18e9** | **2.54 B** | **13.7** |
| 1280 | 16 | 314.6 M | 356.5 M | 2.27e9 | 1.32 B | 3.7 |
| 1536 | 20 | 566.2 M | 616.6 M | 3.89e9 | 0.77 B | 1.25 |
| 2048 | 24 | 1208.0 M | 1275.1 M | 7.95e9 | 0.38 B | 0.30 |

**Why no active-param cap.** Under iso-FLOPs an MoE already pays for its active experts through
`F_tok`, so a second constraint on active params would be **double-binding** and would re-create the tier
problem in miniature. The total-param cap stays only because VRAM, AdamW state and the
`n_params × 12`-byte checkpoint budget are real physical limits. A 16-expert/2-active MoE has ~1.65 B total
but ~251 M active params: under a 1 B *total* cap it is excluded despite being **compute-cheaper** than a
dense 700 M model. That is the strongest argument for making the cap `active`-based — but the cleaner fix
is to raise the total cap for MoE via the checkpoint budget rather than adding a scored notion of
"active", which a miner would then have an incentive to misdeclare.

**Why G7 stays raw.** G7 measures inference cost on the trained checkpoint at fixed context lengths with a
harness-owned loop; it does not depend on the training budget at all. Under a *single* budget with free `N`,
G7 remains comparable across submissions and inference cost stays a genuine architectural property — a
looped model *should* lose on TTFT. Under **tiers** it would not be comparable, because a systematically
smaller tier wins G7 for free; that is a further argument against Option 4. G7 does need three fixes
independent of this design (§4.4): pin the dtype, add a KV cache, and fix the contaminated
`state_bytes_per_token` slope.

## 4. What the subnet should measure to find *scalable* architectures

### 4.1 Why the slope is unfixable, including the clever repair

The report's E-confound is correct and I reproduce it: the measured local slope is `α·(1 − E/L)`.

| `E` | `L=2.6` | `L=3.0` | `L=3.4` | `L=3.8` |
|---|---|---|---|---|
| 1.69 (Hoffmann) | 35.0 % | 43.7 % | 50.3 % | 55.5 % |
| 1.82 (Besiroglu) | 30.0 % | 39.3 % | 46.5 % | 52.1 % |

There is an obvious repair worth taking seriously, because if it worked it would rescue the metric.
**Score the growth of the advantage over a fixed reference, not the slope itself.** Define
`Δ(N) = L_sub(N) − L_ref(N)`. If both share the same irreducible `E` (same data, same tokenizer, same eval),
then `E` **cancels exactly** in the difference:

```
Δ(N) = A_s/N^α_s − A_r/N^α_r        (no E term)
dΔ/dlnN ≈ −(α_s − α_r)·(L − E)      near the measured region
```

So `dΔ/dlnN` is an E-free measure of whether the submission's advantage *grows* with scale. Signal size:

| `L − E` | `Δα = 0.01` | `Δα = 0.02` | `Δα = 0.05` |
|---|---|---|---|
| 1.0 | 0.0100 | 0.0200 | 0.0500 |
| 1.3 | 0.0130 | 0.0260 | 0.0650 |
| 1.7 | 0.0170 | 0.0340 | 0.0850 |

Now the noise. OLS over `m` log-spaced rungs, `SE = σ_lnL/√Sxx`, with the reference **reused** (fixed, so
no √2 penalty) and 3 seeds:

| rungs | body span | `Sxx` | `SE` @σ=0.02 | **MDD** (2 candidates, 95 %) |
|---|---|---|---|---|
| 4 | 16× | 4.27 | 0.0056 | **0.0219** |
| 4 | 25× | 5.76 | 0.0048 | 0.0189 |
| 4 | 9× | 2.68 | 0.0071 | 0.0276 |

**Verdict: still not scorable.** The MDD (0.019–0.028) sits *inside* the plausible signal range
(0.013–0.065). It can separate `Δα = 0.05` architectures and cannot reliably separate `Δα = 0.02` ones —
and that is with 3 seeds, a fixed reference, and the optimistic `σ_lnL = 0.02`. At `σ = 0.05` the MDD is
0.055 and the metric is dead. The E-cancellation repair is real and worth *reporting*; it does not survive
contact with the noise budget as a **scored** axis. This is a stronger, more specific version of the
report's conclusion, and it reaches the same place: **telemetry, never scored.**

### 4.2 The replacement: an IsoFLOP mini-profile

Three ladder designs are possible. I costed all three and the choice is not close.

| design | what it measures | cost/candidate (3 seeds) | flaw |
|---|---|---|---|
| **(a) Fixed-`D` width ladder** | loss vs `N` at fixed data | **$2.16–3.24** | top rung at `D/N_body = 0.7` is **severely undertrained**; curvature is data-limitation, not capacity. Slope still E-confounded |
| **(b) Iso-ratio ladder** (`D_i = r·N_i`) | clean `α` in `N`, all rungs equally trained | **$8.70** (r=5) – **$17.40** (r=10) | cost scales as `N²`; 11 h wall at r=10; still E-confounded |
| **(c) IsoFLOP mini-profile** (fix `C_probe`, vary `N`) | **existence and location of the loss-vs-`N` minimum at fixed compute** | **$12.82** | argmin poorly determined; no `α` |

**(c) wins**, and not on cost — on relevance. In a compute-budgeted competition the decision-relevant
question is not "what is this architecture's `α`" (unanswerable, and the report is right about that). It is
**"at a given compute budget, what is this architecture's best achievable loss, and does its optimal size
differ from the reference's?"** That is exactly an IsoFLOP profile, it is **E-free** (it compares losses at
fixed `C`, never a log-log exponent), and its cost is **fixed per rung** rather than growing with `N`.

Recommended geometry — `C_probe = 2.0e17` per rung (6.7 % of `C_MAX`), 5 rungs, **depth fixed at
`L = 12`, width only** (µP is known not to transfer across depth — Tensor Programs VI, Bordelon et al.):

| rung | `d` | `N_body` | `D = C_probe/F_tok` | `D/N_body` | 1-GPU time (est.) |
|---|---|---|---|---|---|
| R1 | 320 | 14.75 M | 1143 M | 77.5 | 227 min |
| R2 | 448 | 28.90 M | 679 M | 23.5 | 152 min |
| R3 | 640 | 58.98 M | 380 M | 6.4 | 106 min |
| R4 | 832 | 99.68 M | 243 M | 2.4 | 84 min |
| R5 | 1024 | 150.99 M | 169 M | 1.1 | 72 min |

Total `5 × 2.0e17 = 1.0e18` FLOPs = **33 % of one `C_MAX` submission**. With Hoffmann constants the
predicted minimum lands at R3 (`d = 640`) — **interior**, which is the property the range must have: an
argmin at an edge means the range is wrong and the profile is uninformative.

**What to score, and what not to.**

| statistic | determinacy | verdict |
|---|---|---|
| **Level at the profile minimum** (`min_i bpb_i`) | `SE ≈ σ/√seeds ≈ 0.01–0.03 nats` — well determined | **SCORE** |
| **Convexity / fit quality** (`r²` of a quadratic in `ln N`) | robust; a non-convex profile is a strong cheat/instability signal | **SCORE** |
| **argmin location** (`N*_sub`) | `SE(ln N*) ≈ σ/k`: at curvature `k ≈ 0.13`, σ=0.02 ⇒ **±17 %** in `N`; σ=0.05 ⇒ ±47 % | **telemetry + CI, do not score** |
| **Δ-slope / advantage growth** | MDD 0.019–0.028 vs signal 0.013–0.065 (§4.1) | **telemetry, do not score** |

The argmin is poorly determined *by construction*: the minimum is flat (that is what §1.2's plateau means),
so its location is the least identifiable feature of the curve. Scoring it would be scoring noise. Report it
with an honest interval.

**On "level + fit quality rather than raw slope" (the report's suggestion):** I agree, with one correction.
The report proposes `org.g8.ladder_top_bpb` — the level at the **top rung**. That is the wrong level: the top
rung of a fixed-`D` ladder is the most data-starved point, so it measures data limitation. The level at the
**IsoFLOP profile minimum** is the right statistic — it is the architecture's best achievable loss at a known
compute budget, which is precisely the quantity the subnet exists to rank.

### 4.3 Two-stage tournament, costed

**Stage 1 — screen (everyone).** The iso-FLOP run of §3.2 at `C_MAX`, miner-funded via
`X-Lium-Api-Key`, exactly as today. Produces the full G1–G8 battery. This is the scored path.

**Stage 2 — confirm (finalists only).** Operator-run IsoFLOP profile on the top-k by Stage-1 lattice score,
2–3 seeds. Its outputs are **not** part of the Stage-1 composite; they are a separate confirmation record.

Cost per epoch (est., $0.40/GPU-h):

| | per candidate | top-3 | top-5 |
|---|---|---|---|
| 2 seeds | $8.55 | $25.64 | $42.74 |
| 3 seeds | $12.82 | $38.46 | $64.10 |

Against Stage-1 screening cost (miner-funded, `4 h train + 1.2 h eval` at `$1.60/pod-h`):

| submissions/epoch | screen cost | confirm top-5 × 3 seeds | confirm as % of screen |
|---|---|---|---|
| 10 | $83 | $64 | 77 % |
| 30 | $250 | $64 | 26 % |
| 100 | $832 | $64 | 8 % |

The confirmation tier is **cheap in absolute terms** ($64/epoch) and its relative cost falls as the field
grows. At a 10-submission epoch it is a meaningful fraction — so gate it on field size (`k = min(5, ⌈n/4⌉)`)
rather than running it unconditionally.

**Who pays.** Prism miners fund their own pods. A confirmation tier the *operator* runs needs operator
funding (`PRISM_ALLOW_OPERATOR_LIUM=1` already exists as a fallback path). Since WTA means only one hotkey
is paid, spending $64/epoch to verify the winner is proportionate. Do **not** bill it to the miner: the
finalist would then have an incentive to decline, and declining must not be an option.

### 4.4 Making the ladder cheat-resistant

| attack | mechanism | hardening |
|---|---|---|
| **Special-case small widths** | init/scale tricks that only help at `d = 320` | Organizer-fixed geometry **and** organizer-fixed per-rung LR from a µP prescription. Score the **profile minimum**, so degrading any single rung cannot help |
| **Detect the probe** | branch on `d`, on batch shape, or on a `prism_*` ctx key | Build every rung from **one** `build_model` call with only the width knob changed (already the `prism_width_multiplier` convention). Emit `ladder_width_knob_honored` from the `>1.5×` param check. Randomize probe batch indices from `PRISM_EVAL_SECRET_SEED` |
| **Warmup sandbagging** | Porian et al.: a **constant-step** warmup is too long for small models, so it mechanically sabotages the small rungs and inflates any slope — with no code that looks like cheating | **Define warmup as a fraction of total steps, never a step count.** One-line change; the report is right that this is the best value/effort item in the whole program |
| **Embedding-fraction gaming** | 4× width ≠ 4× params when embeddings dominate; inflating vocab shrinks the `ln N` denominator | Count **body params only** for the profile x-axis. At `d = 320` embeddings are 41.6 % of params; using totals inflates a measured slope by ~28 % |
| **Vocabulary inflation** | loss in bits/token shrinks with vocab | Score **bits/byte** at every rung, as G1 already does. Also priced by `F_tok` under iso-FLOPs |
| **Non-monotone rung cherry-picking** | pick rungs where the curve is steep | 5 rungs + published `r²`; gate on convexity |
| **Probe-cost evasion** | MoE routes to fewer experts on probe inputs | `flops_probe_cv` over 8 secret-index probes; take the **max** when CV exceeds `FLOPS_PROBE_CV_MAX` |

Three fixes the ladder depends on that are really pre-existing harness bugs, and must land first:

1. **G8 sweep is 10 steps with no schedule** — it measures the initialization transient, not LR transfer.
   Raise to ≥ 300 steps with an organizer-fixed cosine schedule before any ladder reuses that machinery.
2. **G6 probe x-axis is a miner-controlled counter** (`state["reports"] % PRISM_PROBE_EVERY`). Move to
   **organizer-chosen token milestones** (`1e7, 3e7, 1e8, 3e8, 1e9`). Otherwise probe spacing — and hence
   any AUC — is chosen by the miner.
3. **G7 dtype is never asserted** and the decode loop has **no KV cache** (it re-prefills, so `tpot` is
   O(L) per token and `state_bytes_per_token` is contaminated by a cumulative peak-memory counter that is
   reset once before the loop rather than per length). Pin the dtype; add a cache; reset per length.

### 4.5 Concrete metric specification

Norm kinds are the four the schema actually supports (`anchors.rs`, `#[serde(tag="kind",
rename_all="snake_case")]`): `accuracy{chance}`, `bpb_log_ratio{chance,reference}`,
`efficiency_log_ratio{reference,cap}`, `stability_bounded`. JSON is **flat** — no `norm:{}` wrapper. For
`efficiency_log_ratio`, **`cap < reference` encodes lower-better** (it falls out of the sign algebra; there
is no direction flag — this is exactly what the v0 `org.g6.auc_log_tokens` bug got backwards).

**Group placement note.** `composite.rs` hardcodes `GROUP_KEYS: [&str; 8]` and `N_GROUPS = 8` with
`[f64; N_GROUPS]` arrays throughout. **Adding a G9 is not a JSON-only change.** I therefore place new
metrics in existing groups: compute/data efficiency into **G6** (re-read as "data & compute efficiency"),
scaling/stability into **G8**. A G9 would be cleaner semantically and I note it as a deferred refactor.

#### Scored in v3 (screen tier)

| `org.*` key | Group | Norm kind | chance / reference / cap (**placeholder**) | Harness file |
|---|---|---|---|---|
| `org.g1.bits_per_byte_val` | G1 | `bpb_log_ratio` | chance 3.6 / ref **1.10** | `eval/g1_intrinsic.py` |
| `org.g1.bits_per_byte_fresh_crawl` | G1 | `bpb_log_ratio` | chance 3.6 / ref **1.15** (existing; raise weight) | `eval/g1_intrinsic.py` |
| `org.g1.bits_per_byte_val_train_gap` | G1 | `efficiency_log_ratio` | ref **0.15** / cap **0.02** (lower-better) | `eval/g1_intrinsic.py` |
| `org.g6.auc_log_bytes` | G6 | `efficiency_log_ratio` | ref **1.45** / cap **1.05** (lower-better; **bits/byte**, replaces the nats/token key) | `eval/g6_curve.py` |
| `org.g6.bytes_to_bpb_threshold` | G6 | `efficiency_log_ratio` | ref **8.0e9** / cap **2.0e9** (lower-better; censored ⇒ `1e15` fail-closed) | `eval/g6_curve.py` |
| `org.g6.bpb_at_half_budget` | G6 | `bpb_log_ratio` | chance 3.6 / ref **1.22** — bits/byte at `0.5·C_MAX`, from an **organizer** compute milestone | `eval/g6_curve.py` |
| `org.g8.mup_lr_stability` | G8 | `stability_bounded` | existing; **requires the ≥300-step fix** | `eval/g8_stability.py` |
| `org.g8.loss_spike_score` | G8 | `stability_bounded` | existing | `eval/g8_stability.py` |

**Removed relative to v21's v1/v2:** `org.g8.mup_scaling_slope` (`efficiency_log_ratio`, ref 0.02, cap 0.25).
Per §4.1 it must not be scored in any form.

#### Confirmation tier (separate record, not in the Stage-1 composite)

| `org.*` key | Norm kind | reference / cap (**placeholder**) | Note |
|---|---|---|---|
| `org.conf.isoflop_min_bpb` | `bpb_log_ratio` | chance 3.6 / ref **1.28** | **the** confirmation statistic: level at the profile minimum |
| `org.conf.isoflop_convexity_r2` | `stability_bounded` | — (already in `[0,1]`) | quadratic fit `r²` in `ln N_body`; gate at ≥ 0.85 |
| `org.conf.isoflop_argmin_nbody` | *observed only* | — | report with CI; **never scored** (±17–47 %) |
| `org.conf.advantage_growth` | *observed only* | — | `dΔ/dlnN` vs the fixed reference; **never scored** (§4.1) |

#### Observed-only `org.*`, absent from every anchor set (therefore inert)

Emit these now, under the current wall-clock regime, so v3 anchors can be calibrated on measured
distributions instead of guesses:

`org.diag.flops_attested`, `org.diag.flops_per_token_probe`, `org.diag.flops_probe_cv`,
`org.diag.flops_analytic_ratio`, `org.diag.mfu_achieved`, `org.diag.binding_cap`
(`"flops"|"wall"|"steps"`), `org.diag.spend_fraction`, `org.diag.tokens_seen_verified`,
`org.diag.n_params_body`, `org.diag.n_params_embed`, `org.diag.n_params_active_est`,
`org.diag.effective_flops_per_token_ratio` (vs a dense equivalent — the looping factor `r_eff`),
`org.diag.frontier_delta`, `org.diag.tokenizer_bytes_per_token`, `org.diag.tokenizer_fertility`,
`org.diag.unreachable_token_frac`, `org.diag.effective_rank_mid`, `org.g2.mean_gold_nll`,
`org.g2.brier`, `org.g2.choice_order_consistency`.

Note these must be `org.*`-namespaced but **absent from the anchor set** — not Zone B. Zone B is
`miner.*` and participant-reported by contract, so organizer-measured-but-unscored metrics belong in the
ignored-`org.*` space (the `unknown_metrics_are_ignored` path).

#### G7 changes

Keep the five existing keys. Add `org.g7.throughput_toks_s_per_gparam` (observed first) so size
normalization exists when needed. Fix dtype pinning, the KV cache, and the peak-memory reset per §4.4.

**Flag, not a fix (v21 PR #166):** `org.g7.reasoning_throughput` = `mean(org.g4.*) × org.g7.throughput_toks_s`
**double-counts G4 accuracy** — once inside G4 (weight 0.15) and again inside G7 (weight 0.075). Under a
geometric cross-group mean this silently re-weights G4 upward. Worth raising on that PR; I have not touched it.

## 5. Implementation spec

### 5.1 Enforcement and attestation in the harness

The budget must be enforced where the tokens are handed out, not where the miner is asked to be polite.
Today `ctx["guard"]` is a closure the miner must call, `_CapExceeded` is raised but **caught by nothing**
(it falls into the generic handler and fails the run), and `max_train_steps` is never checked at all. The
replacement puts the cap inside `SeededTrainStream.next_batch`, which the harness owns:

```
# prismlib/stream.py — enforcement point (sketch)
def next_batch(self):
    if self.flops_spent >= self.flops_cap:          # hard stop, harness-owned
        raise BudgetExhausted("flops_cap")          # caught -> graceful checkpoint
    ids, labels = self._draw()
    self.tokens_seen += labels.numel()
    self.flops_spent = self.f_tok_probe * self.tokens_seen
    return ids, labels
```

`BudgetExhausted` must be caught in `train_v3.py` and routed to the **same graceful path** as
`FinishEvaluation` — checkpoint, then eval. Reaching the budget is the *expected* outcome, not a failure;
today's equivalent (`_CapExceeded`) loses the run, which is a live bug in its own right.

`f_tok_probe` is established **before** training starts, in `prismlib/flops.py` (new):

```
# 8 probes on batches drawn at secret indices; harness-driven fwd+bwd
from torch.utils.flop_counter import FlopCounterMode
def probe_flops_per_token(model, stream, secret_seed, n=8):
    samples = []
    for i in _secret_indices(secret_seed, n):
        ids, labels = stream.peek_batch(i)
        with FlopCounterMode(display=False) as m:
            loss = _fwd_bwd(model, ids, labels)     # harness-owned, not miner code
        samples.append(m.get_total_flops() / labels.numel())
    return median(samples), cv(samples)
```

Physical sanity, asserted at eval time: `flops_attested ≤ peak_flops × n_gpu × wall_s × 1.05`. Violation ⇒
`inconsistent_metrics` (the cheat taxonomy already has this code).

### 5.2 File-level change list

| File | Change |
|---|---|
| `crates/prism-recipe/src/lib.rs` | Add `TRAIN_FLOPS_CAP: f64 = 3.0e18`, `MIN_SPEND_FRACTION: f64 = 0.5`, `EVAL_GLOBAL_BUDGET_S`, `flops_cap()` env-override helper mirroring `train_hours_cap()`. Lower `TRAIN_HOURS_CAP` 6.0 → 5.0. Add all to `RecipeDescriptor` + `descriptor()`. Bump `RECIPE_VERSION` → `2.1.0`. **Note: this changes `recipe_pin_hex()`** |
| `crates/prism-recipe/harness/prismlib/flops.py` | **New.** `FlopCounterMode` probe, secret-index selection, CV, analytic cross-check |
| `crates/prism-recipe/harness/prismlib/stream.py` | Track `flops_spent`; hard-stop in `next_batch`; add `peek_batch(i)` for probes |
| `crates/prism-recipe/harness/prismlib/train_v3.py` | Run the probe pre-train; put `flops_cap`/`flops_per_token_probe` in `ctx`; catch `BudgetExhausted` → graceful checkpoint; emit the `org.diag.flops_*` set; **warmup as a fraction of total steps** |
| `crates/prism-recipe/harness/prismlib/manifest.py` | Record the missing knobs in `env_knobs`: `PRISM_MAX_PARAMS`, `PRISM_TRAIN_FLOPS_CAP`, `PRISM_EVAL_TIMEOUT_S`, `PRISM_EVAL_*_BUDGET_S`. Today the manifest attests *neither* the eval budget nor the param cap in force |
| `crates/prism-recipe/harness/prismlib/probes.py` | Trigger on **organizer token/compute milestones**, not `state["reports"] % every` |
| `crates/prism-recipe/harness/eval/g6_curve.py` | Curve in **bits/byte** not nats/token; `auc_log_bytes`, `bytes_to_bpb_threshold` (censor ⇒ `1e15`), `bpb_at_half_budget` |
| `crates/prism-recipe/harness/eval/g8_stability.py` | Sweep ≥ 300 steps + organizer-fixed cosine schedule; ≥ 5 LRs with an edge-hit flag; **drop the scaling-slope emission**; body-only param counts |
| `crates/prism-recipe/harness/eval/g7_inference.py` | Pin dtype; add KV cache; `reset_peak_memory_stats()` **per length**; add `throughput_toks_s_per_gparam` |
| `crates/prism-recipe/harness/eval/g9_isoflop.py` | **New**, confirmation tier only. IsoFLOP profile, 5 rungs, organizer-fixed LR per rung, quadratic fit, `org.conf.*` |
| `crates/prism-recipe/harness/eval/__init__.py` | Global eval budget shared across groups (replaces independent ceilings that sum to 3.9 h under a 3 h kill) |
| `crates/prism-recipe/anchors/v3.json` | **New.** Metrics of §4.5; `gates.max_wall_s` 21600 → 18000; add `gates.max_flops`; `max_params` 1e9 |
| `crates/prism-recipe/src/anchors.rs` | `ANCHOR_SET_V3_JSON` const + `canonical_json` match arm `3 =>`; `LATEST_ANCHOR_VERSION` → 3; `DEFAULT_ANCHOR_VERSION` stays **0** |
| `crates/prism-pipeline/src/composite.rs` | Add a `max_flops` gate alongside `ParamsOverBudget` / `WallClockOverBudget`; add `SpendBelowFloor` for `MIN_SPEND_FRACTION` |
| `crates/prism-lium/…` | Nothing structural — pod rent already carries `gpu_count` (v21 `DEFAULT_POD_GPU_COUNT = 4`) |
| `docs/PRISM.md`, `docs/PRISM_RECIPE.md` | Budget currency, the `(N,D)` menu, the published optimum table |
| `docs/external-miner/` + public `BaseIntelligence/prism` repo | **Mandatory** per root `AGENTS.md`: the budget currency is a miner-facing API change |

### 5.3 Failure modes

| # | Failure | Mechanism | Mitigation | Residual |
|---|---|---|---|---|
| F1 | **Uncounted custom kernel** | A fused Triton/CUDA op registered as one opaque dispatch is invisible to `FlopCounterMode`; v21 allows miner deps, so this is reachable | Analytic cross-check with `FLOPS_ANALYTIC_GAP_MAX = 0.25`; uncounted-op fraction as agentic evidence | **High.** The main hole in the recommendation |
| F2 | **Inflating declared params to buy time** | Under *tiers* this buys a bigger allowance. Under my recommendation it buys **nothing** — the budget keys off measured `F_tok`, not declared size. Inflating params *costs* FLOPs/token | Structural: no declared-size input to the budget | **Eliminated by design** |
| F3 | **Deliberate underspend** | Stop early to be judged against a weaker frontier point | Frontier correction bounded at ≤ 0; `MIN_SPEND_FRACTION = 0.5` eligibility floor | Low |
| F4 | **Probe-shaped input detection** | MoE/early-exit that is cheap on probe batches | Secret indices from `PRISM_EVAL_SECRET_SEED`; probes are real training batches; max-not-mean when `CV > 0.15` | Medium |
| F5 | **G7 incomparable across budgets** | G7 runs on the trained checkpoint at fixed contexts, so it is **already** budget-independent. The real breakages are dtype freedom and the no-KV-cache loop | Pin dtype; add cache; per-length peak reset; add per-Gparam throughput | Low after fixes |
| F6 | **Wall-clock still binds for slow implementations** | A < 20 % MFU submission hits 5.0 h before `C_MAX` | Record `binding_cap`; bounded truncation correction; publish the MFU expectation | Medium — accepted, it is the anti-DoS price |
| F7 | **MFU assumption wrong** | Every token/cost estimate scales linearly with MFU. If real MFU is 15 %, `C_MAX` becomes wall-bound for most of the field | **Measure MFU on both E6 baselines before setting `C_MAX`.** Phase 0 exists for this | Medium |
| F8 | **Eval battery truncation** | Per-group ceilings sum to 3.9 h under a 3 h kill; a kill **fails the whole run** | Global eval budget with graceful degradation + explicit partial flags | Low after fix |
| F9 | **Anchor completeness break** | Declaring a v3 key before the harness emits it makes every in-flight submission `Ineligible` | Emit-then-declare ordering, enforced by the phase gates in §5.5 | Eliminated by ordering |
| F10 | **Reference curve staleness** | `L_ref` is only valid for its (arch, data, tokenizer, GPU stack) tuple | Re-measure on any change; record the tuple hash in the anchor `notes` | Medium — operational discipline |
| F11 | **Confirmation tier gameable via seed luck** | 2–3 seeds is few | Use the same seeds for all candidates (paired comparison); report per-seed values | Low |

### 5.4 Calibration required before any flip

| artifact | runs | cost (est.) | blocks |
|---|---|---|---|
| MFU measurement on both E6 baselines | 2 | ~$13 | `C_MAX`, every cost table |
| IsoFLOP slice at `C_MAX` (5 sizes) → `N*`, `L*(C_MAX)` | 5 | $47 (2 seeds $94) | `bpb_log_ratio` references |
| Truncation curve `L_ref(C)` at `N*` (4 levels) | 4 | $20 | bounded truncation correction |
| Confirmation-tier reference profile (fixed ref, 3 seeds) | 15 | $38 | `org.conf.*` anchors |
| **Total** | 26 | **~$120–170** | v3 anchor activation |

### 5.5 Migration path

The repo's discipline: **v0 is the live default and byte-frozen** (`DEFAULT_ANCHOR_VERSION = 0`, asserted in
`anchors.rs` tests); v1/v2 are hash-committed pre-registration artifacts, unselected; a new scoring surface
means a new anchor version **plus** pre-registration **plus** a governance flip. Live scoring today is
`PRISM_SCORING_MODE=benchmarks` (v4 G2 lattice), *not* the composite — so the composite work is
pre-registration, not a live change, until two independent switches move.

**Phase 0 — observe (no governance action, ships today).**
Emit every `org.diag.*` of §4.5 as unscored, anchor-absent keys, under the **current** wall-clock cap. Add
the `FlopCounterMode` probe in measure-only mode. Record `binding_cap`, `mfu_achieved`, `spend_fraction`.
Land the cheap high-leverage fixes: warmup-as-a-fraction, manifest `env_knobs` completeness, global eval
budget. **Gate for everything downstream: two E6 baselines with measured MFU and a measured FLOPs
distribution.** No anchor bump. No chain-facing change.

**Phase 1 — recipe 2.1.0, dual cap enforced, scoring unchanged.**
`TRAIN_FLOPS_CAP = 3.0e18` enforced in the stream; `TRAIN_HOURS_CAP` 6.0 → 5.0 as the safety bound.
Scoring stays `benchmarks` (v4). Publish the `(N,D)` menu and the optimum table to `docs/external-miner/`
and the public `BaseIntelligence/prism` repo — **required**, since the budget currency is miner-facing.
This is a recipe-pin change (`recipe_pin_hex()` moves), so it is a normal recipe bump with a miner
announcement, not a scoring flip.

**Phase 2 — measure the references.** Run §5.4. Fill the v3 placeholder anchors with measured values.

**Phase 3 — pre-register v3.** Write `anchors/v3.json`; add the `include_str!` const and the `3 =>` match
arm; `LATEST_ANCHOR_VERSION = 3`; `DEFAULT_ANCHOR_VERSION` **stays 0**. The prereg hash commits on the next
scoring run. v0/v1/v2 stay byte-frozen with their own hashes. Still no live change.

**Phase 4 — confirmation tier, observed only.** Ship `eval/g9_isoflop.py`, run it on the top-5 of each
epoch, publish `org.conf.*` as unscored. Accumulate ≥ 2 epochs before considering scoring it.

**Phase 5 — governance flip (explicit, reversible).** Only when every v3 anchor has a measured (non-
`placeholder`) status: set `prism_anchor_set.status = 'active'` + `activated_at`, then
`PRISM_ANCHOR_VERSION=3`, then `PRISM_SCORING_MODE=composite`. Rows then carry `scoring_version 3`.
Reversion is a single env var.

**Ordering constraints that are not negotiable:**

1. **Emit before declare** (F9). A key in an anchor set but missing from metrics.json is a hard
   `MissingMetric` → `Ineligible` → lattice 0 for every in-flight submission.
2. **Fix the G1/G2 bootstrap clustering before calibrating anchors.** The report's Bug 3 is real: G1 and G2
   emit a single cluster per metric, so 40 % of composite weight has **zero** bootstrap variance and the
   `ci_half_width_delta` gate is vacuous on exactly the highest-weight axes. Calibrating anchors first would
   **bake the bug into the reference values.** (Another agent is fixing this in the v21 worktree; this is a
   sequencing note, not a request.)
3. **Fix the G6 censoring and direction bugs before scoring G6 more heavily.** v21's v2.json already
   reverses the `auc_log_tokens` direction and fail-closes censored runs at `1e15`; my `org.g6.*` keys
   supersede those, so v3 must not inherit either bug.
4. **Do not raise the param cap and change the currency in the same release.** They are independent; bundling
   them makes an A/B impossible to attribute.

## 6. Where I disagree

**With the user's hybrid hypothesis.** I accept the diagnosis and reject the mechanism. Tiers whose compute
allowance scales with size remove the interior optimum, make the tier boundary the new cap, are incoherent
with winner-take-all emission, and create tier-shopping as a *rational* strategy rather than an exploit
(§Option 4). The legitimate goal — a budget that does not make architecture classes structurally
un-winnable — is better served by making the currency **measured FLOPs**, which prices looping and MoE
sparsity automatically, with no class to declare and no boundary to game. That is a hybrid in effect: the
budget adapts to the architecture, just not through administrative categories.

**With the research report, on four points.**

1. **`C = 6ND` is not usable here.** At `d = 512` the `lm_head` alone is 36 % of FLOPs/token. Every
   token-budget and ladder-cost number in the report is off by 1.3–1.8× (§1.1). Correcting it moves the
   optimum *up* (133 M → 143–150 M `N_body`), so the direction of the report's conclusion survives — but
   the arithmetic must be redone before any constant is committed.
2. **The 1 B cap is not the important error.** The report treats it as "an unforced error" costing 0.23 nats.
   Both 350 M and 1 B are **non-binding** at `C_MAX`, and the 0.02-nat plateau spans 88–236 M `N_body`
   (2.7×). The cap is a memory/checkpoint parameter, not a scientific one. Publishing the optimum table is
   the whole fix.
3. **`ladder_top_bpb` is the wrong level to score.** The top rung of a fixed-`D` ladder is the most
   data-starved point; scoring it measures data limitation. Score the level at the **IsoFLOP profile
   minimum** instead (§4.2).
4. **The ladder is more expensive than stated.** The report's 47.3 min assumes constant 30 % MFU. With
   size-dependent MFU (~6 % at `d=256` rising to ~24 % at `d=1216` — an estimate, but the direction is not
   in doubt: small models are launch-bound and have poor arithmetic intensity), the same ladder is 52–58 min
   on 4 GPUs and ~$4/candidate with seeds. That is what moves the ladder off the miner's pod and into an
   operator-run confirmation tier.

**Where I agree and would go further.** The E-confound disqualifies the slope — and I showed the natural
repair (E-cancelling advantage-growth) *also* fails, on noise grounds, with a computed MDD of 0.019–0.028
against a signal of 0.013–0.065 (§4.1). The report's conclusion is right; the reason is stronger than stated.

## 7. Open questions

1. **`FlopCounterMode` coverage on the pinned image.** I confirmed the import on torch 2.13; the pod ships
   2.12. Coverage of SDPA variants and of `torch.compile`d regions needs measuring on the real image before
   `C_MAX` is trusted. **This is the one item that could invalidate the recommendation.**
2. **MFU on 4×5090 is assumed, not measured.** The 20–35 % band is the report's; my size-dependent curve is
   an estimate. Phase 0 exists to replace both with measurements.
3. **Whether the confirmation tier should feed emissions at all**, or remain a published audit record. I lean
   audit-only for at least two epochs.
4. **MoE and the total-param cap.** A 16e/2a MoE at ~1.65 B total / ~251 M active is excluded by a 1 B total
   cap while being compute-cheaper than a dense 700 M model. Raising the cap for MoE via the checkpoint
   budget is cleaner than introducing a scored "active params" number a miner could misdeclare — but it
   needs a VRAM feasibility check on 4×32 GB.
5. **Multi-GPU accounting.** v21 rents 4 GPUs but the eval battery stays pinned to GPU 0. `C_MAX` assumes
   4-GPU training; the attestation must record `n_gpu` actually used, or a single-GPU run silently gets a
   4× longer wall-clock allowance.

---

## Annex — arithmetic

All numbers reproducible from the assumptions stated inline. Key constants: RTX 5090 dense bf16 with fp32
accumulate = **209.5 TFLOPS**; 4 GPUs = **838 TFLOPS** peak; `F_tok = 6·N_body·r_eff + 6·d·V + 12·L·d·S`
with `V = 32768`, `S = 512`; `N_body = L·(4d² + 2·ffn·d²)` at `ffn = 4`; Chinchilla `E/A/B/α/β =
1.69/406.4/410.7/0.34/0.28` **used illustratively only** (Besiroglu: the original fit is not reproducible).

**Labelled estimates** (not measurements): all MFU values, including the size-dependent curve
(6 % @ `d=256` → 24 % @ `d=1216`); `$0.40`/GPU-h, anchored on the repo's `$2.5`/h/pod guard and the
`$0.97`/3-run evidence wave; PCIe all-reduce overhead ≈ 1 %; `σ_lnL ∈ [0.01, 0.08]` from PolyPythias and
the Hitchhiker's restart-variance figure; all predicted loss levels.

**Measured/verified in this pass:** the four anchor `NormKind` variants and their exact serde names; the
flat (non-nested) anchor JSON shape; `cap < reference` as the lower-better encoding; arithmetic-within /
geometric-across aggregation with zero-collapse; the metric-set asymmetry (unknown keys ignored, declared-
but-missing keys fatal); `GROUP_KEYS: [&str; 8]` blocking a JSON-only G9; the cooperative `ctx["guard"]`
and the uncaught `_CapExceeded`; `max_train_steps` never enforced; the absence of any FLOPs/MFU/active-param
notion in the harness; `tokens_seen_source` as the existing trust discriminator; the G6 report-count probe
trigger and its nats/token units; the G8 10-step no-schedule sweep; G7's unpinned dtype and cacheless
decode loop; `FlopCounterMode` availability in torch ≥ 2.0.

