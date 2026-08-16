# Appendix 15 — Incentive Mechanism and Subnet Research Productivity
> Research appendix for the Prism v3 evaluation proposal (`docs/spikes/prism-v3/`). Produced 2026-08-16 via web research + read-only repository inspection. Non-normative spike document.

**Authority:** non-normative. Per [`docs/AGENTS.md`](../../../AGENTS.md), when a
spike conflicts with a frozen spec ([`docs/PRISM.md`](../../../PRISM.md),
[`docs/BUNDLE_SPEC.md`](../../../BUNDLE_SPEC.md), the pre-registered anchor sets)
**the normative doc wins**. Nothing below is a scoring or emission contract.

**What shipped from this appendix, and what did not.** The significance-gated
emission rule described in §7.1 is implemented in `crates/prism-competition`
(`paired.rs`, `frontier.rs`, `sig.rs`, `rerun.rs`) and selected by
`PRISM_EMISSION_MODE=sig`. It is **default-off**, and the normative statement
of the rule plus its sequencing constraint lives in
[`docs/PRISM.md`](../../../PRISM.md) § Significance-gated emission — read that,
not this file, for what the system does.

Two of this appendix's own conclusions are load-bearing for why it ships off:

- §1.3c / §9.1 risk 1 — the bootstrap measures **eval-item variance only**, never
  training-seed variance, so the lower bound is overconfident. A significance
  test on a provably wrong standard error is worse than honest WTA.
- §7.3 — the sequencing (fix clustering → measure `σ_seed` → stage the private
  tier → *then* enable the collapse) is not negotiable, and step 2 is not yet done.

**Errata against this branch.** §1.2 reports `top3` and the opt-in owner-arch
credit as absent, having been produced against `/root/gbase`. On the
`prism-v2.1-scoring` branch both **exist**: `EmissionMode::Top3Decay` and
`PRISM_OWNER_ARCH_CREDIT_BPS` are implemented and default-off. The legacy
`OWNER_ARCH_CREDIT_ENABLED` constant is still `false` and stays that way — a
different mechanism from the opt-in split, as §5.4 recommends. The 4×5090 /
1 B-param figures in §1.2 also reflect the older checkout.

**Original front matter follows.**

**Auteur:** agent de conception de mécanismes · **Date:** 2026-08-16
**Statut:** document de conception. Recherche en ligne + lecture **en lecture seule** de `/root/gbase` (aucune modification, aucune commande git d'état).
**Entrées prises comme données:** `/tmp/prism-scaling-research/REPORT.md` (contraintes scientifiques), `docs/PRISM.md`, `docs/THREAT_MODEL.md`, `crates/prism-registry/src/`, `docs/spikes/prism-v3/research/12-score-aggregation.md`.

**Convention de preuve, appliquée partout:**

| Marque | Sens |
|---|---|
| **[E]** | **Évidencé** — mesuré/publié, avec source et chiffre |
| **[J]** | **Jugement de conception** — mon raisonnement à partir de [E], falsifiable mais non mesuré |
| **[S]** | **Spéculatif** — plausible, non étayé ; à traiter comme hypothèse à tester |

---

## Résumé exécutif (FR)

1. **Le WTA est le mauvais choix par défaut — et l'argument décisif n'est pas « l'équité », c'est la valeur espérée d'une copie.** Sous WTA sur estimations ponctuelles, un plagiat du champion a une performance *vraie* identique, donc gagne le pot entier avec probabilité ≈ 0,5 par symétrie du bruit : espérance ≈ **50 % du pot pour le coût d'un pod (~5–15 $)**. Tout le mécanisme repose alors sur le détecteur de copie, que la littérature adverse montre contournable. Une règle de **significativité** ramène cette espérance sous **5 %** *mécaniquement*, sans détecteur. **Nuance imposée par l'expérience de SN9 : epsilon a borné la copie au sommet mais les copies ont quand même peuplé le classement (« copie directe sans pratiquement aucune modification », mesuré au microscope de poids). La significativité protège la part du champion, pas la bande graduée** — donc amélioration majeure, pas élimination. **[E→J]**
2. **L'argument empirique le plus net : le classement à basse fidélité échoue précisément là où le WTA opère.** Deux régimes mesurés **[E]** : pour *écarter* (ρ ≈ 0,6 après quelques époques ; ρ = 0,851 à 5 % du budget ; ~70 % des mauvais restent mauvais) la basse fidélité **informe réellement** ; pour *désigner le vainqueur* — τ de Kendall restreint au **top-10 sur 3 000** — elle est **quasi inutile**, car les meilleurs ont des performances vraies très proches. **Le WTA de Prism se situe entièrement dans le régime défaillant ; la bande graduée et l'entonnoir à deux étages exploitent le régime qui fonctionne.**
3. **Le seul précédent réellement comparable confirme le diagnostic.** Bittensor SN9 applique WTA à **95 %+** (`temperature = 0.01` : « ~96 % au meilleur modèle, ~3 seulement reçoivent du poids ») avec un **seuil epsilon** (ϵ = 0,5 % ; expérience 7B★ à 0,1 % le 2024-08-08) introduit *explicitement* parce que « télécharger le meilleur modèle et le modifier marginalement suffirait à truquer le score ». Ses concepteurs qualifient ensuite le WTA de **défaut de conception** provoquant la **rétention de modèles**, et y répondent par un **epsilon décroissant**. **IOTA (arXiv 2507.17766, 2025) abandonne le WTA en nommant deux échecs : le coût en capital par mineur et le hoarding.** **[E]**
4. **Correction assumée : la théorie des contests penche en réalité *pour* la concentration, et mon premier brouillon appliquait mal Drugov–Ryvkin.** Le choc dont la queue détermine la structure optimale des prix est le **terme de mesure/chance** qui corrompt la relation effort→rang, **pas** l'hétérogénéité de la fonction de production. Le bruit de mesure de Prism (graine, échantillonnage d'items, binomiale à 200 items) est **à queue légère / IFR** ⇒ **D–R favorise le WTA**. Ce qui subsiste en faveur d'un partage : (a) le **coût d'effort convexe** admet plusieurs prix (Moldovanu–Sela) ; (b) si Prism est une **course à l'innovation valorisant la diversité** plutôt qu'un tournoi de rang, le résultat « n optimal = 2 » **ne transfère pas** (Terwiesch & Xu) ; (c) surtout, la **marge de participation est endogène** — ce que la théorie tient fixe est précisément ce que le WTA détruit. **[E→J]** Ma recommandation tient, mais **sur d'autres fondements** (§3.5).
5. **La règle recommandée existe déjà en production ailleurs — et corrige ma propre paramétrisation.** Le **SN56 (Gradients)**, qui comme Prism fait *soumettre du code exécuté par l'opérateur sur GPU égalisés*, déploie exactement ce mécanisme : comparaison **appariée exemple par exemple** sur la *même* tranche, **borne inférieure bootstrap** (99 %, 10 000 rééchantillonnages, **graine fixe** pour que deux vérificateurs concordent), **zone morte** de 0,01 nats, **taux de victoire ≥ 0,55**, **écart moyen minimal**, et **décroissance du titre** (0,165 %/jour). Leur commentaire de code corrige mon erreur : **une marge relative est la mauvaise échelle pour une log-vraisemblance** — « D nats sont D nats de preuve que la perte vaille 0,02 ou 2,0 » — et elle s'effondre précisément là où la métrique sature. Et **ne pas monter la barre trop haut** : au-delà de ~0,55, on sélectionne **les soumissions à faible variance plutôt que les bonnes**. **[E]**
6. **Recommandation : « champion significatif + bande graduée + poche d'exploration », avec deux barres.** Champion **60 %** (dégradé vers un plancher, le reste **brûlé**, si le gain est réel mais marginal — structure à deux barres de SN56) ; rangs 2–4 : **15/10/5 %** ; **10 %** à ≤5 soumissions passant les portes et **avançant la frontière d'au moins un axe**. Le terme statistique **ne décroît jamais** ; seul le plancher économique décroît. Ajouts déjà déployés ailleurs : **EMA sur le vecteur de poids** (SN9/SN37) et **plancher zéroant la queue** (SN37).
7. **La résolution honnête de la batterie est de 3–4 rangs, pas 1 et pas 10.** Avec σ de graine ≈ 0,3–2,5 % relatif et des écarts architecturaux réels de 1–5 %, la différence minimale détectable (une queue, α = 0,05) est **2,33·σ ≈ 2,3 %**. Payer le rang 1 seul jette de l'information ; payer 10 rangs paie du bruit. **[J]** Corroboration : le WTA « réel » de SN9 paie en fait ~3 modèles.
8. **Faille statistique non identifiée dans le dépôt : `SE(C)` mesure la mauvaise variance.** Le bootstrap ne capture que la variance **des items d'évaluation**, jamais celle **de graine/entraînement** — irréductible car chaque soumission n'est entraînée qu'**une fois**. La LCB est donc surconfiante *même après* correction du bug de clustering G1/G2 (40 % du poids à variance nulle, vérifié dans le code). **Preuve directe que ce n'est pas un détail : un simple changement de graine fait tomber le τ de Kendall à 0,48 sur 32 architectures ; un changement de profondeur, à 0,54. [E]** Aucune règle de significativité n'est honnête sans **répliques de graine**. Prérequis n°1.
9. **Le crédit au propriétaire d'architecture doit rester désactivé.** tea.xyz a été farmé jusqu'à **>150 000 paquets** de spam parce que la récompense circulait sur un graphe dont les arêtes étaient déclarées par les bénéficiaires. Prism est structurellement du côté de tea.xyz (jeton spéculatif), pas de thanks.dev (budget fixe). `OWNER_ARCH_CREDIT_ENABLED = false` est le bon réglage. **Crédit à la conception existante : les arêtes du registre Prism sont attestées par l'opérateur (publication seulement après score mesuré), ce qui ferme le vecteur de nœuds fictifs — mais pas l'auto-transaction.** **[E]**
10. **Récompenser la nouveauté *mesurée* est un piège ; récompenser la contribution *marginale* fonctionne.** Numerai a commercialisé « being different pays » mais n'a **jamais** payé la dissimilarité brute. Le bonus exploité (>100 NMR, 85 % de rendement en <6 mois) était le bonus de *classement*, pas la métrique de contribution. TC a été retiré en 2024 pour **opacité et non-vérifiabilité locale**. Transposition pour Prism : **archive d'élites par axe** (une soumission 3ᵉ au composite mais 1ʳᵉ en G3 ou G7 a produit de l'information) — infalsifiable par renommage, contrairement à une distance AST. **Cas concret : sous WTA, l'architecture bouclée que Prism veut tester gagnerait G3/G4/G5, perdrait G7, et toucherait zéro.** **[E→J]**
11. **Trois ajouts anti-Goodhart à plus haute valeur :** (a) **palier privé obligatoire et rotatif** — la pénalité de miroir est aujourd'hui **inerte par construction** en `public_dev` (vérifié dans `rollup.py`) ; (b) **re-run du champion** sur la tranche privée fraîche, **à date non annoncée** (raison donnée par IOTA : un mineur ne doit pas savoir quand il est observé) ; (c) **sortir du leaf mono-groupe v4**. Quatrième, presque gratuit : **faire tourner une baseline de référence chaque round** — la conclusion la plus reproductible de la littérature NAS est que les variations de protocole et de graine produisent des effets **du même ordre** que les effets architecturaux récompensés.
12. **Deux nuances que je refuse de masquer.** (a) Roelofs et al. (>100 compétitions Kaggle) trouvent **peu de preuves** de surapprentissage adaptatif (effets < 1 pt) ; mais leurs *exceptions* sont les jeux de test **effectivement petits** — **n = 200 place Prism dans la catégorie exceptionnelle**. (b) Plus important : **la menace dominante de Prism n'est pas le surapprentissage adverse, c'est la validité de mesure.** Si le budget est contraint, **taille de batterie et répliques passent avant l'anti-adaptivité**. Corollaire contre-intuitif du Ladder : l'erreur ne dépend du nombre de soumissions que **logarithmiquement**, alors que la taille du jeu d'éval borne en `n^{−1/3}` — donc **élargir l'entonnoir est presque gratuit, réduire la batterie est coûteux**. Le verrou 1-max est une défense **anti-sybil/best-of-n**, pas anti-adaptivité.
13. **Écart entre le brief et le dépôt, à trancher avant toute décision :** `top3` et le crédit-propriétaire opt-in décrits comme « v2.1 » **n'existent pas** dans `/root/gbase` — WTA y est câblé et `OWNER_ARCH_CREDIT_ENABLED` est une **constante `false`** avec un test qui casse la compilation si on la retourne, pas un bouton. Je traite v2.1 comme la **cible proposée** (§1.2).
14. **Limite à publier, pas à cacher :** Tay et al. (>100 modèles, 15 M→40 B) mesurent que **le meilleur modèle change selon l'échelle** — le Transformer vanille a le *meilleur exposant* sans être le meilleur à chaque point de calcul, et ALBERT (partage de poids inter-couches, même famille que les architectures bouclées) **régresse** en aval quand le calcul augmente. Prism classe des architectures **à budget fixe et petit** ; il ne peut pas prétendre sélectionner ce qui passe à l'échelle. Le dire explicitement **augmente** la crédibilité auprès du public visé. **[E]**
15. **Décision demandée :** corriger la variance (bug de clustering + répliques de graine) **avant** d'activer toute règle de significativité. Une règle de significativité bâtie sur un SE faux est **pire** que le WTA honnête — elle confère une fausse autorité statistique à un classement biaisé.

---

## 1. Scope, method, and the gap between the brief and the repository

### 1.1 What I read

**Repository (read-only, no edits, no state-changing git):** `AGENTS.md`; `docs/PRISM.md`; `docs/THREAT_MODEL.md`; `docs/AGENTS.md`; `crates/prism-registry/src/{competition.rs, hooks.rs, lib.rs, publish.rs, weights.rs}`; `crates/prism-pipeline/src/composite.rs`; `crates/prism-emit/src/lib.rs`; `crates/prism-recipe/src/lib.rs`; harness `eval/{g2_downstream.py, rollup.py, common.py}`; `docs/spikes/prism-v3/research/12-score-aggregation.md` and `09-miner-metrics-leaderboards.md`.

**Not touched:** the worktree `/root/gbase-v21` (another agent holds it), per instruction.

**Research input taken as given:** `/tmp/prism-scaling-research/REPORT.md`.

### 1.2 The v2.1 gap — state it before reasoning on it

The brief describes v2.1 as adding "opt-in `top3` decay (100%/50%/25%)" and "an opt-in owner-architecture credit". **Neither exists in `/root/gbase`.** Verified:

| Brief element | Verified state in `/root/gbase` |
|---|---|
| Opt-in `top3` decay | **Absent.** `rg` over `crates/` finds no `top3` / emission-decay path. `prism_registry::apply_wta` is unconditional: argmax keeps its score, **every other positive score is set to 0**. |
| Opt-in owner-arch credit | **Not a knob.** `OWNER_ARCH_CREDIT_ENABLED` is a `pub const bool = false` with a `const{assert!}` test (`owner_arch_credit_temporarily_disabled`) that fails the build if flipped without editing the test. Enabling it is a code change, not configuration. |
| v3/v4 composite | **Present and consistent.** Group weights G1..G8 = 0.25/0.15/0.10/0.15/0.15/0.075/0.075/0.05; live default is `benchmarks` (**v4**, equal-weight mean of available G2 accuracies), *not* the composite. |
| WTA by default | **Confirmed**, and stronger than "default": it is the only implemented collapse. |
| 4×RTX 5090 | Repo pins **1× RTX 5090** ("non-5090 / multi-GPU rejected at rent"). |
| Param cap | `MAX_PARAMS = 350_000_000`; `TRAIN_HOURS_CAP = 6.0`. |

**Methodological consequence.** I treat `top3` / owner-credit / 4×5090 as the **proposed target** and the above as the **verified base**. Every recommendation says which it addresses. This matters for one specific conclusion: **the owner-credit mechanism is currently correctly disabled, and my recommendation is to keep it disabled** — so the "opt-in v2.1 owner credit" described in the brief is a change I argue *against* shipping (§5.4), not a feature I assess as present.

One further note on `apply_wta`'s tie-break. Ties resolve to the **lexicographically smallest hotkey**. Under a noisy metric that is quantized to an integer lattice (`round(SCORE_MAX × …)`), exact ties are not exotic — and a lexicographic tie-break is *grindable*: a miner can cheaply generate hotkeys until one sorts low. With WTA this converts a coin-flip into a **deterministic win** for whoever bothered to grind a key. **[J]** The `competition.rs` doc-comment shows awareness of the adjacent risk (an off-metagraph arch owner stealing the share via lex-tie) but the submitter-side grind remains. Under the graded rule I recommend, tie-break value collapses because adjacent ranks pay similarly.

### 1.3 Three verified facts about the current scoring path that constrain everything downstream

**(a) 40 % of composite weight carries zero bootstrap variance — verified independently.** `composite.rs` resamples `MetricSeries::clusters`; resampling a single-element map returns that element, so variance is 0. In `g2_downstream.py` every item of a task is recorded with the **constant** cluster `f"g2/{task}"` (line 55), and `rollup.py::_cluster_means` buckets by that id — one cluster per metric. G1 domains likewise map through `_ITEM_CLUSTERS` with a per-domain (not per-doc) id. G3/G4/G5 do have real cluster variety (`it["cluster"]`, `f"{probe}@{L}"`, `slot{j}`). So G1 (0.25) + G2 (0.15) contribute no variance, `SE(C)` is understated, and the `ci_half_width_delta = 0.05` gate is vacuous on the two heaviest axes. The mirror path (`build_mirrors` → `clusters[f"{gen}#{i}"]`) is *correct*, which is good evidence this is an oversight.

**(b) The mirror-gap penalty is inert in the default tier — by construction and honestly labelled.** `rollup.py::build_mirrors`: in `public_dev`, when no private asset path differs, `mirror = dict(public)` with copied clusters — gap ≡ 0. The docstring says so ("the run is its own mirror"). G4's mirror does use the resolved secret seed, so it is real *if* a secret seed is staged. **Operators must not read "mirror penalty = 0" as evidence of no contamination.**

**(c) `SE(C)` measures eval-item variance only — never training variance.** This is my own finding and it is the most consequential one in this document. Each submission is trained **exactly once** on one pod. The bootstrap resamples *eval items*, so it estimates "how much would this score move on a different eval draw *of this same trained checkpoint*". It cannot and does not estimate "how much would this score move if the same architecture+recipe were trained again with a different seed". The literature puts that second quantity at **up to 3.5 % relative on loss** (restart variance, per the Hitchhiker's guide) and PolyPythias measures cross-seed spread over 50 runs. **[E]** Consequence: **LCB ranking is overconfident about architecture quality even with (a) fixed**, because the dominant nuisance term is absent from the variance model. Any significance rule must add a **seed-variance floor** measured by replication, not derived from the bootstrap (§4.3).

---

## 2. Q1 — Landscape: how comparable systems reward research under noisy, gameable evaluation

### 2.1 Bittensor and decentralized-training networks

**Subnet 9 (Taoverse/Macrocosmos pretraining) — the closest precedent, and the most instructive.** From the Macrocosmos pretraining whitepaper **[E]**:

- **Scoring**: pairwise win-rate on losses over randomly sampled pages of the pinned corpus (originally Falcon Refined Web, later moved to **FineWeb-Edu** after finding the original dataset was "throttling miner performance"). Miners publish to HuggingFace and commit metadata on-chain; validators download and continuously evaluate.
- **The win rule, verbatim from the paper**: `isWin(L_ik, L_jk, b_i, b_j) = L_ik < L_jk` if `i = j`, else `(1 − ϵ)·L_ik < L_jk` — where `b` are the **upload blocks** and ϵ "acts to increase rewards for earlier models". So SN9 combines **(i) an epsilon margin** and **(ii) an explicit earlier-block incumbency advantage**.
- **Why epsilon exists, verbatim**: "*This is necessary given that all models are publicly available in Hugging Face, and without imposing a minimum improvement threshold, downloading the top performing model and minimally tweaking it would be enough to game the scoring strategy.*" **This is exactly Prism's copy-with-tweak problem, and the deployed answer is a margin rule, not a detector.**
- **Emission collapse**: WTA — "The top performing model receives most (**95%+**) of the miners' emissions on the subnet, while every other model receives almost nothing." Rationale given: demand for AI is concentrated on leading models, so "intense competition is more valuable than diversification".
- **Parameters**: ϵ = **0.5 %** at time of writing; a parallel competition **7B★** launched **2024-08-08** with ϵ = **0.1 %** as a head-to-head experiment to find the optimum; **"dynamic epsilon"** (decaying over time) slated for release end of Aug 2024, explicitly to stop competitions stagnating.
- **The documented failure of their own WTA**: §3.4.2 **Model Hoarding** — "*the top miner already has an even better pretrained model, but the subnet's current incentive does not encourage them to publish it. This misalignment… reveals a design flaw*". Their fix is the decaying epsilon. Separately §3.3.3 concedes the equilibrium strategy is to hold "**multiple unsubmitted models at progressively higher performance levels**".
- **Anti-collaboration attacks they document [E]**: miners rescaling weights before normalization layers ("vanishing gradients") to preserve inference quality while sabotaging anyone who resumes training from the published checkpoint. A defence-relevant reminder that publishing artifacts invites *poisoned* artifacts.
- **Measured copying and padding — "monsters and vampires". [E]** Macrocosmos built a weight-visualisation tool and published what it found in the 14B and 3B competitions: "*we found that **many of them were engaging in model-copying**… In some instances, you cannot even see the differences between them… there has been **direct copying with practically zero amendments**… especially… in the 3B competition*". On size-floor gaming: "*some miners appear to be **copying models that are parameter padding, and then further padding their variation**.*"

  **Two consequences, and the first is a direct qualification of my own recommendation.**
  1. **Copying happened at scale *despite* the epsilon rule.** Epsilon makes copying unprofitable **for the top slot**; it does not stop copies **populating the leaderboard**. It bounds the damage, not the behaviour. I return to this in §3.3c and §7.2 because it is the sharpest available critique of a graded band: **if copies occupy ranks 2–4, a graded band pays them.**
  2. **Any structural constraint is satisfied in the cheapest way that passes the check.** SN9's tight parameter bands (`min 13.7 B`, `max 13.9 B`) were met by **padding with dead parameters**. Prism's `MAX_PARAMS = 350_000_000` is a *ceiling* rather than a band, so padding buys nothing — but the general lesson applies to every gate Prism adds. **[E→J]**
- **The collapse is softer than "WTA" implies, in two ways worth copying. [E]** SN9's weights come from a low-temperature softmax over win rate — `temperature = 0.01`, with the code comment "*0.01 gives ~96 % to best model with only ~3 receiving any weights*" — plus an EMA on validator weights (`alpha = 0.5`). SN37 uses `ALPHA = 0.90` with `MIN_WEIGHT_THRESHOLD = 0.18` and comments that new winners accrue weight over ~540 blocks while previous winners phase out over ~2970. So in practice:
  - **A slow EMA smooths handover**, so one lucky evaluation cannot flip emission instantly — a *temporal* analogue of my significance requirement, and a cheap complement to it.
  - **A floor zeroes the tail**, keeping validators in tight agreement (protecting `vtrust`) instead of each expressing its own noisy ranking of also-rans. Prism does not need this for consensus reasons (single sealed master, §2.1) but the same floor usefully prevents paying for unresolvable rank differences.
  - Note that "~3 receiving any weights" means SN9's *realised* collapse is closer to a steep top-3 than to literal WTA — which is mild empirical support for the band depth I recommend in §7.1. **[E→J]**
- **The anti-sybil argument for WTA is sound and I should concede it. [E]** Macrocosmos: "*There is no economy of scale for individual actors who run multiple miners, as only a single model can win at once.*" That is correct as far as it goes — farming hotkeys buys nothing when only rank 1 pays. **Any graded scheme reopens this**, and it is the strongest argument against my recommendation. Prism's mitigations are stronger than SN9's, though: **one accepted submission per `(prism, hotkey)`** enforced in `submission_gating`, metagraph membership required at intake, and precheck quota scoped to **coldkey** rather than hotkey. Combined with a miner-funded pod per submission, the cost of occupying multiple band slots is real rather than nominal. **[J]**
- **Their overfitting stance [E]**: they argue miners *aren't* overfitting the pinned corpus because models still do well on unrelated benchmarks — an argument that **weakens as the eval set shrinks**, which is Prism's regime (§4.1).

**IOTA (SN9's successor; mainnet 2025-06-02) [E].** From *IOTA: A Technical Primer for Release* (arXiv 2507.17766, Jul 2025): SN9 "validated blockchain-based decentralized pretraining as viable, [but] **it contained core issues: (i) every miner had to fit an entire model locally, and (ii) 'winner-takes-all' rewards encouraged model hoarding.**" IOTA replaces the competition with pipeline-parallel cooperative training, rewarding **backward passes successfully processed**, verified by **partial recomputation with cosine similarity** and — the good part — **unannounced monitoring**: "miners are not aware of when they are being monitored, preventing them from selectively behaving correctly only during observed intervals."

**This is the single most important landscape datapoint: the operator with the most experience running a WTA model-competition subnet abandoned WTA in a peer-reviewable paper and named hoarding as one of two reasons.**

Three details that transfer to Prism:

- **The other named failure is per-miner capital cost**, which an external review associates with SN9 being capped at roughly **6 miners** **[3P — unverified count, directionally consistent with the mechanism]**. This is the **participation-margin** problem I argue is missing from contest theory (§3.3a), showing up in the field: WTA on whole artifacts means every entrant funds a full training run for a lottery ticket, and entry collapses. Prism has the same structure (miner-funded pods, whole-artifact competition), so it should expect the same pressure. **[E→J]**
- **Unannounced verification is a mechanism Prism should copy directly.** The champion re-run I propose in §4.3 #2 is stronger if its *timing* is unpredictable rather than every-round-on-schedule, for exactly IOTA's stated reason.
- **Shapley-style attribution (CLASP) was designed and explicitly *not shipped***: "CLASP is not included in the initial release; it remains an active area of research." Its published evidence is a **toy simulation** (5 layers × 5 miners, assumed 10 % loss inflation for bad actors). **Any claim that SN9 pays via Shapley values is unsupported.** This independently corroborates the "Shapley is a trap at this budget" verdict in §8 — the team best placed to ship it, in a setting with far cheaper per-unit evaluation than Prism's, did not.

**Subnet 37 (Taoverse/Macrocosmos finetuning) [E]** shares SN9's codebase (`iswin`/epsilon/WTA carry over) and adds two mechanisms worth importing:

- **Competitions with pre-committed sunsets written into code** — e.g. `SUNSET_B7_BLOCK = 4_675_163`, `SUNSET_INSTRUCT_8B_BLOCK = 5_158_632` ("23:59 GMT+0 Tuesday, March 18, 2025"), with `reward_percentage` splits that shift at a future block. **This is a deployed version of the "saturation tripwire" I recommend in §4.2**: a saturated or contaminated task is retired by a pre-announced, publicly auditable schedule rather than by emergency intervention. Strictly better than discretionary reweighting, and it composes with Prism's existing anchor pre-registration.
- **Bounded normalization for heterogeneous metrics** — `NormalizationId.INVERSE_EXPONENTIAL` with an explicit `ceiling` maps unbounded loss onto a bounded score so one catastrophic sub-task cannot dominate a weighted sum. Prism achieves the same end differently (clipped fixed-anchor normalization to [0,1]), so this is confirmation rather than a gap.
- **SN37's epsilon is ~10× SN9's, and the reason is sample size**: its eval tasks draw **120–150 examples**. **That is the same regime as Prism's 200 items/task, and it is direct evidence that a small eval set forces a larger margin** — i.e. Prism should expect its own noise-calibrated margin to be *wide* until the battery grows (§4.3 #3). **[E→J]**
- SN37 also forces a shared tokenizer (`Xenova/gpt-4`) in most competitions so losses are comparable across submissions. **Prism solves this better**: tokenizer-neutral `bits_per_byte` lets miners bring their own tokenizer without breaking comparability — a genuine design advantage, and the input report's warning against the tokenizer-dependent shadow bpb leaf is the same point.

**SN9 never thresholds the noisy scalar directly — a deployed precedent for the paired test. [E]** The pipeline is: per-batch loss `L_ij` for model `i` on batch `j` → for every model pair on every batch, a **pairwise win/loss indicator** via `iswin(...)` → sum wins → **win rate** → weights via a low-temperature softmax. The noisy per-batch loss is reduced to a **binary comparison on each shared batch**, then aggregated over many batches and many opponents.

This is the same statistical idea as the parameter-free Ladder and as my §7.1 recommendation: **compare submissions on shared items and aggregate the comparisons, rather than comparing independently-estimated levels.** Three independent systems converged on it — SN9's per-batch pairwise wins, the Ladder's paired significance test, and the clustered-SE literature's paired-difference recommendation **[E]**. **Prism's LCB does the less robust thing.** **[E→J]**

**Transferable, and what it costs.** The epsilon rule is proven and cheap. But note its weakness relative to what I recommend: SN9's ϵ is a **fixed relative fraction**, not a function of measurement uncertainty. A fixed 0.5 % is simultaneously *too small* on a noisy axis (it will crown noise) and *too large* on a quiet one (it will suppress real gains). Prism can do better because it already computes a bootstrap distribution — it can make the margin **noise-calibrated** rather than a constant, and the parameter-free Ladder is the precedent for a self-scaling margin (§2.4, §7.1). **[J]**

### 2.1b Subnet 56 (Gradients / Rayon Labs) — a live deployment of the rule I recommend

**This is the most important precedent in this document, and I found it late enough that it functions as an independent check rather than an input.** SN56 moved in July 2025 from "miners return a model" to **miners submit a training repository + exact commit SHA; validators execute it in isolated containers on validator-provided GPUs** — structurally the same shape as Prism (submit code, operator runs it, pinned recipe). **[E]**

Consequences they state, all of which apply to Prism:

- **Compute is equalized**, so the ranking is over *methods*, not budgets — this directly addresses the per-miner capital-cost failure that IOTA named (§2.1).
- **Containers have no internet access**, closing exfiltration and eval-lookup channels. (Prism's equivalent is `unshare --net` netns isolation.)
- **The winning script is published open-source each tournament**, "resetting the baseline for everyone". An external reviewer notes the tension — publishing hands rivals the best recipe **[3P]** — but it is safe *because* champion protection exists, and because everyone re-runs on equal hardware. **Prism's top-model publish to GitHub/HF is the same bet, and this is evidence the bet is survivable.**
- **Paid entry** as an anti-sybil device: `TOURNAMENT_TEXT_PARTICIPATION_FEE_RAO = 350_000_000` (0.35 TAO), env 0.3, image 0.2, raised by dated commit (2026-06-19). Prism's analogue already exists implicitly — the **miner-funded pod** is a per-attempt price.

**The incumbent/challenger rule ("boss round"), live parameters [E]:**

```python
BOSS_ROUND_TIE_DEADZONE_NATS   = 0.01
BOSS_ROUND_MIN_WIN_RATE        = 0.55
BOSS_ROUND_MIN_MEAN_GAP_NATS   = 0.01
BOSS_ROUND_BOOTSTRAP_CONFIDENCE = 0.99
BOSS_ROUND_BOOTSTRAP_RESAMPLES  = 10_000
BOSS_ROUND_BOOTSTRAP_SEED       = 20260808  # "Fixed so two validators scoring the same boss round always agree."
```

Their in-code comments are the clearest statement of the noisy-evaluation problem I found in **any** source, and **one of them corrects my own parameterization**:

> "A relative margin is the wrong scale for a log-likelihood loss: **a difference of D nats is D nats of evidence whether the loss is 0.02 or 2.0**, so `abs(boss) * 1%` collapses to nothing exactly where the task is most saturated. Worse, **a scalar mean carries no information about its own uncertainty, so no threshold on it can separate a real win from held-out sampling noise.** Instead both models are scored on the identical held-out set and **compared example by example. Example difficulty dominates the variance in both losses and cancels in the pairing.**"

The mechanism is a **paired per-example comparison with a bootstrap lower bound**, gated by three complementary conditions:

1. **Dead zone** — ignore per-example differences below 0.01 nats: "*without a dead zone the win count is decided at the 5th decimal.*"
2. **Win rate ≥ 55 % of *decided* examples**, at a one-sided **99 %** bootstrap lower bound over 10 k resamples.
3. **Mean gap ≥ 0.01 nats**, "*so it cannot win on a majority of hairline examples while being materially worse where it loses.*"
4. A **minimum count of decided examples** — saturated tasks fail naturally, and "*the gate is automatically stricter when it has less to go on.*"

**Four things here are sharper than my own reasoning, and I am adopting all four (§7):**

- **Margins on a log-loss metric must be absolute (nats), not relative (%).** This is a direct correction to the `ε₀ = 1 % relative` I had proposed.
- **Why 55 % and not higher** — "*A genuinely better model with a wide per-example spread can sit near 55 % and still be the better model; **demanding much more of it selects for low-variance submissions rather than good ones.**" I had not considered that an over-strict displacement bar has a *selection* effect, not merely a conservatism cost. It is the mirror image of the variance-farming problem: too loose selects for variance, too tight selects against it.
- **Calibrate the bar against the measured separation distribution.** They calibrated against **62 boss-round text tasks over 120 days, median separation 0.0094 nats**, and found that at a 0.02 threshold "*only a quarter of matchups were close enough to even be winnable, which compounded over 4-of-5 tasks makes the crown near-permanent.*" **This is the incumbent-squatting failure of §9.1 risk 2, measured in production, with the fix being to lower the bar to the observed median.** They also log the win rate on every task, won or lost, so the parameter can be recalibrated against data — and mark it "Provisional".
- **A documented false negative that motivated the mean-gap floor:** a larger second threshold "*is what made a model better by 0.011 nats on ALL 800 samples — 100 % win rate — lose the task.*" A cautionary example of a well-intentioned margin rejecting a real improvement.

**Two-bar emission, which is better than my single threshold [E]:** the base pool is floored at 50 % of subnet emission; **a champion must beat the boss by ≥10 % to earn *above* the floor and reduce burn**. So there is one bar to *hold the crown* and a higher bar to *earn premium emission*, with the excess **burned**. Plus `RUNNER_UP_EMISSION_DAYS = 7` and **champion time decay of 0.165 %/day** (halved from 0.33 %/day by the same commit) — tenure erodes the incumbent's advantage, applied to *emission* rather than to the comparison threshold. Note the direction of their revision: they **disabled** progressive per-task dethrone thresholds (`EXPONENTIAL_*_THRESHOLD → 0.0`) in favour of straight paired score comparisons, i.e. **they moved complexity out of the threshold and into the statistics.**

**Anti-gaming devices worth importing [E]:** `MAX_NEAR_DUPLICATE_RATE = 0.20` — reject an eval task whose dataset near-duplicate rate reaches this fraction, the most concrete anti-contamination parameter in this survey; a dedicated **obfuscation-detection** module (recall SN9's weight-obfuscation problem); and **LLM-based submission dedup** (`TOURN_DEDUP_ENABLED`, Claude Opus, $15 budget, plus a repo-diff reviewer) — **they pay an LLM to catch copy-and-tweak**, confirming that a margin rule alone does not remove the need for a detector (§3.3c). Also `trust_remote_code` **off** during eval, which constrains architecture novelty — a tradeoff Prism deliberately makes the other way, since novel architectures are the point.

**Why this matters for confidence in §7:** an independently-designed, live, economically-adversarial system converged on **paired per-example comparison + bootstrap lower bound + dead zone + tenure decay + burn of the unallocated remainder**. That is the mechanism I recommend, arrived at from different premises. Where we differ, they have production calibration data and I do not, so I defer to them on parameterization.

### 2.1c Templar (SN3) and Yuma-level concerns

**Templar's Gauntlet [E]** converts a `LossScore` into an **OpenSkill rating** — an explicit rank-rating fix for noisy per-round measurement, the same "don't threshold the raw scalar" family as SN9's pairwise wins and SN56's paired comparison. It also uses **superlinear reward to deter sybils** and, notably, a **Proof-of-Computation** step naming copying as an explicit threat — i.e. a *separate* anti-copy mechanism alongside the rating system, again confirming that margin/rating rules do not subsume copy detection.

**Yuma consensus / weight-copying.** Validator weight-copying is a documented Bittensor-wide problem, mitigated by commit-reveal and liquid alpha; the research notes a documented **CR3 cryptographic break (Aug 2025)** **[E]**, which is a reminder that consensus-layer mitigations are themselves attack surface. For Prism this is **largely out of scope by architecture**: weights originate at the single master gateway and are **sealed**; validators fetch sealed weights rather than independently scoring (`AGENTS.md`, `PRISM.md`). So a graded emission vector does not create validator-divergence risk the way it would in a subnet where each validator scores independently. **[J]** That removes what would otherwise be the strongest *technical* objection to moving off WTA.

**Other networks, briefly.** Prime Intellect (INTELLECT-1/2) demonstrates decentralized *training* but is not a scored competition of rival architectures. Gensyn's verification line (proof-of-learning, refereed delegation/Verde) addresses "did you actually do the compute", which Prism solves differently (operator-run pods + signed receipts + telemetry contract). Neither supplies an emission-collapse precedent.

### 2.2 Numerai — the strongest real-money precedent for paying for a noisy signal

Full history, all **[E]** unless marked:

| Phase | Mechanism | Outcome |
|---|---|---|
| 2019 | Marketed "**being different pays**" — paid for "originality and uniqueness" | Implemented **not** as similarity-distance but as **Meta Model Contribution (MMC)** |
| 2019–20 | **MMC1** = literal leave-one-out on the stake-weighted meta model; **MMC2** = residualization (neutralize submission against SWMM, covariance with target) | LOO replaced by residualization for stability/interpretability |
| 2020 | **Leaderboard bonus exploited** — accounts `Madmax`/`Madmin`/`The_Guy`, 40 NMR each → 222 NMR combined; "**over 100 NMR and 85 % returns in less than 6 months**" via a hedged multi-account straddle | **Bonus removed** 2020-09-09. Operator statement: "*We see it as our responsibility to make not-exploitable payout systems.*" |
| 2022 | **True Contribution (TC)** — gradient of optimized-portfolio return w.r.t. stake, via `cvxpylayers` | Opt-in; MMC staking discontinued 2022-04-09 |
| 2024-01-02 | **TC retired, back to MMC**, made **mandatory** (0.5×CORR + 2×MMC) | Stated reasons: TC is "**blackbox**", "**tied to certain optimizer settings**" that "**change from time to time**", needs "constant alterations" |
| 2026-01-01 | CORR 0.5→**0.75**, MMC 2.0→**2.25**; new target Ender20 | Dec 2025 payouts $254,137 in NMR |

**Four lessons, and they are the backbone of my Q4 answer:**

1. **Nobody ever successfully paid for measured originality.** "Different" only paid when it was **orthogonal *and* predictive**. Paying for difference alone is not a mechanism anyone has made work. **[E]**
2. **The exploited component was the score-shaped side-payment, not the contribution metric.** Contribution metrics survived ~7 years; the leaderboard bonus lasted months. Any bonus that is a function of rank rather than of contribution is the attack surface. **[E]**
3. **TC died of opacity.** "Legibility and local verifiability are load-bearing incentive properties." **[E]** Prism should read this as a hard constraint: its composite (anchors + clustered bootstrap + LCB + gates + mirror penalties) is *already* near the edge of what a miner can recompute locally. Adding a novelty term or a Shapley term pushes it over. **[J]**
4. **Pure marginal-contribution payment under-rewards whoever built the frontier.** Numerai's answer is to keep a substantial **level** term (now 0.75×CORR) alongside MMC. **This is direct real-money precedent for a split pool — pay partly on level, partly on marginal contribution — which is what I recommend.** **[E→J]**

**What does *not* transfer.** Numerai's residualization is linear algebra on prediction vectors: you can literally subtract the meta-model from a submission. **Architectures do not linearly decompose** — there is no "subtract the champion architecture" operation. So the mechanism transfers only as an *analogy*: per-axis marginal contribution (does this submission advance a frontier nobody else advances?), not literal residualization. I flag this because it is the easiest place to over-read the precedent. **[J]**

### 2.3 NAS: the reproducibility crisis and what it says about short-budget ranking

- **Random search is a brutal baseline, and the convergence across independent groups is the field's most reproducible finding. [E]** Li & Talwalkar: random search + early stopping ≥ ENAS on PTB and CIFAR-10; random + weight sharing ≈ DARTS/SNAS. Sciuto et al.: "**On average, the state-of-the-art NAS algorithms perform similarly to the random policy**". Yang et al. (8 methods × 5 datasets): many fail to beat the *random average*; **protocol tricks dominate; macro-structure matters more than the searched micro-structure**. Zela et al.: DARTS *worse than* random-search-with-weight-sharing on S1–S4. An extensive appraisal puts the probability that weight-sharing beats random search at **7 %–78 % depending on search space**. NAS-Bench-Suite: conclusions from 1–2 benchmarks do not transfer across 25 space×dataset combos.

  **Honest caveat on a number I would have liked:** no paper in this set publishes a clean "X % of reported NAS gains survived" figure. The consensus that most pre-2020 gains came from search-space engineering, protocol tricks and seed selection is well-supported; the specific fraction is **thin** and I will not invent one.

  Two consequences for Prism, and the first is a genuine strength of the existing design:
  1. **Prism's pinned-recipe discipline is exactly right.** Fixed data (SHA-256-verified FineWeb-Edu shard), fixed budget, organizer-run harness, pinned container image — this is the "hold the protocol constant" lesson the NAS field learned the hard way, and Prism has it by construction.
  2. **Every round should include an explicit baseline/naive entrant, scored like any other submission.** The literature's most reproducible finding is that protocol and seed variation produce effects **of the same magnitude as or larger than** the architectural effects being rewarded. Prism ships reference baselines (`transformer_pp`, `hybrid_delta`) — running one *every round* and publishing its score turns those confounds into a measured control instead of an assumption. **[E→J]** This is cheap and I would add it to the §4.3 shortlist if I were allowed a fourth entry.
- **Low-fidelity ranking is a two-regime phenomenon, and this is the most important empirical result in this document after the noise numbers.** **[E]**

  | Regime | Evidence | Verdict |
  |---|---|---|
  | **Screening** (discard the bottom half) | Spearman ρ reaches **0.6 after a few epochs** and rises steadily (RANK-NOSH, NAS-Bench-201, 3 datasets); at 10 epochs **~70 %** of bottom-half architectures stay bottom-half; sum-of-training-losses reaches **ρ = 0.851 at 5 % of budget** and **0.95 at 25 %** (TSE, NeurIPS 2021) | Low fidelity is **genuinely informative** |
  | **Selecting the winner** (top-1 of the field) | Kendall τ across fidelities **restricted to the top-10 of 3,000** individuals is explicitly "lower than some studies reported", *because top individuals have very close true performance* (MPENAS, GECCO 2023); independent replication finds top-10 %-yes / top-1 %-no | Low fidelity is **near-useless** |

  Additional cross-fidelity numbers for calibration: Kendall τ ≈ **0.42** between a 1-epoch proxy and full training (rising to 0.73 under a distillation loss designed to raise it); sub-sampled-data proxies at 1/27 of data give τ from **−0.03 to 0.57**, at 1/3 of data **0.41–0.92** **[E]**.

  **This is the cleanest available statement of why WTA is the wrong collapse for Prism: a 6 h run is a low-fidelity instrument, and winner-selection is precisely the regime where low-fidelity instruments fail. Prism's WTA rule sits entirely in the failing regime, while the screening regime — where the instrument works — is exactly what the graded band and the two-stage funnel exploit.** **[E→J]**

- **Seed noise alone re-ranks architectures, measured.** 32 architectures trained under **2 different seeds** → **Kendall τ = 0.48**, with mean per-architecture test-accuracy change **0.13 % ± 0.08 (max 0.39 %)** — described as "substantial considering the small gap between random architectures and NAS methods" **[E]**. And re-training the same 32 architectures at a **different depth** (8 vs 20 cells) gave **τ = 0.54** — architectures re-rank when you change scale, even within one benchmark. **This is direct measured support for §1.3c and §3.1: seed variance is not a rounding error, it is comparable in magnitude to the architectural effect being rewarded.** **[E]**

- **Do not build a learning-curve extrapolator; buy replicates instead.** For the binary "which of two is better" task, a parametric extrapolation model (MMF) is **outperformed by simply using the last observed anchor** (Kielhöfer et al.); parametric models only win when the *exact value* is needed, not the comparison **[E]**. LC-PFN's only slowdowns across 53 datasets were on the **NAS-Bench-201** tasks, whose curves were "very unlikely under our prior" **[E]** — a caution against trusting a learned prior on genuinely novel architectures, which is exactly Prism's population. **Consequence: spend marginal compute on seed replicates, not on curve prediction.** **[E→J]**

- **If you must shrink fidelity, shrink width — not epochs, not data.** EcoNAS (CVPR 2020) measured reduction factors across channels/resolution/samples/epochs: at equal total iterations, **more samples with fewer epochs beats more epochs with fewer samples** for rank consistency, and **reducing channels is more reliable than reducing resolution** **[E]**. Directly actionable for the screen tier (§5.5): reduce width, keep the data and the schedule shape.

- **Successive halving / ASHA is the mechanism to copy, and its designers state why it tolerates Prism's exact problem.** ASHA promotes a config as soon as ≥η observations exist at its rung; suboptimal early promotions "have only a modest impact on performance, not only because the ranking… is often fairly consistent across rung levels, but also because rungs grow over time" **[E]**. **ASHA is explicitly engineered to be robust to low-fidelity ranking noise** — which is the structural argument for the two-stage funnel in §5.5.
- **Architecture rankings genuinely flip with scale, with named examples.** Tay et al., *Scaling Laws vs Model Architectures* (EMNLP Findings 2023) — **10 architectures**, **>100 models** pretrained and finetuned, **15 M → 40 B params** **[E]**:
  - "**The best performing model can fluctuate at different scales.**"
  - **Vanilla Transformer has the best scaling behaviour (α_{F,U} = 0.54, α_{F,D} = 0.28) even though its absolute performance at each individual compute region is not the greatest.** — i.e. *the winner at one compute point is not the best scaler*, which is the exact confusion a single-fidelity leaderboard invites.
  - **Concrete rank flip:** *Evolved Transformer beats vanilla at tiny-to-small scale on downstream tasks but falls behind when scaled up.* MoS-Transformer beats vanilla in some compute regions and not others.
  - **Performer:** α_{F,U} = 0.25; upstream perplexity improves only **2.7 %** base→large vs **8.4 %** for vanilla.
  - **ALBERT scales *negatively* downstream: α_{F,D} = −0.12** — worse as compute increases. (Note ALBERT's mechanism is **cross-layer weight sharing** — structurally the same family as the looped/recurrent-depth architectures Prism's flagship A/B is designed to test.)
  - Authors' own advice: "**be cautious when staking an expensive run on an architecture that drastically modifies the attention mechanism**"; Mixers and Performers are "high risk". They also concede the converse, which is Prism's honest defence: "not every practitioner would require models that scale to billions… inductive biases tailored to small or low compute will be [valuable]."

  **This is the deepest threat to Prism's premise and it deserves to be stated plainly rather than buried:** a competition that ranks architectures at ~4.5e18 FLOPs is measuring "good at 160 M params / 6 h", and the literature's direct measurement is that this ranking **does not reliably transfer** to the regime anyone cares about. Combined with the scaling report's finding that the local slope is E-confounded and unusable, Prism cannot currently claim to select architectures that scale. It can honestly claim to select architectures that are **better at a fixed, pinned, small budget**. **[J]** The design should say that out loud (§6.5) — over-claiming here is the reputational risk that dwarfs any mechanism detail.
- **Weight-sharing/one-shot proxies rank badly**, which is why Prism's decision to train each submission standalone is right despite the cost. **[E]**
- **Zero-cost proxies** (synflow, NASWOT, jacov) are cheap and inconsistent across spaces; NAS-Bench-Suite-Zero's contribution was showing how space-dependent they are. **Usable as a pre-pod screen, never as a scorer.** **[E→J]**

### 2.4 Competition design: leaderboards that resist adaptive overfitting

- **The Ladder (Blum & Hardt, ICML 2015) [E] — the closest thing to a proof that my recommended rule is correct.** Three load-bearing elements:
  1. **Threshold** — accept an update only if the new loss beats the incumbent by more than η.
  2. **Withholding** — if the threshold is not met, **re-publish the old number**. The participant learns only "did not clear η", nothing more.
  3. **Rounding** — published values are quantized to precision η, so each release leaks only `O(log(1/η))` bits.

  Guarantee (Thm 3.1), for **adaptively chosen** submissions: `Pr[min_i R_D(f_i) − R_t > ε + η] ≤ exp(−2ε²n + (1/η + 2)log(4t/η) + 1)`, giving leaderboard error `O(log^{1/3}(kn) / n^{1/3})` at `η = O(n^{−1/3}log^{1/3}(kn))`.

  **The design implication I had not appreciated, and it reverses a natural instinct: the error depends on the number of submissions `k` only logarithmically, while the binding term is the holdout size `n` (an `n^{−1/3}` rate — the price of full adaptivity).** In plain terms: **many submissions are nearly free; a small eval set is what kills you.** For Prism this is doubly useful — it says the funnel can be widened aggressively (§6.2) *without* adaptivity cost, and it independently confirms that **n = 200 items/task is the thing to fix first** (§4.1). It also means submission rate-limiting, the instinctive defence, is aimed at the wrong variable.

  **The parameter-free variant is what to copy**: replace fixed η with a significance test on whether the new submission genuinely improves on the incumbent, accepting at an `Ω(1/√n)`-scale margin (~`1.1×std` in their deployment). **The step shrinks automatically as the leader improves — no tuning.** Validated two ways: against a boosting attack on a Kaggle-style mechanism (`n = 4000`), the naive leaderboard becomes "strongly biased" and degrades rapidly with submission count while the Ladder "encounters only a small bias"; and it "achieves high accuracy" on real Kaggle submission files.

  **Two conclusions for Prism. First, the "else" branch is the mechanism** — the recommendation in §7.1 that the champion simply *holds* on a failed challenge is not a workaround for noise, it is the element that carries the guarantee. **Second, Prism's LCB ranking is not this**: LCB compares independently-bootstrapped *levels*, discarding the pairing that the parameter-free Ladder exploits, and therefore tolerating more noise than necessary. Switching champion displacement to a **paired** test is a strict improvement with a proof behind it. **[E→J]**
- **Thresholdout / reusable holdout (Dwork et al.) [E].** Report holdout performance only when it differs from training performance by more than a noisy threshold; each release spends privacy budget. Same family as the Ladder, same lesson: **information release must be rate-limited to preserve validity**.
- **Kaggle's actual evidence is more nuanced than the folklore [E].** Roelofs et al., *A Meta-Analysis of Overfitting in ML* (NeurIPS 2019), >100 Kaggle competitions: "**little evidence of substantial overfitting**", public/private correspondence "remarkably good", effect sizes among top submissions "**typically small (e.g. less than 1 % classification accuracy)**". Outliers exist and "usually have pathologies such as **non-i.i.d. data splits or (effectively) small test sets**."

  **I am not going to paper over that this cuts against the standard anti-Goodhart pitch.** The honest reading is conditional: holdout reuse is robust *when the test set is large and the metric is stable*. Prism's G2 caps at **200 items/task** (`eval_asset_cap(200, 8, …)`), which is squarely the "effectively small test set" pathology Roelofs flags as the exception. So Prism should expect the *exceptional* behaviour, not the reassuring average. And unlike a Kaggle competition with a one-time prize and a final private scoring, Prism pays a **recurring emission stream** against a **standing public battery** — a strictly stronger adaptive pressure than anything in Roelofs' sample. **[E→J]**
- **The Leaderboard Illusion (2025) [E] — and it quantifies the best-of-n attack, which is the number to remember.** On Chatbot Arena: selective disclosure (27 private Meta variants pre-Llama-4), data asymmetry (two providers ~19–20 % of battles each vs 29.7 % for 83 open models combined), up to **112 % relative** Arena-score gain from arena-distribution access — and critically, **best-of-n private submission is worth ~100 Elo from 10 variants**. That is a pure noise-exploitation gain: nothing about the model improved, only the selection over noisy draws. Recommended fixes: publish all variant scores, prohibit selective retraction, uniform private-testing limits. Organizers dispute parts of the analysis (**[contested]**), but the best-of-n mechanism itself is not in dispute.

  **Why this matters more for Prism than for Arena:** best-of-n and the winner's curse are the same mathematics, and under WTA the payoff is the *maximum* of the field's noisy draws — so WTA does not merely tolerate best-of-n, it **pays for it**. Prism's defences are the right ones and mostly already exist: **one accepted submission per hotkey** (`submission_gating`), coldkey-scoped precheck quota (3/coldkey/UTC day — note it is scoped to **coldkey**, so hotkey rotation does not reset it), and a `precheck` route that runs no pod and does not score. The missing piece is **confirmation runs on fresh seeds before emission** (§5.5), which is what actually voids a lucky draw rather than merely rate-limiting attempts.
- **ARC Prize / MLPerf [E].** ARC: a single sharply-defined metric, an 85 % gate, efficiency as a **constraint** not an axis, and the private set scored **once**. MLPerf: **no scalar aggregate**, closed division with accuracy gates. Both are evidence for gates-over-scores and against elaborate scalarization. Cost of the ARC approach: the grand prize went unclaimed — in an emissions context that means **burn**, which Prism already supports.

### 2.5 Statistics of ranking under noise, and contest theory

- **Best-arm identification.** Fixed-confidence sample complexity scales as `Σ_i 1/Δ_i²` (up to log factors) where `Δ_i` is arm `i`'s gap to the best. The operational consequence for Prism is stark: **you cannot buy your way out of the gap problem by adding eval items**, because the dominant term is training-seed variance and each arm is pulled **exactly once** (§1.3c). Prism is not running a bandit; it is running a **one-pull-per-arm** experiment, which is the worst case for identification. **[E→J]**
- **Winner's curse / regression to the mean.** Selecting the argmax of noisy estimates yields an expected true value **below** the observed maximum, with bias growing in the number of candidates and in noise. **[E]** Two direct consequences for Prism: (i) the *published* champion score is optimistically biased, so the public leaderboard systematically overstates progress; (ii) if the champion's score is **carried forward** without re-measurement (which `prism-emit` does — positive scores carry until superseded), the incumbent is defended by an **inflated** number, which perversely makes displacement harder over time. That is an argument for periodic **re-measurement** of the champion independent of the anti-overfit argument. **[J]**
- **Moldovanu & Sela (AER 2001) [E].** With private ability and effort-cost heterogeneity: **linear or concave** cost ⇒ put the entire purse in a single first prize; **convex** cost ⇒ several positive prizes can be optimal (necessary and sufficient condition given).
- **Drugov & Ryvkin (JET 2020), *Tournament rewards and heavy tails* [E] — the decisive result.** "*While a winner-take-all prize schedule maximizes aggregate effort for light-tailed shocks, prize sharing becomes optimal when shocks acquire heavy tails… Extreme prize sharing — rewarding all ranks but the very last — is optimal when shocks have a decreasing failure rate, such as power laws.*" Formally: **IFR noise ⇒ WTA optimal; DFR/heavy-tailed noise ⇒ share.** Companion paper (*How noise affects effort*, JET 2020): more dispersed noise reduces equilibrium effort, and the right order is the **dispersive order**, not variance.
- **Fu & Lu (Economic Theory 2012) [E].** When contest technology is **sufficiently noisy**, multi-stage beats single-stage; additional stages always increase total effort; with concave/moderately-convex impact, put the whole purse in the final prize. **This is a genuine tension with Drugov–Ryvkin and with the metascience evidence, and I resolve it explicitly in §3.4 rather than picking the convenient citation.**
- **Two-stage/shortlist results [E].** To maximize *peak* performance, ~2 finalists is typically optimal; the optimal number of **non-zero** prizes is shortlist size **minus one** (exactly one zero-prize rank is needed).
- **Peer review cannot fine-rank near the top [E].** Fang, Bowen & Casadevall (eLife 2016), 102,740 NIH grants at percentile ≤20: **r² = 0.0078**, random-forest variance explained **~1 %**, **ROC AUC 0.54**, and **17 %** of percentile-zero grants produced **zero** citations. Verbatim: peer review has "*minimal impact in stratifying meritorious applications relative to what would be expected from a random ranking*." Replicated at NHLBI (1,492 R01s) and NIMH (1,755 R01s).
- **Randomizing within a merit band costs nothing measurable [E].** Volkswagen Foundation "Experiment!" — >5,000 applications, 183 grants, a **supervised physical lottery** among jury-approved proposals created two cohorts from one pool: "*very similar research outputs and outcomes, including publications, patents, and funding/career effects… no significant difference in quality between the two groups*", plus **increased diversity** and **more risk-taking**. HRC New Zealand Explorer Grants randomize among all applications meeting criteria, on the stated rationale that "**fine-grained ranking near the margin can carry more noise than signal**".
- **Concentration vs dispersal of research funding [E].** Fortin & Currie (PLOS ONE 2013): impact is a **decelerating** function of funding, impact-per-dollar **falls** with grant size, "inconsistent with the hypothesis that larger grants lead to larger discoveries". Mongeon et al. (Research Evaluation 2016), 12,720 researchers: concentration produces "**diminishing marginal returns**"; "the most funded researchers do not stand out". Aagaard et al. (QSS 2020), systematic review of 92 papers from 3,567 screened: "**strong inclination toward arguments in favor of increased dispersal**"; concentration may cause "**diseconomies of scale**".
- **Registered reports as an anti-Goodhart device [E].** Scheel, Schijen & Lakens (AMPPS 2021): positive-result rate **96.05 %** in standard reports (N=152) vs **43.66 %** in Registered Reports (N=71). Pre-registration more than halves the positive-result rate — the cleanest available quantification of how much outcome-contingent reporting distorts a literature. Prism's anchor pre-registration (`/v1/preregistration`, hash-committed anchor sets) is the same device, and this is the number that justifies it.
- **Prediction markets forecast replication with real accuracy [E].** Dreber et al. (PNAS 2015) and the pooled four-project analysis (PLOS ONE 2021) show markets beat survey means at predicting replication outcomes. Relevant as a **cheap forecasting layer** over pending submissions, but it is a governance add-on, not a scoring path. **[S]** for Prism specifically.

---

## 3. Q2 — Winner-take-all vs top-k decay vs significance-aware collapse

### 3.1 First, get the noise magnitude right

Everything here depends on one number: the sd of the **measured composite for a fixed architecture+recipe across independent training runs**. Call it `σ_run`. It is **not** what Prism currently estimates (§1.3c).

Decomposition:

```
σ_total² = σ_eval²  (eval-item sampling — what the bootstrap measures)
         + σ_seed²  (init/data-order/nondeterminism — NOT measured, irreducible at 1 run/submission)
```

From the literature **[E]**: restart variance up to **3.5 % relative** on loss; PolyPythias (50 runs, 14 M–410 M, 10 seeds) finds most runs within 2 sd. Taking `σ_CE ≈ 0.01–0.08` nats/token and converting to the tokenizer-neutral unit Prism scores on (`bits_per_byte = CE/(ln2 · bytes_per_token)`, `bytes_per_token ≈ 4` for a typical BPE):

| σ_CE (nats/token) | σ_bpb (bits/byte) | as % of bpb ≈ 1.15 |
|---|---|---|
| 0.01 | 0.0036 | **0.31 %** |
| 0.02 | 0.0072 | 0.63 % |
| 0.05 | 0.018 | 1.6 % |
| 0.08 | 0.029 | **2.5 %** |

So **`σ_seed` on the dominant axis is ≈ 0.3–2.5 % relative.** **[J, from [E] inputs]**

Two observations worth their own line:

- **SN9's ϵ = 0.5 % sits inside this band**, near its lower half — mild independent corroboration that a well-run competition's epsilon lands at roughly one seed-sd, arrived at empirically by a different team on a different metric. **[J]** The live decay schedules confirm the same order: SN9 `LinearDecay(0.005 → 0.0001)` over ~2 days, SN37 `LinearDecay(0.05 → 0.01)` over 1–5 days **[E]**.
- **SN37's epsilon is 10× SN9's, and the diagnosis matters for Prism.** Its eval tasks draw only **120–150 examples**, so **epsilon is doing double duty: absorbing measurement noise *and* deterring copying — two jobs that do not share an optimum** **[E→J]**. This is precisely the conflation my §7.1 rule avoids by separating the **statistical** term (scales with measured uncertainty, never decays) from the **economic** floor (policy choice, decays with tenure). Prism, at n = 200, is in SN37's regime, not SN9's — so a single blended constant would be badly compromised in both directions.
- **Independent evidence for run-to-run instability from a third team.** Templar explicitly discards its raw loss score, stating that **"even adjacent iterates can lead to very different scores for the same peer"**, and keeps only the ranking **[E]**. That is a fourth system concluding the raw noisy scalar is not directly usable — and direct field support for the `σ_seed` concern in §1.3c.
- **G2 is hopeless at the current cap and this is already established.** MDD at n=200 is **6.6–9.8 pp** against FLOP-matched expected margins of 1–3 pp, and **three of eight tasks normalize to a constant 0** for the entire field. A leaf built on G2 alone (the live v4 default) is therefore mostly measuring noise plus dead weight. **[E, from the input report]**

**Minimum detectable difference.** Two submissions each trained **once**, each with run-level sd `σ_run`, give `SE(Δ) = σ_run·√2`. Under the one-sided α = 0.05 rule I recommend in §7.1, a difference is declared real when `Δ > 1.645·SE(Δ) = 2.33·σ_run`. (Convention stated explicitly because it is easy to mix up: a two-sided 95 % criterion would give 2.77·σ_run, ~19 % larger. I use one-sided throughout, matching the existing `lcb_z = 1.645`.)

Pairing helps on the eval component — per-question outcomes across models on the same harness correlate **0.3–0.7** **[E]**, so the eval part of `σ_paired` is materially below `√2·σ_eval`. But pairing **cannot** reduce `σ_seed`: two different architectures trained once each have independent seed draws. So `σ_seed` sets a hard floor.

| σ_run (relative) | MDD = 2.33·σ_run | Verdict on a claimed 2 % gain |
|---|---|---|
| 0.3 % | **0.70 %** | detectable |
| 1.0 % | **2.33 %** | **borderline — not detectable** |
| 2.5 % | **5.8 %** | far from detectable |

**Consequence: with realistic seed noise, a 2 % architectural improvement is at or below the detection threshold of a single 6 h run.** This is the central quantitative fact for the whole mechanism question. **[J]**

### 3.2 How many ranks can be honestly resolved?

Plausible spread of *true* quality among serious submissions at a fixed pinned budget: **0–5 %** relative on the dominant axis (a genuinely better architecture at matched compute; the Cerebras 111M→256M step is ~10 % for 5× the compute, so 1–5 % from architecture alone at fixed compute is the right order). Against `MDD ≈ 2.3 %` (σ_run = 1 %):

**Honest resolution ≈ 3–4 distinguishable tiers.** Not 1 (WTA discards the tier structure) and not 10 (paying rank 5 vs rank 8 is paying noise). **[J]**

**This is the statistical justification for a top-3/top-4 graded payout — and it happens to validate the v2.1 `top3` instinct on grounds the brief did not state.** The reason to pay three ranks is not fairness or participation; it is that **three is approximately the number of levels the instrument can measure.**

### 3.3 The four strategies, scored against the four failure modes

Let `P` be the epoch's prism share, `k` the number of serious submissions, and assume the top few are statistically indistinguishable (the regime §3.1 establishes).

#### (a) WTA on point estimates — the current implementation

- **Copy-with-tweak: catastrophic.** A functional copy has *identical true quality*. Two draws from the same distribution ⇒ **P(copy wins) = 0.5** by symmetry. **Expected value of a pure copy ≈ 0.5·P for the price of one pod (~$5–15).** The entire mechanism's integrity rests on the copy detector — cheap byte/AST screens, LLM similarity, agentic review. The adversarial-code-clone literature documents that semantics-preserving transformations (renaming, control-flow flattening, dead-code insertion) evade AST/embedding detectors at high rates **[E]**. **A mechanism whose primary defence is a detector that is known to be evadable is a mechanism with a single point of failure.** **[J]**
  - Mitigating detail: `prism-emit` **carries** the champion's positive score forward rather than re-measuring it, and by the winner's curse that carried number is an *optimistic* draw. So the copy must beat an inflated target, pushing P(copy wins) somewhat below 0.5. This helps by accident, not by design, and it simultaneously entrenches the incumbent (§2.5). **[J]**
- **Variance-farming: strongly rewarded.** Under WTA the payoff is convex in the score — only the max matters — so a rational miner **maximizes variance**, not expected quality. Submitting a high-variance long-shot (aggressive LR, exotic unstable component) strictly dominates a careful 1 %-better submission when both are within noise. This is the textbook risk-shifting result: Knoeber & Thurman's field test confirmed "**the worst players use the riskiest strategies**" **[E]**. Prism's gates (`g3 ≥ 0.25`, `g8 ≥ 0.5`) truncate the *downside* of a wild submission at zero, which makes the bet **more** attractive, not less. **[J]**
- **Exploration incentive: destroyed for everyone but the leader.** Expected value for a non-leading miner ≈ `P/k` minus a certain pod cost. As `k` grows or the leader's advantage becomes visible, EV goes negative and **entry collapses**. Contest theory usually assumes a fixed contestant pool; in a permissionless subnet the pool is **endogenous**, so WTA's effort-maximizing property is undercut by a participation margin the theory does not model. **[J]** This is the mechanism behind SN9's observed hoarding equilibrium **[E]**.
- **Incumbent squatting: yes, via score carry.** The champion earns 100 % while spending nothing after the initial run.

#### (b) Top-k decay (the proposed v2.1: 100 %/50 %/25 %)

- **Improves**: participation (three ranks paid ⇒ positive EV deeper into the field), income variance, entry.
- **Does not fix**: ranking among indistinguishable submissions is *still* noise-driven, so a copy now reliably lands **in the paid band** — arguably worse, since a copy is guaranteed ~top-3 by construction and no longer needs to win the coin flip. **Top-k decay alone raises the EV of copying from 0.5·P to something closer to a certain 0.29·P** (the 50 % share of a 100/50/25 normalization), trading a lottery for an annuity. **[J]**
- **Verdict: necessary but insufficient.** It must be combined with a margin rule, not substituted for one.

#### (c) Significance-aware collapse (champion holds unless beaten by > measurement uncertainty)

- **Copy-with-tweak: the *champion* slot is solved mechanically; the *band* is not.** A copy has true Δ = 0, so P(it passes a one-sided α = 0.05 test) ≤ **5 %**, less after the economic floor. **EV of capturing the champion share falls from ~0.5·P to <0.05·P — a >10× reduction, with no reliance on any detector.** The copy detector becomes defence-in-depth instead of load-bearing. **This is the single strongest argument in this document.** **[J, derived from [E]]**

  **The qualification, which SN9's measured experience forces and which I will not soften: epsilon bounded copying at the top slot but copies still populated the leaderboard — "direct copying with practically zero amendments" [E].** A significance gate protects the 60 % champion share; it does **not** protect the 30 % graded band, where a copy is a statistical tie with the champion and therefore lands high by construction. So the band *does* rely partly on the copy gate, and I should not claim otherwise. What limits the exposure to ~30 % of `P` rather than 100 %: the 1-max-per-hotkey gate, the miner-funded pod cost per attempt, coldkey-scoped precheck quota, and the existing byte/AST/agentic screens. **Net: significance gating converts copying from a majority-of-pot attack into a bounded, detector-mediated, per-identity-capped nuisance — a large improvement, not an elimination.** **[J]**
- **Variance-farming: neutralized.** A high-variance junk submission's *expected* significant win is capped at the test's false-positive rate. The payoff stops being convex in the score.
- **Incumbent squatting: introduced.** This is the real cost, and it is exactly what SN9 hit **[E]**. The fix is SN9's: **decay the economic floor with the age of the title** so a hoarder is progressively easier to displace — while the **statistical term never decays**, because that term is a truth condition, not a policy preference.
- **Exploration: mixed.** Fewer champion flips means a challenger who *is* better but by less than MDD earns nothing, which discourages incremental honest work. Mitigated by paying the graded band and the exploration pool.

#### (d) Merit-band quantization with equal split or lottery inside the band

- The Volkswagen result says randomizing within a merit band costs **no measurable quality** **[E]**, and the NIH evidence (AUC 0.54) says fine-ranking near the top is theater **[E]**. Under this rule, beating the leader by 0.1 % moves you **into the same band, not above it** — which removes the epsilon-chasing attack surface entirely.
- **Why I do not recommend it as the primary rule anyway:** a lottery makes emissions unpredictable for the miner and is politically hard to defend in a token context where participants expect merit-legibility; and equal-split-in-band invites **sybil crowding of the band** (many marginal-but-qualifying entries from one operator), which the 1-max-per-hotkey gate limits but does not eliminate across hotkeys. I use band logic **inside** the graded tier (adjacent ranks pay similarly) rather than as the top-level rule. **[J]**

### 3.4 Resolving the contest-theory tension honestly

Three bodies of evidence appear to conflict:

| Source | Says | Objective being maximized |
|---|---|---|
| Moldovanu–Sela; Fu & Lu | Concentrate the purse; narrow to ~2 finalists | **Effort** from a fixed pool |
| Drugov–Ryvkin | Share the purse when noise is heavy-tailed | **Effort**, but noise-shape-aware |
| Fortin/Mongeon/Aagaard | Disperse; concentration has decreasing marginal returns | **Discovery per dollar** |

#### Correction: I first misapplied Drugov–Ryvkin, and the correction favours concentration

**This was flagged in review and the reviewer is right on the theory.** In a Lazear–Rosen/Drugov–Ryvkin rank-order tournament, output is `effort + idiosyncratic shock`, and the shock whose tail determines the optimal prize schedule is the **measurement/luck term that corrupts the effort→rank mapping** — not heterogeneity in the production function. Prism's measurement noise (seed variance, eval-item sampling, 200-item binomial accuracy) is **light-tailed / IFR**. **Applied correctly, Drugov–Ryvkin therefore favours WTA, not sharing.** My original §3.4 reasoning — "rank is mostly determined by idea quality, which is heavy-tailed, therefore share" — substituted a different random variable for the one the theorem is about. That is an error, not a nuance, and it removes what I had listed as reason #2 in my verdict.

Two things partially survive, and I mark them at the confidence they deserve:

- **Moldovanu–Sela still pulls the other way.** Linear/concave effort cost ⇒ single prize regardless of ability distribution; **convex** cost ⇒ several prizes may be optimal. Effort cost in architecture search under a fixed compute cap is plausibly convex (diminishing returns to successive search within one 6 h budget). So the **cost** condition favours multiple prizes while the **noise-tail** condition favours one. The two conditions are independent and they genuinely split. **[E→J]**
- **The rank-order tournament may not be the right model at all.** If Prism is better described as a **stochastic innovation race** — the operator funds many independent draws and keeps the best — then the governing result is Terwiesch & Xu, who show the "optimal n = 2" conclusion (Fullerton–McAfee, Che–Gale) **does not transfer** when value comes from solution *diversity* and the sponsor retains the best draw. Taylor 1995 adds that free entry is not optimal because per-researcher effort falls in n. **This, not the tail argument, is the defensible contest-theoretic case for paying more than one entrant.** **[E→J]**
- **[S]** One speculative possibility I will not lean on: Prism's *composite* is a gated weighted geometric mean, so the aggregate could have a fatter lower tail than its light-tailed components. That would matter for the D–R condition, but I have not analysed it and it should not be used as an argument.

#### What actually reconciles them

- Contest theory maximizes **effort** from a fixed pool. Prism's scarce inputs are **GPU-hours** (miner-funded, hard-capped per submission) and **ideas**; effort cannot be increased past the 6 h cap. And the pool is **endogenous** — permissionless entry means the participation margin, which the theory holds fixed, is exactly what WTA damages. This is the objection that does not depend on the tail condition. **[J]**
- Fu & Lu's "noisy ⇒ multi-stage" is fully compatible and I adopt it (§5.5). Their "whole purse to the final prize" conclusion rests on the effort objective. Note also their result that a **best-of-N contest *is* a lottery contest** **[E]** — which is the formal statement of the best-of-n attack in §2.4.
- Metascience measures **discovery per dollar**, Prism's actual objective, and points to dispersal — but it measures publications/citations, not capability gains, and the scaling literature points the other way for **exploitation**. Reconciliation: **disperse for search, concentrate for scale-up.**

### 3.5 Verdict

**WTA is still the wrong collapse for Prism — but the correct argument is narrower than my first draft claimed, and it is not the noise-tail argument.** Structure: significance-gated champion tenure + a graded band + an exploration pool. Reasons, re-ranked after the correction above:

1. **Copy EV.** WTA gives a pure copy an expected **50 % of the pot**, making an evadable detector load-bearing; a significance gate cuts that to **<5 %** mechanically. **This argument is independent of contest theory and is the load-bearing one.** **[J]**
2. **The instrument resolves ~3–4 tiers** (§3.2), so paying rank 1 alone discards information the measurement actually contains, and paying 10 ranks pays noise. Corroborated by SN9's realised collapse paying ~3 models **[E]**.
3. **Endogenous participation.** WTA on whole miner-funded artifacts drove SN9 to ~6 miners **[3P]** and produced documented **hoarding**, which its own authors called a design flaw and IOTA named as a reason to abandon WTA **[E]**. Contest theory assumes away the margin that broke here.
4. **Fine-ranking near the top is not statistically supportable** (NIH AUC **0.54**; Volkswagen two-cohort null result) **[E]**, so a rule that refuses to distinguish statistical ties is more defensible than one that pretends to.
5. **Diversity-valued innovation races** justify paying more than one entrant (Terwiesch & Xu) **[E→J]**, and **convex effort cost** admits multiple prizes (Moldovanu–Sela) **[E]**.

**Where the reviewer and I still differ, stated plainly.** They would keep rewards concentrated (effectively 100 % to a significance-gated champion) and move selection out of the noise floor via the Ladder gate, replicates, and fresh-seed confirmation. I recommend 60 % champion + 30 % band + 10 % pool. **We agree completely on the mechanism that matters most — paired significance gating, replicates, fresh-seed confirmation, hard per-identity caps — and disagree only on the residual 40 %.** Given that the tail argument no longer supports me, the honest characterisation is: **the case for the band rests on participation, copy-EV containment, and measured tier resolution, not on contest theory, which on balance favours concentration.** An operator who weights theoretical cleanliness over participation risk could defensibly ship champion-takes-90 % with the same gate, and I would not call that wrong — only more exposed to the entry collapse SN9 experienced.

---

## 4. Q3 — Anti-Goodhart: defences, sufficiency, and traps

### 4.1 The threat is sharper than "leaderboards get overfit"

Prism's specific exposure, assembled from verified facts:

1. The battery is **public** and the anchors are **pre-registered** (by design — pre-registration is an anti-Goodhart device, §2.4, and worth the exposure it creates).
2. G2 caps at **200 items/task** — Roelofs' "effectively small test set" pathology, the documented exception to holdout robustness **[E]**.
3. The reward is a **recurring emission stream**, not a one-time prize — strictly stronger adaptive pressure than any Kaggle competition in Roelofs' sample **[J]**.
4. The mirror-gap penalty, the designed contamination detector, is **inert in the default tier** (§1.3b, verified).
5. Three of eight G2 tasks contribute a **constant 0** to every submission, so the live v4 leaf has less dynamic range than its dimensionality suggests **[E]**.

So the risk is concrete: **the cheapest way to raise the live v4 leaf is to fit the four G2 tasks that still move**, and the detector meant to catch that is currently switched off by tier selection.

**But the priority ordering deserves to be stated against the evidence, because it is not what the framing of the question implies.** The two largest empirical investigations of leaderboard overfitting (Roelofs et al. on >100 Kaggle competitions; Recht et al. on ImageNet/CIFAR reuse) both conclude that **adaptive overfitting is a smaller problem than theory feared, and that measurement validity is the larger one** **[E]**. Set against that: a *single seed change* re-ranks NAS architectures at **Kendall τ = 0.48** **[E]**.

**Conclusion I will state plainly, including where it cuts against my own §4 shortlist: Prism's dominant error source is not miners farming the holdout — it is that a 200-item task with a ~9 pp noise floor cannot resolve the differences being paid for.** The anti-Goodhart machinery below is cheap insurance and worth deploying, but if there is a budget conflict, **battery size, replicate counts, and construct validity come first.** The anti-Goodhart additions in §4.3 are ranked among themselves; §7.3's sequencing (variance first) is the one that governs overall.

**The item-count arithmetic is forced, and worth stating because it bounds what any procedure can achieve.** At `p = 0.5` on 200 items, the 95 % CI half-width on a *difference* is **±9.8 pp**. Halving that requires **~800 items** (width scales as `1/√n`). **No statistical procedure — not LCB, not pairing, not bootstrap — recovers resolution the item count does not contain.** Pairing reduces the *eval* component of variance by exploiting the 0.3–0.7 item correlation, which is real and worth having, but it cannot manufacture information. So `PRISM_EVAL_G2_CAP` is not a tuning knob; it is the binding constraint on G2's usefulness. **[E]**

**One caution against concluding too fast that the noise is irreducible.** A large share of the published "one-shot NAS is broken" result turned out to be an **evaluation-protocol bug rather than an intrinsic limit**: correcting batch-norm statistics before evaluation raised rank correlation by **~270 %**, taking Spearman to 0.46–0.71 **[E]**. **Before accepting Prism's noise floor as physics, look for the analogous protocol defect.** Prism already has three known candidates from the input report — the G1/G2 single-cluster bug, the G6 anchor inversion and censoring bug, and warmup-as-fixed-step-count — and the first of those *understates* variance while the others distort rankings outright. Fixing protocol defects is far cheaper than buying items or seeds. **[E→J]**

### 4.2 Defence-by-defence assessment

| Defence | Status in Prism | Assessment |
|---|---|---|
| Fixed pre-registered anchors (not field-relative) | **Has** — `anchors/v0.json`, hash-committed via `/v1/preregistration` | **Correct and load-bearing.** Field-relative normalization (z-score/min-max/rank) is Sybil-attackable; Prism avoided that. Registered-reports evidence (96 % → 44 % positive results) quantifies why pre-registration works **[E]** |
| Public/private mirror gap, τ_m = 0.05, on G2/G4/G5 | **Has, but inert by default** | Design is sound; **not sufficient as deployed.** Requires private staging to mean anything (§4.3 #1) |
| Lexicographic gates (`g3 ≥ 0.25`, `g8 ≥ 0.5`, budget caps) | **Has** | **Correct.** Gates are the strongest anti-degenerate-win device (MLPerf/ARC precedent **[E]**). Caveat: gates truncate downside and therefore *subsidize* variance-farming under WTA (§3.3a) — they work as intended only alongside a margin rule |
| Geometric mean across groups | **Has** (composite mode only) | **Correct** — weak axes have the highest marginal elasticity `w_k/g_k`, so stuffing a saturated axis pays ~nothing. But **the live default is v4 = arithmetic mean over G2 alone**, which has neither property |
| LCB ranking (`C − 1.645·SE`) | **Has** | Right instinct, **miscalibrated**: zero variance on 40 % of weight (§1.3a) and no seed-variance term at all (§1.3c). Also discards pairing, which the Ladder's parameter-free variant exploits **[E]** |
| Procedural/generated probes (G3/G4 from seeded generators) | **Has** | **Strongest anti-contamination asset in the system** — memorization-proof by construction, and template-regenerable. Under-weighted at G3 0.10 relative to its evidential value **[J]** |
| Fresh-crawl bits/byte (post-cutoff text) | **Has** (`g1.bits_per_byte_fresh`) | **Cheap and excellent.** Raise its weight |
| Rotating/held-out private slices | **Partially** — staging machinery exists (`PRISM_EVAL_TIER=private`, `.ready` gate, secret seed in env only, never on disk); rotation policy does not | **Cheap to add**, highest value (§4.3 #1) |
| Randomized probe parameters | **Partially** — generators are seeded, secret seed resolved per run | **Cheap to add** as an epoch-level rotation. **Trap if done wrong** — see §4.4 |
| "Prove-it-again" champion re-runs | **Absent** | **Cheap to add, second-highest value** (§4.3 #2) |
| Metric-diversity requirement (no single group carries a win) | **Partially** — geometric mean in v3; **absent in live v4** | **Third-highest value** (§4.3 #3) |
| Adversarial/held-out task families on a schedule | Absent | Moderate value, real cost; **partial trap** (§4.4) |
| Saturation tripwires (regenerate/reweight an axis whose top-quartile spread collapses) | Absent | **Cheap and underrated.** A pre-declared rule beats a discretionary one |
| Permanent non-retractable publication of all scored submissions | **Has in substance** (rows persist; champions on the public site) | Matches the Leaderboard Illusion fix **[E]**. Extend to *all* scored submissions, not champions only (§6.5) |
| Membership-inference scoring | Absent | **Trap** — near-chance on LLMs (Duan et al.) **[E]** |
| ECE / WeightWatcher α as scored metrics | Absent | **Trap** — binning-biased; contested **[E]** |
| Novelty as a scored axis | Absent | **Trap** (§5.2) |

### 4.3 The three highest-value additions

**#1 — Make the private tier mandatory for any scored epoch, and rotate the slice each epoch.**

*Why it is first:* it is the difference between having a contamination detector and having a comment that says you have one. Today `mirror = dict(public)` in `public_dev` and the gap is identically zero (§1.3b, verified in `rollup.py`).

*Concretely:* refuse to score an epoch whose realized `eval_tier != private` (fail-closed, consistent with existing posture); build the private pack per epoch with a fresh secret seed and a **freshly instantiated** item set (GSM-Symbolic-style template regeneration, not merely withheld items — "private" must mean "newly generated", because withheld-but-static items leak through repeated exposure); keep τ_m = 0.05 initially and **measure the realized gap distribution on the baselines before tightening it**.

*Cost:* pack-building is already scripted (`build_private_pack.py`). The real cost is operator discipline per epoch. *Risk:* a rotating private slice adds an epoch-to-epoch variance component to scores — which is precisely why the champion-displacement test must be **paired within an epoch** (§5.1).

**#2 — "Prove it again": re-run the incumbent champion on each epoch's fresh private slice.**

*Why:* it is a direct, causal test of the exact failure mode. If the champion's score holds on public anchors but degrades materially on a freshly generated private slice, that is anchor-overfit, measured rather than inferred. It also corrects the winner's-curse inflation in the carried score (§2.5) and is the only defence listed that can catch overfitting **after** it has already been rewarded.

*Concretely:* the champion's checkpoint is already parked with a verified receipt (`prism_artifacts::verify_parked`, `MANIFEST.json`/`RECEIPT.json`), so this is **eval-only — no retraining.** Against the report's proposed budget that is ~1.2 h instead of 6 h+. Rule: if the champion's re-measured composite drops by more than `z·SE(Δ_paired)` relative to its own prior measurement on a comparable slice, it **loses champion tenure** (reverts to the graded band) and the next-best significant submission is promoted.

*Make the timing unpredictable.* IOTA's verification design states the reason explicitly: "**miners are not aware of when they are being monitored, preventing them from selectively behaving correctly only during observed intervals**" **[E]**. A re-run that happens every round on a known schedule is a known quantity to design against; a re-run that happens with probability ~1/2 per round, at an unannounced point, is not. Same cost in expectation, strictly more information.

*Cost, stated honestly:* **the operator pays this pod.** Miners fund their own runs via `X-Lium-Api-Key`; a champion has no incentive to fund its own audit. At the repo's observed baseline (a 3-run wave plus ~14 failed provisions totalled **$0.97** on 1×5090 **[E]**) and a $2.5/h/pod cost guard, an eval-only re-run on 4×5090 is plausibly **~$3–8**. That is cheap against the cost of paying a full epoch's emissions to an overfit champion. *This is a real budget line, not a free lunch, and it should be funded explicitly.* **[J]**

**#3 — Retire the single-group live leaf.**

*Why:* the live default (`benchmarks`, v4) is an equal-weight arithmetic mean over G2 accuracies **only**. Every anti-Goodhart property the design claims — geometric-mean weak-axis elasticity, gates, mirror penalties on multiple groups, metric diversity — lives in the **composite** path, not the live one. Meanwhile G2 is the *worst* group at this scale: MDD 6.6–9.8 pp vs 1–3 pp real margins, and three tasks pinned at normalized 0 **[E]**.

*Concretely (interim, no anchor-set bump needed):* drop the three dead tasks (Winogrande, ARC-challenge, OpenBookQA) and BoolQ from the live leaf, raise `PRISM_EVAL_G2_CAP` to ≥1000 for LAMBADA/ARC-easy/PIQA, and add `mean_gold_nll`/Brier as observed. *Then* move to the composite once (a) the clustering bug is fixed, (b) anchors are measured, (c) seed variance is characterized. **Do not flip to `composite` before (a) — calibrating anchors against a zero-variance bootstrap bakes the bug into the anchor set.**

*Enforcement of diversity proper:* the geometric mean already prevents a zero axis from being compensated. Add a **pre-registered rule that no single group may supply more than a fixed fraction of a champion's *margin* over the runner-up** — if it does, the win is flagged and the challenger's displacement fails. **[S]** — I have not seen this deployed anywhere and it needs simulation before shipping; the geometric mean plus gates may already be sufficient.

### 4.4 Traps — defences that look good and are not

- **Randomizing probe parameters *between* the incumbent and the challenger.** If the champion was measured on epoch `E`'s slice and the challenger on `E+1`'s, the comparison confounds architecture with slice difficulty. **Rule: randomize across epochs, but always score the compared pair on the same draw within an epoch.** This is why #2 (re-running the champion) is not optional once slices rotate — it is what makes the pairing possible. Getting this wrong converts an anti-Goodhart measure into a noise amplifier. **[J]**
- **Adding more benchmark families "for diversity".** At ~160 M params most commonsense benchmarks are at or below chance **[E]**. Adding families that cannot discriminate *lowers* the composite's dynamic range and adds variance — the opposite of the intent. Add **resolution on axes that already move** (more items on LAMBADA/ARC-e/PIQA, NLL/Brier where accuracy is pinned) before adding new families. **[J]**
- **Tightening τ_m before measuring the gap distribution.** The mirror penalty subtracts `max(0, gap − τ_m)`. A τ_m tuned on `public_dev` (where gap ≡ 0) is uncalibrated for the private tier. Measure first.
- **Membership-inference, ECE, WeightWatcher α as scored metrics.** Near-chance / binning-biased / contested respectively **[E]**.
- **Novelty distance as a scored axis.** §5.2.
- **Treating the agentic LLM review as a grader.** `PRISM.md` is already explicit that it is a coherence gate, never a grader. Correct — an LLM judge on the scored path is a Goodhart surface with a prompt-injection attack attached.
- **Believing the CI half-width gate does anything today.** It is vacuous on G1/G2 (§1.3a). It will start rejecting submissions once clustering is fixed — expect a visible score drop across the board, and **calibrate anchors after, not before.**

---

## 5. Q4 — Rewarding genuine novelty

### 5.1 The core asymmetry

Copy-with-tweak is cheap against a **level**-based score because level is exactly what a copy reproduces. Three families of response exist, and only one has survived contact with adversaries in production:

| Approach | Production evidence | Verdict |
|---|---|---|
| **Detect** the copy (byte/AST/embedding similarity) | Evasion is measured and, for some tools, total — see below **[E]** | Necessary, insufficient, must not be load-bearing |
| **Measure** novelty and pay for it | Nobody has made this work. Numerai marketed it and implemented contribution instead **[E]**. Novelty search in EC is documented to reward meaningless diversity **[E]** | **Trap** |
| **Make the copy worthless** by construction | SN9's epsilon **[E]**; Numerai's residual MMC **[E]** | **This is the answer** |

The significance rule (§3.3c, §6.1) is the *primary* novelty defence: a copy has true Δ = 0 and therefore <5 % chance of any payout. **It requires no novelty measurement at all.** Everything below is secondary.

### 5.2 Why novelty-as-a-scored-axis is a trap

- **It is a maximizable target, so it will be maximized — and the measured evasion numbers are worse than "degraded". [E]** Sağlam et al. (ICSE-SEET 2024) generated obfuscated plagiarism (statement reordering/insertion) and measured detector separation between known-plagiarised and genuinely-unrelated pairs: **JPlag held (100 % vs 5.8 %, 94.2 pp separation)**, **Dolos collapsed to 12.4 pp**, and **MOSS scored obfuscated plagiarism at median 7.5 % — *below* the 4.5 % median for unrelated submissions.** Mossad (OOPSLA 2020) defeats MOSS/JPlag/Sherlock in minutes via dead-statement insertion, producing variants rated no more suspicious than legitimate work. A 2026 comparative study gives the cheap kill: identifier renaming + function reordering drops MOSS to 73 %; a bare `for`→`while` conversion drops it to 44 %.

  **The actionable asymmetry: JPlag survived what killed MOSS and Dolos.** So "similarity detection is hopeless" is too strong — *tool choice matters measurably*. Prism's `challenge-ast` band thresholds (≥8500 suspicious / ≥9500 cheat) should be validated against deliberately obfuscated positives before being trusted, and the JPlag-style token-sequence approach is the one with evidence behind it. **[E→J]**

  Learned embeddings do not rescue this: **ALERT achieves 27.79 % attack success on CodeBERT clone detection; GraphCodeAttack reaches 0.40 on clone detection and 0.841 on authorship attribution; code2seq predictions flip from added print statements** **[E]**.
- **Novelty search's own literature refutes the strong form.** Lehman & Stanley showed novelty-alone can beat objective-driven search in deceptive domains, but Cuccu & Gomez (EvoStar 2011) establish it **"does not scale to large search spaces"**, worked in the original maze only because position correlated with utility, and — decisively — **"one can always design a fitness function such that the solutions discovered by novelty alone perform arbitrarily badly."** MAP-Elites beat novelty-search-with-local-competition on all four criteria at **p < 1e-7** **[E]**.
- **It inverts the goal.** Prism wants novelty *that pays off*. A term that pays for difference pays for difference whether or not it works.
- **Legibility cost.** Numerai retired TC for opacity **[E]**. A novelty term computed from embeddings is exactly the kind of non-locally-verifiable component that failed there.

**Use novelty as a gate and a tie-break, never as a scored axis.** Prism's copy gate already does the gate half correctly, including the subtle right choices: same-hotkey/**same-coldkey** prior art excluded, strictly-earlier champions only, ties and unknown timestamps falling through, baseline exempt, and a bounded `precheck` (3/coldkey/UTC day) so miners can probe without burning their 1-max slot.

### 5.3 What to reward instead: per-axis marginal contribution

The transferable core of Numerai's MMC is **"pay for what the current frontier does not already have"**, not "pay for being different". The architecture-competition analogue:

**Maintain a per-axis elite archive.** For each group G1..G8, track the best score achieved by any accepted submission. A submission earns exploration credit if it **advances the frontier on at least one axis**, even if its composite is not first. Concretely: a submission ranked 3rd overall that is **1st on G3 (associative recall)** or **1st on G7 (inference cost)** has contributed transferable architectural insight and should be paid from the exploration pool.

Why this is the right shape:

- **Hard to game, because the axes are real measurements.** You cannot fake being best at algorithmic reasoning by renaming variables. Compare with AST distance, which you can fake in ten minutes. **[J]**
- **It matches the science.** The input report predicts a looped/recurrent-depth architecture will **win G3/G4/G5 and lose G7**, and that the A/B result is determined by which cap binds. Under WTA that architecture earns **zero** and its insight is discarded. Under a per-axis archive it earns something and the insight is published. **This is the concrete case where WTA actively destroys research value** — and it is Prism's own flagship experiment. **[E→J]**
- **It is a Quality-Diversity archive** (MAP-Elites) **[E]**, with the group structure supplying the descriptor space for free — and **the property that makes MAP-Elites safe here is that the cells are operator-defined: bounded, enumerable, and not inventable by participants.** That is the precise structural difference from a novelty-distance score, where the miner chooses the direction of "difference" and can therefore manufacture it. **A miner cannot create a ninth group to be best at.** **[E→J]**
- **It is legible**: a miner can compute "am I best on any axis?" locally from published per-axis scores.
- **It should be mandatory, not opt-in.** Numerai's documented failure mode: when contribution metrics were optional, large stakers simply staked the easy metric, so persistently harmful models never burned and the ensemble stopped improving — which is why MMC was made mandatory in 2024 **[E]**. An opt-in per-axis pool would be selected into only by those it favours.

**Caveat I want on the record.** A per-axis archive creates 8 sub-competitions, each with its own noise floor, and several Prism axes are weak instruments (G2 especially, and G6/G8 have known bugs per the input report). Frontier advances on a noisy axis will sometimes be noise. Mitigations: apply the **same significance test** to axis-frontier claims; exclude axes with known-broken anchors until fixed; cap the pool so a false positive costs ≤ its share. **[J]**

**The pairwise-correlation idea, and why I am cautious.** Gitcoin's pairwise-bounded matching directly penalizes *correlated behaviour*, which is the signature of copying, and is the only reviewed mechanism that attacks correlation head-on (measured ~60 % reduction in suspicious activity via Passport, not elimination) **[E]**. The analogue would be discounting rewards between submission pairs with highly correlated per-item error patterns. **I do not recommend it as a primary mechanism**: it makes each submission's payout depend on the rest of the field, reintroducing exactly the **field-dependence** that Prism's fixed-anchor design deliberately avoided (and which the repo's own aggregation research identifies as Sybil-attackable). Worth logging as **observed telemetry** — correlated error patterns are excellent *evidence* for the agentic reviewer — but not as a payout term. **[J]**

### 5.4 Verdict on architecture-owner credit: keep it off

**Recommendation: do not enable `OWNER_ARCH_CREDIT_ENABLED`. The current `false` is correct.**

Evidence:

- **tea.xyz is the definitive case.** Rewards routed over a dependency graph whose edges the beneficiaries create produced **~15,000** spam npm packages by Apr 2024, **14,000** across ecosystems, and an Amazon-Inspector-detected campaign of **>150,000 packages** by Oct–Nov 2025 — "one of the largest package flooding incidents in open source registry history". The mechanic was explicitly **circular dependencies to inflate impact scores**. Remediation lagged ~18 months. **[E]**
- **The structural comparison is what decides it.** thanks.dev routes over the *same kind of graph* without a farming catastrophe — because its payout is a **fixed philanthropic budget** and the edges come from the **donor's** repositories. **Prism's payout is a token with speculative upside, which puts it structurally in tea.xyz's position, not thanks.dev's.** **[E→J]**
- **Retroactive-funding precedents fail on attention, not math.** Optimism RetroPGF Round 3: badgeholders received **>15 DMs each**, 600+ projects, self-selected review ⇒ Optimism's own retrospective calls it "more like a popularity contest"; the anti-collusion quorum *became* the popularity mechanism. **[E]**
- **Prism's own code already names the attack**: with owner credit on, "a non-training / off-metagraph arch owner [could] steal or burn Prism's share via lex-tie".

**Important credit where due:** Prism's registry is *better positioned* than tea.xyz on the one dimension that matters most — **edges are operator-attested, not miner-declared.** An architecture publishes only after surviving every gate and reaching `terminated` with a real measured score; `arch_id` is a content hash; duplicates share the first registration; the copy gate makes later copies terminal. So the graph cannot be inflated with fictitious nodes. That removes the *node*-spam vector but not the *incentive* vector: with owner credit on, the profitable strategy becomes registering an architecture and having others (or your own second hotkey) train it, harvesting a share of work you did not do. The 1-max gate and coldkey-scoped precheck quota raise the cost of the self-deal but do not eliminate it. **[J]**

**If lineage credit is ever wanted, derive the edges from measurement — not from a declaration.** Shen & Barabási's collective-credit algorithm has the property tea.xyz lacked: credit is inferred from **third-party co-citation behaviour**, not asserted by the beneficiary, which makes edges costly to forge; it validates against Nobel attributions independently of author order **[E]**. The Prism analogue would be deriving lineage from **operator measurements of the trained artifact** (measured weight/activation similarity, AST distance computed by the harness) rather than a miner-supplied "I built on X" field. Caveat that keeps this out of my recommendation: Shen–Barabási is citation-count-based and therefore inherits RetroPGF's **popularity bias** — it allocates shares plausibly but does not size the pie, and popularity is the wrong signal for architectural merit. **[E→J]**

**Keep lineage as reputation, not as emission.** Owner attribution in `/v1/architectures`, the public leaderboard, the top-model publish to GitHub/HF — all good, all costless, all already built. **Route emission to whoever produced the measured improvement.** If the operator later wants to reward foundational architectures, do it as a **discretionary, capped, off-emission grant**, not as an automatic share of a token stream.

### 5.5 Two-stage tournament: yes, and it changes the calculus

Fu & Lu's result — **multi-stage beats single-stage when contest technology is noisy** **[E]** — applies directly, since a single-seed 6 h run is a noisy technology.

| Stage | Content | Cost | Who pays |
|---|---|---|---|
| **0 — free screen** | Existing pre-pod gates: copy gate, static cheat, cheap similarity, agentic sources review, plus (proposed) zero-cost proxies as *observed* signals | ~$0, no GPU | Operator (already built) |
| **1 — screen run** | Short standardized run (~1.5 h) → G1 bits/byte, G6 curve shape, G8 stability, G3 probes. Ranks into tiers, does not pay | ~1.5 h pod | **Miner** (already the model, via `X-Lium-Api-Key`) |
| **2 — confirmation** | Full 6 h run + full private battery, **with ≥2 seeds for finalists** | 2× 6 h pod | **Operator** — see below |

**Why ≥2 seeds at stage 2 is the point.** It is the *only* way to estimate `σ_seed` (§1.3c) and therefore the only way to make the significance test honest. A one-seed confirmation run cannot distinguish architecture from luck, no matter how many eval items it uses. **This single change does more for measurement validity than any additional metric.** **[J]**

**Who pays, and why it must be the operator.** Miners rationally fund their own screening run — it is their shot on goal, and the existing miner-funded model already works. But a **reliable ranking is a public good** for the subnet: every participant and every validator depends on it, no individual miner captures its value, and a miner asked to fund a second seed of their own submission has an incentive to fund the *luckier* one. **Operator-funded confirmation is what makes the number trustworthy.** Budget: at ~1.5 h screen + 2×6 h confirmation for ~3 finalists per epoch, the operator's marginal cost is roughly **36 pod-hours/epoch**; at the $2.5/h/pod guard that is ≲$90/epoch — small against a 50 %-of-subnet emission share. **[J]** (Cost estimate is arithmetic on the repo's stated price guard, not a market quote.)

**How staging changes the incentive calculus:**

- **Copying gets worse for the copier.** A copy must survive a paired, multi-seed confirmation against the incumbent — the regime where Δ = 0 is most visible.
- **Variance-farming gets worse.** A lucky screen draw is re-tested at stage 2 with independent seeds; the lottery ticket is voided.
- **Exploration gets cheaper.** A cheap screen means a speculative idea costs ~1.5 h to get a signal, not 6 h. This is the mechanism that most raises **discovery rate per dollar** (§6).
- **Coarse screening is fine, and theory says so.** Screening literature finds coarse/imperfect screening can be *beneficial* when up-front complexity is low **[E]** — so the screen does not need to be accurate, only cheap and unbiased.

---

## 6. Q5 — Research productivity of the subnet as a whole

The unit of value is the **rate of transferable architectural insight**, not one submission's score. Below, all numbers state their assumptions; several are arithmetic on stated assumptions rather than measurements, and I mark them.

### 6.1 How many submissions per epoch make the ranking meaningful?

Two separate requirements:

**(a) Enough submissions that the best-of-field plausibly exceeds the incumbent.** If a fraction `p` of serious submissions represent a genuine improvement detectable at MDD ≈ 2.3 %, then `P(at least one displacement in an epoch with k submissions) = 1 − (1−p)^k`. **[J]**

| `k` | `p` = 0.05 | `p` = 0.10 | `p` = 0.20 |
|---|---|---|---|
| 3 | 14 % | 27 % | 49 % |
| 5 | 23 % | 41 % | 67 % |
| 10 | 40 % | 65 % | 89 % |
| 20 | 64 % | 88 % | 99 % |

Reading: at `p` = 0.10 (one serious submission in ten is a real, detectable advance — optimistic given the NAS base rates in §2.3), **k ≈ 10 submissions/epoch** gives a ~65 % chance of a genuine champion change. Below `k ≈ 5` the mechanism mostly re-affirms the incumbent, and the design question stops being "how to rank" and becomes "how to attract entry" — which is itself an argument against WTA. **[J]**

**(b) Enough that the *paid band* is not mostly noise.** With 3–4 resolvable tiers (§3.2), paying 3 ranks requires `k` meaningfully greater than 3 or the "band" is just the whole field. **`k ≥ 6–8` is the point at which a top-3 band carries information.** **[J]**

**Consequence:** epoch cadence should be set so that a typical epoch accumulates **≥6, ideally ~10** scored submissions — not by the chain epoch. Prism's score-carry already decouples these (positive scores carry until superseded), which is the right primitive. **Make the *evaluation round* explicit** — e.g. a 7-day round with a pre-registered private slice — rather than letting it be an implicit consequence of carry. **[J]**

### 6.2 Pool the budget into fewer/longer runs, or spread into more/shorter?

**Spread.** Three converging lines:

1. **Metascience [E]:** impact is a decelerating function of funding; impact-per-dollar falls with grant size; a 92-paper systematic review leans toward dispersal.
2. **Compute-optimality [E, input report]:** the optimum at this budget is **N ≈ 160 M** with D/N ≈ 30 — *below* the existing 350 M cap. Longer runs at a fixed cap do not buy proportional information; a 1 B cap would cost ~+0.23 nats. So the marginal value of a longer run is *already* low at the current operating point.
3. **NAS multi-fidelity [E]:** τ ≈ 0.42 at 1 epoch, rising with fidelity. A cheap proxy gets the coarse ordering; the fine ordering needs fidelity. So **use cheap runs to reject, expensive runs to confirm** — exactly the two-stage design (§5.5).

**The floor on run length matters and is a real constraint.** Too short and you measure warmup rather than architecture — the input report's finding that G8's µP sweep is 10 steps of pure warmup is the cautionary example **[E]**, as is Porian et al.'s result that constant-step warmup mechanically penalizes small models. **Screen runs must be long enough to be past warmup and into a stable curve; ~1.5 h with warmup defined as a *fraction of total steps* is my proposal.** **[J]**

**Illustrative allocation** (assumptions: 4×5090 pod, ~$2.5/h/pod guard, one screen per submission, 3 finalists × 2 seeds):

| Regime | Screens | Confirmations | Pod-hours/epoch | Submissions/epoch |
|---|---|---|---|---|
| All-6h, no staging (current shape) | — | 10 × 6 h | 60 | 10 |
| **Two-stage (recommended)** | 10 × 1.5 h | 3 × 2 × 6 h | **51** | 10 |
| Two-stage, wider funnel | 20 × 1.5 h | 3 × 2 × 6 h | 66 | **20** |

The middle row costs **less** than the status quo while adding multi-seed confirmation; the third row **doubles the funnel** for ~10 % more pod-hours than the status quo. **This is the highest-leverage change available for discovery rate.** Note the split of who pays: screens are miner-funded (already the model), confirmations operator-funded (§5.5). **[J — arithmetic on stated assumptions]**

**Widening the funnel is also supported by theory, not just by cost arithmetic.** The Ladder's error bound depends on submission count `k` only **logarithmically**, while the binding term is holdout size `n` at an `n^{−1/3}` rate **[E]** (§2.4). So accepting more submissions per round costs almost nothing in statistical validity, whereas a small eval set costs a great deal. **This inverts the instinctive defence:** the reflex under adaptive-overfitting worry is to rate-limit submissions, but the mathematics says spend the effort on eval-set size instead and let the funnel be wide. Prism's 1-max-per-hotkey gate should therefore be understood as a **best-of-n / sybil** defence (which it is, and a good one) — **not** as an adaptivity defence, which it barely affects. **[E→J]**

### 6.3 Portfolio / diversity of the accepted set

A subnet whose accepted set is ten variants of one architecture has a discovery rate near zero regardless of scores. Measure and manage it:

- **Publish a diversity statistic per epoch** — e.g. occupancy of the per-axis elite archive (how many distinct submissions hold at least one axis frontier), and pairwise lineage distance among the paid band. Observed-only, never scored (§5.2).
- **The exploration pool (§7.1) is the instrument** that buys diversity: it pays for distinct, gate-passing, axis-advancing entries.
- **Trap:** do not enforce diversity by *rejecting* similar submissions. That converts a portfolio goal into an eligibility fight and hands power to a similarity metric that is gameable in both directions (both to appear novel and to frame a rival as derivative). Pay for diversity; do not police it. **[J]**

### 6.4 Measuring the thing itself

Proposed operator dashboard, per epoch — all cheap, all derivable from data Prism already stores:

| Indicator | Definition | Why |
|---|---|---|
| **Displacement rate** | Fraction of epochs with a significant champion change | Directly measures whether the mechanism is discovering |
| **Frontier advance** | Best composite this epoch − best ever (paired, on the same slice) | Progress net of noise |
| **σ_seed** | sd across replicate seeds of the baseline | **The calibration input for everything else.** Must be re-measured each epoch |
| **Archive occupancy** | Distinct submissions holding ≥1 axis frontier | Portfolio diversity |
| **Champion durability** | Epochs held before displacement | Detects squatting (too high) or noise-churn (too low) |
| **Re-run regression** | Champion's score drop on fresh private slice | Direct anchor-overfit signal (§4.3 #2) |
| **Mirror gap distribution** | Realized `x_public − x_mirror` across submissions | Contamination pressure; also calibrates τ_m |
| **Entry rate / EV** | New hotkeys per epoch; realized payout ÷ pod cost | Is participation economically viable? |

**The one I would watch hardest is `σ_seed`.** If it comes in at the high end (2.5 % relative), the honest conclusion is that a single 6 h run **cannot** rank architectures at the precision the current design assumes, and the response is more seeds per submission rather than more metrics. That would be a genuinely unwelcome finding and it should be allowed to surface. **[J]**

### 6.4b The measurement nobody has published — and Prism is unusually well placed to make it

**Every cross-fidelity rank-correlation number in §2.3 comes from CNN image benchmarks (NAS-Bench-101/201, CIFAR, ImageNet-16). There is no published low-vs-high-fidelity rank-correlation study for LM pretraining at ~350 M params on a fixed wall-clock budget.** That gap is directly load-bearing for Prism: the entire justification for a 6 h screen rests on numbers measured in a different modality, at a different scale, with a different budget dimension being cut.

**The experiment is cheap and Prism is the natural venue:** take ~30 submissions already scored at 6 h, re-run them at 4–6× the budget, and report **both** global Spearman ρ **and** precision@top-3 (the two-regime distinction of §2.3 is invisible in ρ alone, which is exactly how the NAS literature initially missed it). Cost is ~30 × 30 h ≈ 900 pod-hours — a real number, but a one-time one, and it answers the question the whole mechanism is premised on.

**This is also the single most valuable thing Prism could publish externally** (§6.5), and it bears directly on Q5's "compound external research value": it is a genuine, citable contribution to the NAS/scaling-law literature that only an operator running a standing architecture competition can produce. **[J]**

### 6.5 What the operator should publish

Prism already publishes the top model to GitHub and a reloadable pack to HuggingFace, with journaling. Extend, in value order:

1. **All scored submissions, permanently, non-retractable** — scores, per-axis breakdown, gate status, CIs. This is the Leaderboard Illusion fix **[E]**, and today the public gallery lists **champions only** (`Score>0`), which is a selective-disclosure shape even though it is not selective *by intent*.
2. **Negative results — and there is a sharper incentive argument here than "publishing failures is nice".** Registered reports produce **43.66 % positive first-hypothesis results vs 96.05 % in standard reports** (Scheel et al. 2021), and Allen & Mehler find **61 % of 296 preregistered hypotheses unsupported** vs 5–20 % in the general literature **[E]**.

  **The consequence for Prism is a genuine tension, not a footnote:** Prism's anchor pre-registration *is* a registered-report structure, so it should expect **~55–60 % of honest, well-specified attempts to fail.** If a well-specified null pays **nothing**, then the rational miner submits only safe epsilon-tweaks of the champion — **which is precisely the copy-with-tweak attack the whole design is trying to suppress.** So the exploration pool is not a diversity nicety; it is the term that makes honest high-variance research rational. This is the strongest argument I have for keeping it, and it emerged from the evidence rather than from the mechanism sketch. **[E→J]**

  Guard against the obvious abuse: pay for **informative** nulls only — gate-passing, lineage-distinct, conclusively measured (adequate `#decided`, no partial-battery truncation) — never for "I submitted something and it lost".
3. **The private slice, released after the round closes**, with its generation seed — so results are reproducible after the fact without being gameable during.
4. **σ_seed and the MDD, per epoch.** Publishing the noise floor makes the significance rule auditable and pre-empts "I was robbed" disputes.
5. **A standing statement of what the ranking does and does not claim** (§2.3): Prism ranks architectures **at a pinned small budget**; Tay et al. is direct evidence that such rankings need not transfer across scale **[E]**. Publishing this *increases* credibility with the research audience Prism wants.
6. **Reproducible artifacts** — parked checkpoints with receipts (already built), plus the exact patch and pin.

---

## 7. Recommended mechanism, with concrete parameters

### 7.1 The rule

**Per evaluation round (target: 7 days, ≥6 scored submissions):**

**The displacement test is a paired per-example comparison, following SN56's live design [E] rather than my original level-difference formulation.** Champion `A` and challenger `B` are scored on the **identical** private slice for the round, each with ≥2 seeds at confirmation.

```
Per eval example i:  d_i = bpb_A(i) − bpb_B(i)        (positive ⇒ B better)

DECIDED(i)   iff  |d_i| ≥ deadzone                    (absolute, in bits/byte)
win_rate     = #{i : d_i > 0, DECIDED(i)} / #DECIDED
mean_gap     = mean{ d_i : DECIDED(i) }

B displaces A  iff   LCB_99%( win_rate )  ≥  min_win_rate      # clustered paired bootstrap
                AND  mean_gap             ≥  min_mean_gap      # absolute, not relative
                AND  #DECIDED             ≥  min_decided
                AND  mean_gap             ≥  econ_floor(t)     # tenure-decaying policy bar
```

**Two bars, following SN56 [E]:** clearing the above **holds/transfers the crown**; a strictly higher bar (`premium_gap`) is required to earn *above* the base floor, with the excess **burned**. This separates "who is champion" from "how much the champion is paid", so a marginal-but-real win does not automatically unlock the full share.

**Emission split of the round's prism share `P`:**

| Tier | Share | Recipient |
|---|---|---|
| **Champion** | **60 %** | A, or B if B displaced A. Scaled down toward the base floor if `mean_gap < premium_gap`; remainder burns |
| **Runner-up band** | **15 % / 10 % / 5 %** | Ranks 2/3/4 among gate-passing submissions |
| **Exploration pool** | **10 %** | Up to 5 submissions that pass all gates **and** advance ≥1 per-axis frontier (same paired test), one per hotkey, split equally |
| Unallocated | **burn** | Consistent with existing fail-closed burn semantics |

**Parameters. Where SN56 has published production calibration and I have not, I use their values and say so.**

| Parameter | Value | Basis |
|---|---|---|
| Comparison | **paired, per-example, on the identical slice** | SN56 **[E]**; parameter-free Ladder **[E]**; SN9 per-batch wins **[E]** — three independent convergences |
| `deadzone` | **0.01 bits/byte, absolute** | SN56 uses 0.01 **nats** for the same purpose **[E]**. **Absolute, not relative** — "D nats is D nats of evidence whether the loss is 0.02 or 2.0"; a relative margin collapses exactly where the metric saturates. This **supersedes** the `1 % relative` I first proposed |
| `min_win_rate` | **0.55** at a **99 %** one-sided bootstrap LCB | SN56 live value **[E]**. Deliberately *not* higher: "demanding much more selects for low-variance submissions rather than good ones" |
| `min_mean_gap` | **= deadzone** (0.01 bits/byte) | SN56 sets them equal after a documented false negative where a larger second threshold rejected a model better by 0.011 nats on **100 %** of 800 samples **[E]** |
| `min_decided` | set so the win-rate SE ≤ ~5 % | SN56's rationale: "the gate is automatically stricter when it has less to go on" **[E]** |
| Bootstrap | **10 000** resamples, **fixed seed**, clustered on eval clusters | SN56 **[E]** — a fixed seed makes the verdict reproducible by anyone re-scoring, which is the local-verifiability property Numerai's TC lacked (§2.2) |
| `econ_floor(t)` | starts at ~**1× the measured median separation**, decays with tenure | **Calibrate against Prism's own measured separation distribution before choosing a number.** SN56's calibration (62 tasks / 120 days, median separation **0.0094 nats**) showed a bar above the median makes the crown near-permanent **[E]**. I decline to invent Prism's value — it must be measured (§7.3 step 2) |
| Champion decay | **~0.15 %/day** of the advantage, or a linear decay to a small floor over ~2–5 days | Three live implementations to interpolate from **[E]**: SN56 **0.165 %/day** (halved from 0.33 %); SN9 `LinearDecay(0.005 → 0.0001)` over **~2 days**; SN37 `LinearDecay(0.05 → 0.01)` over **1–5 days**. **Note the shape: linear decay to a nonzero floor, not exponential to zero** — the floor keeps a minimum copy deterrent permanently in place |
| Statistical term | **never decays** | A truth condition, not a policy preference. SN56 decays *emission*, not the comparison; that is the correct split |
| Champion share | **60 %** | Below SN9's 95 % (which produced hoarding **[E]**); SN56 floors the base pool at 50 % **[E]** |
| Band depth | **3 runners-up** | ≈ the instrument's 3–4 resolvable tiers (§3.2); SN9's realised collapse also pays ~3 **[E]** |
| Exploration pool | **10 %**, ≤5 slots, 1/hotkey | Bounded so a false-positive frontier claim costs ≤2 % of `P` |
| Weight EMA | **α ≈ 0.5–0.9** on the emitted vector | SN9 `alpha=0.5`, SN37 `ALPHA=0.90` **[E]** — temporal smoother; composes with, does not replace, the significance test |
| Tail floor | zero out shares below a fixed threshold | SN37 `MIN_WEIGHT_THRESHOLD=0.18` **[E]** — avoids paying unresolvable rank differences |
| Seeds at confirmation | **≥2** (3 preferred) | The only way to estimate σ_seed (§1.3c, §5.5) |
| Round length | **7 days**, ≥6 scored submissions | §6.1; SN56 uses `TOURNAMENT_INTERVAL_HOURS = 120` (5 days) **[E]** |
| Private tier | **mandatory**, freshly regenerated per round | §4.3 #1 |
| Champion re-run | **every round, unannounced timing**, eval-only | §4.3 #2; IOTA's unannounced-monitoring rationale **[E]** |
| Eval near-duplicate cap | reject a slice with ≥**20 %** near-duplicate rate | SN56 `MAX_NEAR_DUPLICATE_RATE = 0.20` **[E]** — the most concrete anti-contamination parameter found |

**Note on units.** The rule above is written on `bits_per_byte` because that is Prism's tokenizer-neutral, lower-better dominant axis, and because a per-example paired difference is well-defined there. For the accuracy-style groups (G2–G5) the same paired machinery applies to per-item correctness with the dead zone set to 0 (an item is decided or it is not). **The composite `C` is the wrong object to run the displacement test on** — it is a weighted geometric mean of group aggregates, so it has no per-example decomposition. Run the test on the axis, then require the composite not to regress. **[J — this is a structural consequence of Prism's composite shape and needs care in implementation.]**

### 7.2 Why these numbers and not others

- **Why 60/15/10/5/10 rather than 100/50/25 normalized?** Top-3 decay alone gives a copy a *reliable* band position (§3.3b). Pairing the band with a significance gate means a copy cannot reach the champion share. **It can still reach the band** — SN9 measured exactly that outcome under an epsilon rule **[E]** — which is why the band is capped at 30 % total and why the copy gate remains necessary rather than optional. Sizing the champion share at 60 % is the deliberate trade: large enough that the protected slot carries most of the value, small enough that losing it does not zero a serious participant.
- **Two cheap complements from SN9's implementation that I would ship alongside [E]:** an **EMA on the emitted weight vector** (SN9 `alpha = 0.5`, SN37 `0.90`) so handover is smooth and a single anomalous round cannot swing emission, and a **minimum-weight floor** that zeroes the tail (SN37 `MIN_WEIGHT_THRESHOLD = 0.18`) so unresolvable rank differences are not paid at all. The EMA is a *temporal* smoother and the significance test is a *statistical* one; they compose, and neither substitutes for the other.
- **Why not pay rank 5+?** Below the 3–4 resolvable tiers, rank ordering is noise (§3.2), and the NIH evidence (AUC 0.54) is the general case **[E]**.
- **Why does the statistical term never decay?** It is a truth condition. Decaying it means knowingly crowning champions on noise. SN9's ε conflates the statistical and economic roles into one constant; **SN56 gets this right by decaying emission rather than the comparison threshold [E]**, and I follow them.
- **Why absolute rather than relative margins?** Because a log-likelihood difference is evidence at a fixed scale: "D nats is D nats of evidence whether the loss is 0.02 or 2.0" **[E]**. A relative margin vanishes precisely where the metric saturates — which is where Prism will spend most of its time as the field converges. **This corrected my initial proposal.**
- **Why not set the win-rate bar high for safety?** Because it is not free safety: a genuinely better architecture with wide per-example spread sits near 55 %, so a high bar **selects for low-variance submissions over good ones** **[E]**. Over-tightening is the mirror image of variance-farming, not its cure.
- **Why burn the remainder?** It matches the existing gateway burn-vector semantics, and ARC Prize's unclaimed grand prize is the precedent that gates can legitimately stall payouts **[E]**.

### 7.3 Implementation cost

Small, and mostly in one place. `apply_wta` is ~20 lines with existing test coverage; `competition_scores` already returns a **map** of per-hotkey credits and `apply_wta` is what collapses it. Replacing the collapse with a graded allocator is a localized change plus tests. The expensive parts are **not** code: mandatory private staging per round, operator-funded confirmation runs, and seed replication.

**Sequencing — this order is not negotiable:**

1. Fix G1/G2 bootstrap clustering (per-item ids, matching the already-correct mirror path). **No anchor bump.**
2. Measure `σ_seed` with baseline replicates. **Prerequisite for any significance rule.**
3. Stage the private tier; make it mandatory for scored epochs.
4. Fix the G6 anchor/censoring bugs and the warmup-fraction rule (per the input report) — they distort the very rankings the new rule would act on.
5. *Then* ship the significance-gated graded collapse.
6. Champion re-runs, exploration pool, publication changes.

**Doing (5) before (1)–(2) would be actively harmful:** a significance test built on a variance estimate that is provably wrong on 40 % of the composite's weight, and that omits the dominant noise term entirely, would give false statistical authority to a biased ranking. That is worse than honest WTA, because WTA at least does not claim to be significant.

---

## 8. Transferable mechanisms table

| # | Mechanism | Source / evidence | Prism status | Verdict |
|---|---|---|---|---|
| 1 | Epsilon threshold to displace incumbent | SN9 ϵ=0.5 %, 7B★ 0.1 % **[E]** | **Cheap to add** | **Adopt, improved** — make it noise-calibrated, split stat/econ terms |
| 1b | **Paired per-example displacement test + bootstrap LCB + dead zone** | **SN56 live** (`deadzone 0.01 nats`, `min_win_rate 0.55` @99 %, `min_mean_gap`, fixed bootstrap seed) **[E]** | **Cheap to add** | **Adopt as specified** — the single highest-value import in this table |
| 1c | **Absolute (nats) not relative (%) margins** | SN56 in-code rationale **[E]** | Absent | **Adopt** — corrects my own first proposal |
| 1d | **Two bars: hold-the-crown vs earn-premium-emission**, remainder burns | SN56 base floor 50 %, burn bar raised 0.05→0.10 **[E]** | Absent | **Adopt** |
| 1e | Calibrate the bar against the **measured separation distribution** | SN56: 62 rounds/120 days, median 0.0094 nats **[E]** | Absent | **Adopt as process** — do not guess the value |
| 1f | Equalized compute + pinned commit + no-internet containers | SN56 **[E]** | **Has** (pinned pin+patch, netns isolation, operator-run pods) | Keep — Prism is already here |
| 1g | Paid entry as anti-sybil | SN56 fees 0.2–0.35 TAO **[E]** | **Has implicitly** (miner-funded pod) | Keep; revisit if band-slot farming appears |
| 1h | Rank-rating instead of raw scalar (OpenSkill) | Templar SN3 **[E]** | Absent | Alternative to 1b; **1b is better suited** to a fixed battery |
| 1i | Eval-slice near-duplicate cap | SN56 `MAX_NEAR_DUPLICATE_RATE = 0.20` **[E]** | Absent | **Cheap to add** |
| 1j | LLM-based submission dedup alongside the margin rule | SN56 (Claude Opus, $15/round budget) **[E]** | **Has** (cheap similarity + agentic review) | Keep — confirms a margin rule does not replace a detector |
| 2 | Decaying epsilon vs hoarding | SN9 §3.4.2 **[E]** | Cheap to add | **Adopt** — decay econ term only |
| 3 | Earlier-upload tie-break / incumbency | SN9 `isWin` **[E]** | Has (score carry) | Keep; but re-measure the champion (winner's curse) |
| 4 | WTA at 95 %+ | SN9 **[E]**; abandoned in IOTA **[E]** | **Has (100 %)** | **Trap** — its own authors documented hoarding |
| 5 | Ladder mechanism (release only significant improvements) | Blum & Hardt **[E]** | Cheap to add | **Adopt** — the paired parameter-free variant |
| 6 | Thresholdout / reusable holdout | Dwork et al. **[E]** | Cheap to add | Adopt in spirit (rate-limit information release) |
| 7 | Paired difference testing (exploit 0.3–0.7 item correlation) | Clustered-SE eval work **[E]** | **Cheap to add** | **Adopt** — strictly better than independent LCB |
| 8 | LCB / variance-priced ranking | Prism v3 **[E]** | **Has** | Keep, but recalibrate (clustering + seed variance) |
| 9 | Fixed pre-registered anchors | BIG-bench, Open LLM v2 **[E]** | **Has** | Keep — foundational |
| 10 | Pre-registration / registered reports | 96 %→44 % positive results **[E]** | **Has** (`/v1/preregistration`) | Keep |
| 11 | Private, freshly-regenerated final eval | ARC Prize, Kaggle **[E]** | **Partially** (machinery exists, not mandatory) | **Adopt — highest value** |
| 12 | Template regeneration (GSM-Symbolic style) | GSM-Symbolic, GSM1k **[E]** | **Has** for G3/G4 generators | Keep; extend rotation per round |
| 13 | Mirror-gap contamination penalty | Prism design | **Has but inert by default** | **Fix by staging private tier** |
| 14 | "Prove it again" champion re-runs | Winner's curse **[E]**; no direct precedent found | **Cheap to add** (eval-only) | **Adopt — 2nd highest value** |
| 15 | Lexicographic gates | MLPerf 99 %/99.9 %, ARC 85 % **[E]** | **Has** | Keep — but note they subsidize variance-farming under WTA |
| 16 | Geometric mean / power mean p<0 | Fleming–Wallace; Jigsaw p=−5 **[E]** | **Has** in v3; **absent in live v4** | **Adopt in the live leaf** |
| 17 | No scalar aggregate (dashboard only) | MLPerf **[E]** | N/A — emissions need a scalar | Hybrid: scalar for emission, dashboard for humans |
| 18 | Mean win rate / Borda | HELM, abandoned **[E]** | Absent | **Trap** — field-dependent, Sybil-attackable |
| 19 | Bradley–Terry / Elo on miner-vs-miner | Chatbot Arena **[E]** | Absent | **Trap** — collusion + selective disclosure |
| 20 | Publish all scores, no retraction | Leaderboard Illusion **[E]** | Partially (champions only publicly) | **Adopt fully** |
| 21 | Bounded best-of-n (submission caps) | Kaggle 2-final-subs **[E]** | **Has** (1-max + precheck 3/coldkey/day) | Keep — well designed |
| 22 | Two-stage cheap-screen → expensive-confirm | Fu & Lu **[E]**; Kaggle **[E]** | Partially (pre-pod screens exist) | **Adopt** — add screen tier + multi-seed confirm |
| 23 | ~2 finalists to maximize peak effort | Shortlist theory **[E]** | Absent | Partially adopt (narrow confirmation tier) |
| 24 | Merit band + lottery inside band | Volkswagen null result **[E]**; HRC NZ **[E]** | Absent | **Adopt as band logic**, not as a literal lottery |
| 25 | Marginal contribution (MMC-style) | Numerai 7 yrs **[E]** | Absent | **Adopt as per-axis frontier credit** (literal residualization does not transfer) |
| 26 | Pay for measured originality/novelty distance | Numerai marketed, never implemented **[E]**; MOSS scores obfuscated plagiarism **below** unrelated-pair baseline **[E]** | Absent | **Trap** |
| 26b | **JPlag-style token-sequence similarity** (survived what killed MOSS/Dolos: 94.2 pp vs 12.4 pp separation) **[E]** | — | Has AST bands (unvalidated vs obfuscated positives) | **Cheap to add** — validate `challenge-ast` thresholds against deliberately obfuscated positives |
| 27 | Novelty search for its own sake | Cuccu & Gomez: "arbitrarily badly"; does not scale **[E]** | Absent | **Trap** as an objective; fine as archive structure |
| 28 | Quality-Diversity / MAP-Elites archive | MAP-Elites > NSLC on all 4 criteria, p<1e-7 **[E]** | Absent | **Adopt** — G1..G8 supply the descriptor space, and **cells are operator-owned so miners cannot invent one to farm** |
| 28b | Make the contribution metric **mandatory, not opt-in** | Numerai: opt-in ⇒ nobody staked it ⇒ ensemble stopped improving **[E]** | N/A | **Adopt** for the per-axis pool |
| 28c | Pay for **informative nulls** | Registered reports: 43.66 % vs 96.05 % positive rate **[E]** | Absent | **Adopt** — otherwise epsilon-tweaks are the only rational submission |
| 28d | Lineage edges from **third-party/measured** evidence, not declaration | Shen & Barabási (PNAS 2014) **[E]** | Partially (operator-attested registry) | Structurally right; inherits popularity bias — **not recommended as emission** |
| 29 | Shapley / leave-one-out contribution | Data Shapley; 2^(N−1) exact, N retrainings LOO **[E]** | Absent | **Trap at this budget** — cost wall |
| 30 | Lineage/owner royalty over a dependency graph | tea.xyz >150k spam packages **[E]** | **Correctly disabled** | **Trap — keep disabled** |
| 31 | Retroactive public-goods funding | RetroPGF popularity bias **[E]** | Absent | **Trap** at this scale (attention-bounded) |
| 32 | Quadratic funding | Gitcoin sybil; Passport ~60 % reduction **[E]** | Absent | Not applicable (no donor side) |
| 33 | Pairwise-bounded matching (penalize correlation) | Gitcoin **[E]** | Absent | **Interesting, not primary** — reintroduces field-dependence |
| 34 | Operator-attested graph edges | tea.xyz vs thanks.dev contrast **[E]** | **Has** (registry publishes only after measured score) | Keep — this is why Prism's registry is safer than tea's |
| 35 | Prediction markets over pending submissions | Dreber et al. **[E]** | Absent | **[S]** — governance add-on, not a scoring path |
| 36 | Publish negative results | Registered-reports evidence **[E]** | Absent | **Adopt** — cheap, compounds external value |
| 37 | Zero-cost NAS proxies as a screen | NAS-Bench-Suite-Zero **[E]** | Absent | Adopt as **observed** pre-pod signal only |
| 38 | Weight-sharing / one-shot proxy scoring | Rank-correlation failures **[E]** | Absent | **Trap** — Prism's standalone training is correct |
| 39 | Membership inference / ECE / WeightWatcher α scored | Duan et al.; binning bias **[E]** | Absent | **Trap** |
| 40 | LLM-as-judge on the scored path | — | Absent by policy | **Trap** — Prism's coherence-gate-only stance is correct |
| 41 | Saturation tripwires (pre-declared reweight) | Benchmark Lottery **[E]**; SN37 ships **block-scheduled competition sunsets** in code **[E]** | Absent | **Cheap to add** — copy SN37's pre-committed-block form |
| 43 | **Unannounced** verification timing | IOTA: miners "not aware of when they are being monitored" **[E]** | Absent | **Cheap to add** — applies to the champion re-run (§4.3 #2) |
| 44 | EMA on the emitted weight vector | SN9 `alpha=0.5`; SN37 `ALPHA=0.90` **[E]** | Absent | **Adopt** — temporal smoother, composes with the significance test |
| 45 | Minimum-weight floor zeroing the tail | SN37 `MIN_WEIGHT_THRESHOLD=0.18` **[E]** | Absent | **Adopt** — avoids paying unresolvable rank differences |
| 46 | Forced shared tokenizer for comparable loss | SN37 `Xenova/gpt-4` **[E]** | **Better solved** — tokenizer-neutral `bits_per_byte` | Keep Prism's approach |
| 47 | Padding to satisfy a size band | SN9 measured param-padding + copies of padded models **[E]** | N/A — `MAX_PARAMS` is a ceiling, not a band | Keep as ceiling; general lesson: gates are met the cheapest way that passes |
| 42 | Divisions with separate pools (fixed-recipe vs open) | MLPerf closed/open **[E]** | Absent | **[S]** — good idea, premature at current submission volume |

---

## 9. Risks, and where this analysis is weakest

### 9.1 Top risks of the recommended design

1. **`σ_seed` may be large enough to invalidate the whole ranking premise.** If replication shows `σ_seed ≈ 2.5 %` relative, then MDD ≈ 5.8 % and **essentially no realistic architectural improvement is detectable in one 6 h run.** The mechanism would then be paying for noise no matter how the collapse rule is written. Mitigation: measure it *first* (§7.3 step 2); if it is that large, the response is more seeds per submission (fewer submissions, more replication) — a strictly different design than the one above. **This risk invalidates more of this document than any other, so it is listed first.**
2. **Incumbent squatting under significance gating — and this one has been observed in production, so it is not hypothetical.** SN56 calibrated their displacement bar against 62 boss rounds over 120 days and found that a threshold above the **median separation (0.0094 nats)** meant "only a quarter of matchups were close enough to even be winnable, which compounded over 4-of-5 tasks makes the crown **near-permanent**" **[E]**. They also recorded a false negative where a model better on **100 % of 800 samples** lost the task **[E]**. Mitigations: set the economic floor **at or below Prism's own measured median separation** (not at a guessed value), decay it with tenure, run unannounced champion re-runs, and keep the graded band so challengers stay solvent. Residual risk: if the true improvement rate is genuinely low, squatting is the *correct* outcome and will still look like stagnation — **indistinguishable from "nobody has built anything better", which is politically hard even when it is the honest answer.**
3. **Operator cost and discipline become load-bearing.** Mandatory private staging per round, operator-funded multi-seed confirmations, and champion re-runs are all recurring operator obligations. If any lapses, the mechanism silently degrades to its unprotected form — `eval_tier` falls back, mirror gaps go to zero, significance tests run on one seed. Mitigation: fail-closed on tier (refuse to score a non-private epoch) rather than warn.

**A tension between two of my own recommendations, which I should not leave implicit.** §4.3 #1 says make the private tier mandatory and rotating; Numerai's TC postmortem says **participants must be able to compute the objective locally or they cannot optimize it** — and that legibility failure is what killed a conceptually better metric **[E]**. A hidden, rotating eval slice directly reduces local computability. The reconciliation I would ship: **keep the *rule* fully public and locally recomputable** (the paired test, dead zone, win-rate bar, bootstrap seed, anchors, weights — all pre-registered and hash-committed), and let only the *slice contents* be private, **released after the round closes** so every past verdict is independently auditable. That preserves "you can check the arithmetic and reproduce last round" while denying "you can fit this round". Numerai reached the same resolution by publishing the Meta Model signal so MMC became locally computable. **[E→J]**

**One landscape fact that should temper expectations about the whole category.** No live Bittensor subnet is doing genuine NAS: **SN31 (NASChain) went dormant and SN49 (Hivetrain) was deregistered** **[E]**. The structural cause named in the research is uncomfortable and worth stating: **subnet survival depends on alpha token price, not on research output.** A mechanism that maximizes research productivity but not token-holder attention can be selected against regardless of its scientific merit. That is outside what an emission rule can fix, but it argues for the publication programme in §6.5 being treated as a survival requirement rather than a nice-to-have. **[E→J]**

Further risks worth tracking: **complexity as an attack surface** — the Jigsaw precedent is that competitors optimize the exact published functional form **[E]**, and Numerai retired TC for opacity **[E]**; every term added is a term to be gamed and a term miners cannot verify locally, which argues for shipping the champion/band/pool split and *stopping there*. **Exploration-pool farming** is bounded by miner-paid pod cost (~$5–15/submission), the 1-max gate, gate-passing requirements, and the ≤2 %-of-`P` per-slot cap — but if the pool's per-slot payout ever exceeds pod cost by a wide margin, expect gate-passing-but-uninteresting submissions; keep the ratio near 1–3× and re-check as token value moves. **[J]** And **graded emission changes the on-chain weight vector shape** — multiple positive scores per epoch instead of one; I have reasoned that this is safe because Prism's weights originate at a single sealed master gateway rather than from independent per-validator scoring, but **I have not traced the full seal/aggregation path and this needs verification before shipping** **[J]**.

### 9.2 Where the evidence is thin, and where I may be wrong

- **I got the contest theory wrong on first pass, and the correction runs against my own recommendation (§3.4).** I claimed Prism's rank-determining shock is heavy-tailed (idea quality) and therefore in Drugov–Ryvkin's *sharing* regime. That substituted the wrong random variable: the shock whose tail governs the optimal prize schedule is the **measurement/luck term corrupting the effort→rank map**, which for Prism is light-tailed/IFR — so **applied correctly, D–R favours concentration.** The recommendation survives on copy-EV, participation-margin, tier-resolution, and the SN9/IOTA field evidence, **but it no longer has the tail argument behind it, and a reviewer who weights contest theory heavily could defensibly concentrate further.** This was the most load-bearing error in the draft; it was caught in review rather than by me, and the residual disagreement is documented at the end of §3.5 rather than resolved in my favour.
- **What is still worth measuring, and why it is no longer decisive.** Prism's realized submission-quality distribution is unmeasured (no corpus yet). It matters for how many tiers are worth paying and for the Terwiesch–Xu diversity argument, but it is **not** the D–R test — I previously conflated the two. **[J]**
- **The composite's tail behaviour is unanalysed. [S]** Prism's composite is a gated weighted geometric mean, which could have a fatter lower tail than its light-tailed components; that would bear on the D–R condition. I have not analysed it and did not use it as an argument.
- **The copy-EV numbers are derived, not observed.** "0.5·P under WTA" assumes a functional copy has identical true quality and symmetric noise; "<5 %" is the one-sided test size. Both are arithmetic on stated assumptions. Real copies differ slightly (that is the "tweak"), so a real copy's true Δ is small-positive rather than exactly zero, which raises its win probability above the nominal α. **The direction and order of magnitude are robust; the exact numbers are not.** **[J]**
- **No direct precedent for "prove it again" re-runs.** I found no deployed system that re-runs past champions on fresh private slices as an anti-overfit device. The reasoning is sound and the cost is low, but this is my construction, not an imported mechanism — treat it as untested. **[J]** (The *unannounced-timing* half of it is imported from IOTA **[E]**.)
- **Three things that started as my judgement and ended up corroborated**, which is worth recording so the confidence labels are not read as uniformly soft: the paired-comparison displacement rule (SN56 ships it, with published calibration), the separation of statistical from economic margin (SN56 decays emission not the threshold; SN37's blended ε is visibly compromised by doing both jobs at n=120–150), and the seed-variance concern (four independent systems — SN9, SN56, Templar, and the NAS seed-swap τ=0.48 result — converge on the raw noisy scalar being unusable).
- **What I could not verify at all:** the on-chain weight-vector implications of graded emission for Prism's sealed-master path; whether Prism's AST bands survive deliberately obfuscated positives; and the claim that SN9's 7B competition began at ε = 3 % (contradicts the whitepaper's 0.5 %; appears only on third-party aggregators).
- **The metascience dispersal evidence measures publications, not capability.** Fortin/Mongeon/Aagaard measure papers and citations. The deep-learning scaling literature points the *opposite* way for exploitation. My reconciliation (disperse for search, concentrate for exploitation) is inference, not a measured finding. **[J]**
- **Roelofs genuinely cuts against the anti-Goodhart case, and I am relying on a conditional reading.** Their headline is "little evidence of substantial overfitting". My argument that Prism falls in their *exception* category (effectively small test sets) is well-supported by their own stated caveat, but it is an inference about which side of their boundary Prism sits on. Someone could reasonably read Roelofs as "relax, holdouts are robust". I think that reading is wrong for n=200 with recurring emissions, but it is not a settled question.
- **The 3–4 resolvable tiers estimate depends on an unmeasured quality spread.** I assumed 0–5 % relative true spread among serious submissions, anchored on the Cerebras 111M→256M step. If real spread is wider, more ranks are resolvable and a deeper band is justified; if narrower, even top-3 is too deep.
- **NAS cross-fidelity τ numbers come from vision/CIFAR NAS spaces**, not LM architecture search at 160 M params. Direction (coarse ordering survives, fine ordering does not) is very likely to transfer; the magnitude (τ ≈ 0.42) should not be quoted as Prism's number.
- **Tay et al. is the finding I would most like to be wrong about**, because if architecture rankings routinely flip between 4.5e18 FLOPs and deployment scale, then Prism's output is of limited transfer value regardless of how well its mechanism works. It measured up to 40 B params across 10 architectures, so it is strong evidence. The honest position is that Prism selects architectures **at a pinned small budget** and should say so.
- **I did not verify the on-chain weight-vector shape implications** (§9.1) or the gateway seal path for graded emission.
- **Numbers I did not independently verify** and am relaying from the input report: the G2 MDD table, the compute-optimal N ≈ 160 M, the G6 anchor bugs. The clustering bug (§1.3a) and the mirror-inertness (§1.3b) I verified directly in the source.

### 9.3 What I would do first if I had one week

1. Fix G1/G2 clustering; re-run both baselines; publish the *new* (higher) SE values.
2. Run the same baseline **5×** with different seeds. Publish `σ_seed`. **This single number determines whether the rest of the design is coherent.**
3. Stage a private tier on one real epoch and publish the realized mirror-gap distribution.
4. Only then decide the collapse rule — with a measured noise floor instead of an assumed one.

---

## Sources

**Bittensor / decentralized networks**
- Macrocosmos, *LLM pretraining: The Use-Case Blockchain Has Been Waiting For?* (SN9 whitepaper; `isWin` rule §2.3.3, WTA 95 %+ §3.3.2, ϵ=0.5 % §3.3.3, model hoarding §3.4.2, 7B★ ϵ=0.1 % 2024-08-08, rebasing + vanishing-gradient attacks §3.4.3) — https://www.macrocosmos.ai/research/pretraining_whitepaper.pdf
- *IOTA: A Technical Primer for Release* (arXiv 2507.17766, Jul 2025) — the two named failures of SN9's model competition (per-miner capital cost; WTA→hoarding); activation-work rewards; recomputation + cosine similarity; **unannounced monitoring**; CLASP explicitly **not shipped** — https://arxiv.org/abs/2507.17766
- Macrocosmos, *Subnet 9 — IOTA* docs — https://docs.macrocosmos.ai/subnets/subnet-9-iota
- Macrocosmos, *Monsters, vampires, and X-rays: subnet 9's Halloween deep dive* (~Oct 2024) — **measured** model-copying ("direct copying with practically zero amendments") and parameter-padding — https://macrocosmosai.substack.com/p/monsters-vampires-and-x-rays-subnet
- macrocosm-os/pretraining (validator/eval mechanics; `temperature = 0.01` → "~96 % to best model with only ~3 receiving any weights"; `alpha = 0.5`) — https://github.com/macrocosm-os/pretraining · constants: https://github.com/macrocosm-os/pretraining/blob/main/constants/__init__.py
- macrocosm-os/finetuning (SN37: `COMPETITION_SCHEDULE_BY_BLOCK`, block-scheduled sunsets, `ALPHA = 0.90`, `MIN_WEIGHT_THRESHOLD = 0.18`, `INVERSE_EXPONENTIAL` normalization, forced tokenizer, 120–150-row eval tasks) — https://github.com/macrocosm-os/finetuning
- taoverse (shared `iswin` / epsilon / competition library) — https://github.com/macrocosm-os/taoverse
- **Gradients / G.O.D (SN56, Rayon Labs)** — the live paired-comparison boss-round rule, dead zone, bootstrap LCB, two-bar emission, champion time decay, near-duplicate cap, obfuscation detection, LLM dedup — https://github.com/gradients-ai/G.O.D · tournament constants: https://github.com/gradients-ai/G.O.D/blob/main/validator/tournament/constants.py · subnet overview: https://bittensor.ai/subnets/56 · revision commit (2026-06-19, burn bar 0.05→0.10, champion decay 0.33→0.165 %/day, progressive dethrone thresholds disabled): https://github.com/gradients-ai/G.O.D/commit/c83bc13a6087121e176c600ee666e90d33706a71 · earlier progressive-threshold design: https://github.com/gradients-ai/G.O.D/commit/a0da6a2687d6bc5407ee9a0c23104bd4bf357827
- SubnetRadar, *Subnet 56* (external review; open-publication tradeoff) — https://subnetradar.com/research/subnets/56 **[3P]**
- Templar (SN3) — LossScore → OpenSkill rating, superlinear sybil deterrence, Proof-of-Computation anti-copy **[E/DOC]**
- Subnet Alpha, *IOTA* — https://subnetalpha.ai/subnet/iota/ · Taopedia, *Subnet 9* — https://taopedia.org/wiki/subnet_9/ — **[3P aggregators, several apparently LLM-generated; not treated as authoritative. The claim that SN9's 7B competition began at ε = 3 % appears only here, contradicts the whitepaper's 0.5 %, and I could not verify it — treated as unsupported.]**

**Numerai (noisy-signal payouts, originality → MMC → TC → MMC)**
- *A New Data Science Competition Where Being Different Pays* — https://blog.numer.ai/a-new-data-science-competition-where-being-different-pays/
- *MMC2 Announcement* — https://forum.numer.ai/t/mmc2-announcement/93
- *Leaderboard Bonus Exploit Uncovered* (Madmax/Madmin/The_Guy; >100 NMR, 85 % in <6 months) — https://forum.numer.ai/t/leaderboard-bonus-exploit-uncovered/200
- *MMC — Payout Details and Analysis* (bonus removed 2020-09-09) — https://forum.numer.ai/t/mmc-payout-details-and-analysis/220
- *True Contribution Details* (cvxpylayers gradient-of-portfolio) — https://forum.numer.ai/t/true-contribution-details/5128
- *Changing Scoring & Payouts Again To MMC Only* (TC retired: blackbox, optimizer-coupled) — https://forum.numer.ai/t/changing-scoring-payouts-again-to-mmc-only/6794
- *MMC staking starts Jan 2, 2024* — https://forum.numer.ai/t/mmc-staking-starts-jan-2-2024/6827
- Numerai docs — scoring / MMC / staking — https://docs.numer.ai/numerai-tournament/scoring · https://docs.numer.ai/numerai-tournament/scoring/meta-model-contribution-mmc
- *Payout Updates for the 2026 Season* (CORR 0.75, MMC 2.25, Ender20) — https://blog.numer.ai/numerai-monthly-numercon-speakers-new-dataset-target-2026-payout-updates/

**Leaderboards / competition design**
- Blum & Hardt, *The Ladder: A Reliable Leaderboard for ML Competitions* (ICML 2015) — http://proceedings.mlr.press/v37/blum15.pdf · https://arxiv.org/abs/1502.04585
- Roelofs et al., *A Meta-Analysis of Overfitting in Machine Learning* (NeurIPS 2019) — https://proceedings.neurips.cc/paper_files/paper/2019/file/ee39e503b6bedf0c98c388b7e8589aca-Paper.pdf
- Recht et al., *Measuring Generalization and Overfitting in ML* (thesis, Kaggle + ImageNet analyses) — https://www2.eecs.berkeley.edu/Pubs/TechRpts/2019/Archive/EECS-2019-102.pdf
- Dwork et al., *Preserving Statistical Validity in Adaptive Data Analysis* / reusable holdout — https://arxiv.org/abs/1411.2664
- *The Leaderboard Illusion* (2025) — https://arxiv.org/abs/2504.20879
- Chatbot Arena — https://arxiv.org/abs/2403.04132
- HELM — https://arxiv.org/abs/2211.09110 · *Efficient Benchmarking* (MWR "unreliable and gameable") — https://arxiv.org/abs/2308.11696
- *The Benchmark Lottery* — https://arxiv.org/abs/2107.07002
- BIG-bench — https://arxiv.org/abs/2206.04615
- GSM1k — https://arxiv.org/abs/2405.00332 · GSM-Symbolic — https://arxiv.org/abs/2410.05229
- Evan Miller, *Adding Error Bars to Evals* (clustered SEs, 0.3–0.7 item correlation, paired analysis) — https://arxiv.org/abs/2411.00640
- ARC Prize 2025 technical report — https://arxiv.org/abs/2601.10904 · rules — https://arcprize.org/competitions/2025
- MLCommons MLPerf Inference rules (closed-division accuracy gates) — https://github.com/mlcommons/inference_policies

**NAS: benchmarks, reproducibility, fidelity**
- Li & Talwalkar, *Random Search and Reproducibility for NAS* — https://arxiv.org/abs/1902.07638
- Yang, Esperança & Carlucci, *NAS evaluation is frustratingly hard* (ICLR 2020) — https://arxiv.org/abs/1912.12522
- Lindauer & Hutter, *Best Practices for Scientific Research on NAS* — https://arxiv.org/abs/1909.02453
- NAS-Bench-101 — https://arxiv.org/abs/1902.09635 · NAS-Bench-201 — https://arxiv.org/abs/2001.00326 · NATS-Bench — https://arxiv.org/abs/2009.00437
- Zela et al., *Understanding and Robustifying DARTS* — https://arxiv.org/abs/1909.09656
- Sciuto et al., *Evaluating the Search Phase of NAS* — https://arxiv.org/abs/1902.08142
- White et al., *How Powerful are Performance Predictors in NAS?* (NeurIPS 2021) — https://papers.neurips.cc/paper_files/paper/2021/file/ef575e8837d065a1683c022d2077d342-Paper.pdf
- NAS-Bench-Suite-Zero (zero-cost proxies) — https://arxiv.org/abs/2210.03230
- *Multi-fidelity NAS with Knowledge Distillation* (τ ≈ 0.42 at 1 epoch; 1/27–1/3 data τ tables) — https://arxiv.org/abs/2006.08341
- *Dynamic Ensemble of Low-Fidelity Experts* (τ 0.2549 → 0.7064; inappropriate low-fidelity info *damages* prediction) — https://ojs.aaai.org/index.php/AAAI/article/view/26339
- Tay et al., *Scaling Laws vs Model Architectures* (best model fluctuates with scale; upstream⊥downstream; ALBERT trends negative) — https://arxiv.org/abs/2207.10551 · https://aclanthology.org/2023.findings-emnlp.825.pdf

**Contest theory / tournaments under noise**
- Moldovanu & Sela, *The Optimal Allocation of Prizes in Contests* (AER 91(3):542–558, 2001) — https://doi.org/10.1257/aer.91.3.542 · working paper PDF: https://www.econ.uni-bonn.de/micro/en/moldovanu/publications-1/pearson22.pdf
- Drugov & Ryvkin, *Tournament rewards and heavy tails* (JET 190, 2020) — https://www.sciencedirect.com/science/article/abs/pii/S0022053120301095
- Drugov & Ryvkin, *How noise affects effort in tournaments* (JET 188, 2020) — https://ideas.repec.org/a/eee/jetheo/v188y2020ics0022053120300636.html
- Lazear & Rosen, *Rank-Order Tournaments as Optimum Labor Contracts* (JPE 89(5), 1981) — https://doi.org/10.1086/261010
- *Tournaments with a Standard* (IFR ⇒ WTA; DFR ⇒ equal prizes to qualifiers) — https://ideas.repec.org/p/arx/papers/2412.01139.html
- Fu & Lu, *The optimal multi-stage contest* (Economic Theory 51(2), 2012) — https://ideas.repec.org/a/spr/joecth/v51y2012i2p351-382.html
- *Screening in Multistage Contests* (M&SOM 2023) — https://doi.org/10.1287/msom.2021.0378
- Delfgaauw et al., *Prize Spread and Noise in Elimination Tournaments* (field experiment) — https://repub.eur.nl/pub/25711/2011-1201.pdf

**Research funding: concentration, review reliability, lotteries, preregistration**
- Fortin & Currie, *Big Science vs. Little Science* (PLOS ONE 2013) — https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0065263
- Mongeon et al., *Concentration of research funding leads to decreasing marginal returns* (Research Evaluation 2016) — https://doi.org/10.1093/reseval/rvw007
- Aagaard, Kladakis & Nielsen, *Concentration or dispersal of research funding?* (QSS 2020) — https://direct.mit.edu/qss/article/1/1/117/15557/
- Fang, Bowen & Casadevall, *NIH peer review percentile scores are poorly predictive of grant productivity* (eLife 2016; AUC 0.54, r²=0.0078) — https://elifesciences.org/articles/13323
- Danthi et al. (NHLBI, Circulation Research 2014) — https://doi.org/10.1161/circresaha.114.302656
- Volkswagen Foundation "Experiment!" partial randomization (two-cohort null result) — https://sfdora.org/2025/03/27/insights-on-partial-randomization-in-research-funding-learnings-from-the-volkswagen-foundation/
- HRC New Zealand Explorer Grants — https://casrai.org/guides/hrc-explorer-grant · Avin, *Mavericks and lotteries* — https://www.sciencedirect.com/science/article/pii/S0039368118300190
- Taxonomy of funding lotteries (Research Evaluation rvae025, 2024) — https://doi.org/10.1093/reseval/rvae025
- Scheel, Schijen & Lakens, *An Excess of Positive Results* (96.05 % vs 43.66 %) — AMPPS 4(2), 2021
- Dreber et al., *Using prediction markets to estimate the reproducibility of scientific research* (PNAS 2015) — https://www.pnas.org/doi/10.1073/pnas.1516179112

**Novelty, credit routing, and their failures**
- Lehman & Stanley, *Abandoning Objectives: Evolution through the Search for Novelty Alone* — https://doi.org/10.1162/EVCO_a_00025
- Cuccu & Gomez, *When Novelty Is Not Enough* (EvoStar 2011) — novelty search does not scale; "arbitrarily badly" construction — https://doi.org/10.1007/978-3-642-20525-5_24
- Quality-Diversity / MAP-Elites — https://arxiv.org/abs/1504.04909
- Sağlam et al., *Detecting Automatic Software Plagiarism via Token Sequence Normalization* (ICSE-SEET 2024) — JPlag 94.2 pp vs Dolos 12.4 pp separation; **MOSS below the unrelated-pair baseline** — https://dl.acm.org/doi/10.1145/3639474.3640084
- Devore-McDonald & Berger, *Mossad: Defeating Software Plagiarism Detection* (OOPSLA 2020) — https://doi.org/10.1145/3428206
- Yang et al., *ALERT: Naturalness-Aware Attack on Pre-trained Models of Code* (ICSE 2022) — 27.79 % attack success on CodeBERT clone detection — https://doi.org/10.1145/3510003.3510146
- *GraphCodeAttack* — 0.40 clone detection / 0.841 authorship attribution — https://arxiv.org/abs/2308.11161
- Taylor, *Digging for Golden Carrots: An Analysis of Research Tournaments* (AER 1995) — free entry not optimal — https://www.jstor.org/stable/2118189
- Fullerton & McAfee, *Auctioning Entry into Tournaments* (JPE 1999) — optimal n = 2 — https://doi.org/10.1086/250072
- Terwiesch & Xu, *Innovation Contests, Open Innovation, and Multiagent Problem Solving* (Management Science 2008) — why optimal-n=2 does **not** transfer when value comes from solution diversity — https://doi.org/10.1287/mnsc.1080.0884
- Shen & Barabási, *Collective credit allocation in science* (PNAS 2014) — https://doi.org/10.1073/pnas.1401992111
- Gensyn verification economics (refereed delegation 2–10× vs ~4 orders of magnitude for SNARKs) — https://www.gensyn.ai/ · Verde/refereed-delegation writeups
- Inference Labs, Bittensor commit-reveal cryptographic break (Aug 2025; fixed in Bittensor 9.9.0) — **[E]** reported via the companion research file
- Ghorbani & Zou, *Data Shapley* (2^(N−1) exact cost) — https://proceedings.mlr.press/v97/ghorbani19c/ghorbani19c.pdf
- tea.xyz farming: Sonatype — https://www.sonatype.com/blog/devs-flood-npm-with-10000-packages-to-reward-themselves-with-tea-tokens · Socket — https://socket.dev/blog/tea-xyz-spam-plagues-npm-and-rubygems-package-registries · AWS Inspector (>150,000 packages, Oct–Nov 2025) — https://aws.amazon.com/blogs/security/amazon-inspector-detects-over-150000-malicious-packages-linked-to-token-farming-campaign/ · Endor Labs — https://www.endorlabs.com/learn/the-great-indonesian-tea-theft-analyzing-a-npm-spam-campaign
- Optimism RetroPGF 3 retrospective — https://optimism.io/blog/retropgf-3-learnings-reflections · feedback thread — https://gov.optimism.io/t/retropgf-round-3-feedback-thread/6177
- Buterin, Hitzig & Weyl, *A Flexible Design for Funding Public Goods* (Management Science 2019) — https://doi.org/10.1287/mnsc.2019.3337 · pairwise coordination subsidies — https://ethresear.ch/t/pairwise-coordination-subsidies-a-new-quadratic-funding-design/5553
- thanks.dev / Canonical deployment — https://canonical.com/blog/canonical-thanks-dev-giving-back-to-open-source-developers
- Shen & Barabási, *Collective credit allocation in science* (PNAS 2014) — https://doi.org/10.1073/pnas.1401992111

**Evaluation pitfalls cited in support**
- Duan et al., *Do Membership Inference Attacks Work on LLMs?* — https://arxiv.org/abs/2402.07841
- Zheng et al., *LLMs Are Not Robust Multiple Choice Selectors* — https://arxiv.org/abs/2309.03882
- Schaeffer et al., *Are Emergent Abilities of LLMs a Mirage?* — https://arxiv.org/abs/2304.15004
- Choshen et al., *A Hitchhiker's Guide to Scaling Law Estimation* (restart variance ≤3.5 %) — https://arxiv.org/abs/2410.11840
- *PolyPythias* (50 pretraining runs, seed stability) — https://arxiv.org/abs/2503.09543
- Porian et al., *Resolving Discrepancies in Compute-Optimal Scaling* (warmup artifact) — https://arxiv.org/abs/2406.19146
- Cerebras-GPT (FLOP-matched small-scale benchmark baselines) — https://arxiv.org/abs/2304.03208

**Repository files read (read-only, unmodified)**
`AGENTS.md` · `docs/PRISM.md` · `docs/THREAT_MODEL.md` · `docs/AGENTS.md` · `docs/spikes/prism-v3/research/12-score-aggregation.md` · `docs/spikes/prism-v3/research/09-miner-metrics-leaderboards.md` · `crates/prism-registry/src/{competition.rs,hooks.rs,lib.rs}` · `crates/prism-pipeline/src/composite.rs` · `crates/prism-emit/src/lib.rs` · `crates/prism-recipe/src/lib.rs` · `crates/prism-recipe/harness/eval/{g2_downstream.py,rollup.py,common.py}`

**Additional sources for the NAS / competition-design evidence**
- RANK-NOSH (ICCV 2021) — per-epoch ρ vs fully-trained ranking — https://arxiv.org/abs/2108.08019
- TSE, *Speedy Performance Estimation for NAS* (NeurIPS 2021) — ρ = 0.851 at 5 % of budget, 0.95 at 25 % — https://openreview.net/forum?id=XvOH0v2hsph
- MPENAS (GECCO 2023) — cross-fidelity τ **restricted to the top-10 of 3,000** — https://doi.org/10.1145/3583131.3590513
- Sciuto et al., *Evaluating the Search Phase of NAS* — τ = −0.004 (RNN) / 0.195 (CNN); "NAS algorithms perform similarly to the random policy" — https://arxiv.org/abs/1902.08142
- Zhao et al., *Few-shot NAS* (ICML 2021) — τ = 0.013 for one supernet over 1,296 archs — https://arxiv.org/abs/2006.06863
- ASHA (MLSys 2020) — asynchronous successive halving; robustness to low-fidelity ranking noise — https://arxiv.org/abs/1810.05934
- EcoNAS (CVPR 2020) — shrink channels, keep data; 400× search-time reduction — https://arxiv.org/abs/2001.01233
- NAS-Bench-x11 / learning-curve extrapolation — https://arxiv.org/abs/2111.03602 · LC-PFN (NeurIPS 2023) — https://arxiv.org/abs/2310.20447
- Kielhöfer et al., *LCE Methods across Extrapolation Settings* — for binary "which is better", the **last observed anchor beats a parametric fit** — https://ada.liacs.nl/papers/KieEtAl24.pdf

**Companion research files in this directory**
- `research-novelty-bounties.md` — evidence review for §5–§6 (novelty measurement and its evasion, full Numerai originality→MMC→TC→MMC history, Shapley cost wall, funding concentration vs dispersal, lottery funding, registered reports, tea.xyz / RetroPGF / Gitcoin / thanks.dev lineage failures)
- `research-nas-competition.md` — evidence review for §2.3–§2.5 (NAS benchmarks and reproducibility, weight-sharing rank-correlation failures, zero-cost proxies, cross-fidelity correlation, Ladder/Thresholdout mechanics, Kaggle overfitting meta-analysis, best-arm identification, contest theory under noise)
- `research-bittensor.md` — evidence review for §2.1 (SN9 scoring pipeline and epsilon implementations, WTA parameters, documented failure modes, IOTA, SN37 competition framework, and comparable networks)

