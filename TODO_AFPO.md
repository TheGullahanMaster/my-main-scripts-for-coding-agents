# TODO_AFPO — stages, self-organising islands, and the rest of the roadmap for `afpo.py`

`TODO.md` was written against `evo68.py` (it names `benchmark_evo68.py`, HoF,
family scheduler). This file retargets that roadmap to `afpo.py`, adds optional
**stages**, and replaces hand-assigned island roles with **self-organising
specialisation** that works for any island count.

Line references are to `afpo.py` as of 2026-09-29.

## Status 2026-09-29

Implemented in `afpo.py` / `afpo_gui.py` (tests: `test_afpo_stages.py`, plus a
GUI form test in `test_afpo_gui.py`):

- **Phase 1 stages**: `stage_config`, `promote_stages`, `advance_topology`;
  terminal + GUI setup; checkpoints/resume; per-stage-level ring migration.
  Decisions taken: each cell is a full `IslandRuntime`; fitness thresholds are
  adaptive quantiles (default median) of the stage above, never loosened;
  population is split evenly across cells.
- **Phase 2 roles**: `update_roles` (responsibility-weighted lexicase case
  order via `case_weights`), `assign_role_parameters` (tree size / crossover /
  Bayesian-proposal spread), complementarity-based retirement, collapse
  splitting, `migrate_fragments`. Complementarity is measured on the validation
  split when there is one (a mild use of validation data inside the search).
- **Equivalence collapse** (Phase 5 item): an associative-commutative normal
  form key (`equivalence_key`), not an e-graph. Used for offspring redraws,
  `novelty_pool`, `unique_models` and `ParetoArchive`; `--equivalence-collapse
  off` restores syntactic identity; checkpoints from before it resume with it off.
- **Benchmark**: `benchmark_afpo.py` (Phase 6 harness).
- **Per-island roles** (2026-10-03): with roles on, every island after the
  first gets its own role, chosen in the terminal setup (comma list) or with
  one GUI dropdown per island; the setup key is `roles.assignments`. `auto` is
  the self-organising specialist above (the default, so older setups and
  checkpoints behave exactly as before); the fixed presets
  (`preset_role_parameters`) are `generalist`, `simplifier` (anchored, see
  below; hoist/shrink/new `prune` moves, accepts smaller trees with unchanged
  output, gathers every island's migrants), `explorer` (25%
  fresh random offspring, no semantic step cap, more crossover/proposals),
  `refiner` (constant/point moves, step cap 1 target s.d.), and
  `family:<group>` (new structure from arithmetic + one operator group; MDL
  still priced on the run's whole grammar). Fixed roles are never reweighted
  or retired; their held-out contribution is still logged. Only a first
  look so far (coffee cooling, Newton's law, 4 islands, population 120, 60
  generations, groups 1-9, 4 seeds): all-auto reached the noise floor 4/4
  (mean 136 MDL bits, 2 readable exponentials); auto + simplifier +
  family:3 reached it 4/4 but with rbf/sigmoid-tail forms (mean 183 bits);
  auto + auto + simplifier reached it 2/4 (mean 166 bits). So no preset is a
  default; benchmark them (`benchmark_afpo.py` takes `roles.assignments`)
  before tuning `SIMPLIFIER_PARSIMONY`, `EXPLORER_NOVELTY`,
  `REFINER_MAX_DELTA` or `ROLE_MUTATION_BIAS`. Stages with roles are covered
  by tests only.
- **Anchored simplifier** (2026-10-03, `anchored_band` / `anchored_survivors`
  / `anchored_emigrants`): the first simplifier (60% size cap, 5% parsimony
  near-tie band) only held migrant copies: at 50 generations on the coffee
  data its shortest model within 1% of the best loss was always another
  island's. Now it solves min MDL s.t. loss <= anchor + max(5% |anchor|,
  noise floor), anchor = the island's lowest loss: half the survivors are the
  shortest in-band models no larger than the anchor, 2/3 of parents come from
  them (their children are capped at the anchor's size), the other half of
  the island is ordinary Pareto survival with the normal size allowance, and
  emigrants are the shortest models within 1% (else 5%). Same diagnostic
  afterwards: 29-32 in-band models (other islands 0-7) and the shortest one
  within 1% is the simplifier's own child. `afpo_og.py` freezes the earlier
  version (`AFPO_MODULE=afpo_og python bench_roles.py ...`).

Not done: role-specific ALPS reseeding uses global tree size; residual-signature
descriptors (Phase 3) and everything in Phases 4-5 other than the above.

## Benchmark verdicts 2026-09-29 (20 seeds, piecewise / ratio_wrap / MyTempos)

`benchmark_afpo_20seed.json`; paired on case+seed against the current
defaults, bootstrap 95% intervals, per case (pooling hides MyTempos).

- Keep **scale-balanced selection** on: switching it off loses 0.18 R² on
  MyTempos [-0.31, -0.06]; no effect on the other two.
- Keep **equivalence collapse** on (-0.17 without it on MyTempos [-0.33, -0.01])
  and **quality/coverage QD parent choice** (-0.21 with legacy [-0.35, -0.07]).
- **Residual archive**: no clear difference anywhere (MyTempos -0.08 without
  it, interval spans 0). Kept on; unproven.
- **Stages, islands, roles**: nothing beats the defaults on any case.
  `stages_both` is worse on piecewise [-0.043, -0.002]; islands+stages+roles
  is worse on piecewise and MyTempos. All stay off by default.
- ratio_wrap is too noisy to rank anything (intervals ~ +/-0.3).
- **Numeric guard check** added afterwards (`--numeric-guard-check`, default
  on): models whose output depends on the +/-1e12 value clamp or on an
  operator's input clip (pow exponent, exp, sinh, cosh, tan, ...) are
  infeasible; the constant fitter never tunes into one. Found on capped data
  (min(x, 17) learned through the clamp). On 5 seeds x 9 cases it made no
  accuracy difference (+0.003 [-0.023, +0.041]); on a synthetic ammo cap it
  removed the exploits (2 of 5 runs -> 0 of 5).
- Found: catastrophic test failures (R² down to -10) from memorising
  constructs: `mod(x + 1e5..1e6, c)` sawtooths and `pow` with huge constants
  fit training rows but not the rows between them. Two MyTempos runs
  selected a constant model.
- **Constant selection** (fixed 2026-09-29): not a selection bug. MyTempos
  keeps 5 validation rows; in both runs they were mostly "wrapped" rows the
  search never modelled, so every input-using candidate lost to a constant on
  validation and choosing the constant was correct on the evidence. Now the
  run prints a warning explaining this, the selection record/model card carries
  it, and the best training fit is offered second, labelled "Lowest Training
  Loss (not supported by validation)".
- **Interpolation check** (`--interpolation-check`, default on) against
  memorisation: probes at random fixed fractions between nearest-neighbour
  rows; charges the median excess of the between-row miss over the at-row
  miss against the neighbours' target band. Random fractions are essential
  (a sawtooth whose period nearly divides the spacing is aliased at exact
  midpoints); the median keeps steep-but-imperfect jumps unpenalised (a mean
  version made piecewise measurably worse, -0.028 [-0.074, -0.001]). 20 seeds:
  MyTempos +0.17 [+0.08, +0.28]; ratio_wrap +0.12 [-0.10, +0.37] with runs
  below 0 going 2 -> 1; piecewise and the six easy cases unchanged. Remaining
  gap: a sawtooth whose period itself nearly matches the row spacing can
  still slip through (1 of 20 ratio_wrap runs).
- **Noise-floor ties** (fixed 2026-09-29, from a user run on weight_gradient =
  forward_input * upstream_delta): the exact 51-bit model never reached the
  Pareto archive (only the NSGA elite was offered; the best-so-far tracker also
  sees offspring), and "Best Score" recommended a 160-bit copy bloated with
  log10(...) factors because its validation loss was 1.5% lower -- at 4.7e-11,
  i.e. CSV rounding noise. Now: comparisons use max(relative tolerance,
  LOSS_NOISE_FLOOR = 1e-9); the archive is offered the best-so-far model each
  generation and applies the parsimony near-tie filter. Replaying that run's
  checkpoint recommends the exact product.
- The numeric guard check does not reduce memorisation (ratio_wrap runs
  below 0: 2 with and 2 without it, 20 seeds).

## Harsh gradients 2026-10-01 (jump scan, jump mutation, between-row selection)

Discontinuous targets (mod, stairs, if/else, integer rules) failed because a
constant that only places a jump has a zero finite-difference gradient almost
everywhere, so Levenberg-Marquardt never moved it.  Added (all default on;
checkpoints from before them resume with them off):

- `--jump-constant-scan`: before the gradient fit, constants under a jump
  operator (mod/floordiv/quantize, comparisons, floor/round/..., if_else
  condition) are placed by a batched candidate scan (data midpoints for
  thresholds, range fractions for periods, a log grid, simple numbers), then
  two shrinking local grids.
- `--jump-mutation-weight` (1): wraps a subtree in mod(s,c), floordiv(s,c)
  or if_else(gt(x,c),s,s') as one move; the portfolio adapts its weight.
- `--selection-probe-filter`: the final choice drops candidates whose
  predictions between nearest-neighbour rows (training + validation) leave
  the neighbours' band far more often than the best candidate's.  Integer
  valued inputs are never probed (parity/Collatz have no in-between).

Results, 10 seeds, solved = held-out R^2 >= .99 (bench_harsh.py) or .999
(bench_jump.py): bench_harsh 62 -> 89/100, bench_jump 28 -> 58/80; smooth
benchmark_afpo cases unchanged (5 seeds); bench_complex.py (3-8 inputs,
reused variables) 2/40 exact recoveries either way: still open.  Run time on
bench_jump about equal (25.7 s -> 24.6 s mean) after batching the scan.

Notes: `afpo_lib/fitcore.pyx` (the Cython fitter) was not in the repository
then (added 2026-10-03), so every run here used the Python fitter; the scan runs before either
fitter.  bench_harsh's kink_ifelse jumps at 0.65 but keeps the test row that
straddles it, so some of its "failures" are that one ambiguous row.

## Smooth activations and the loss noise floor 2026-10-01

SiLU came out exact in 6/10 seeds and GELU in 0/10. The search stopped at
x*sigmoid(1.6x) (R^2 0.99996), because a plain point swap to erf gives
x*erf(kx) (R^2 0.23: erf is centred on 0, sigmoid on 0.5).  Added (default
weight 1; switched off when the grammar lacks their operators; checkpoints
from before them resume with them off):

- `--squash-swap-weight`: sigmoid/tanh/erf swapped for one another at the
  same level, range and slope (sigmoid(z) -> 0.5+0.5*erf(0.443z)).
- `--smooth-swap-weight`: relu <-> softplus(4z)/4 or z*sigmoid(4z),
  abs -> z*tanh(4z), sign -> tanh(4z).
- `--gate-mutation-weight`: s -> s*sigmoid(c*u), s*(1+erf(c*u)) or
  s*(1+tanh(c*u)), with u = s or an input.
- `--loss-noise-floor auto`: the tie band (formerly a fixed 1e-9) is
  1000x the loss of rounding the targets to their written digits, within
  [1e-18, 1e-9].  With float64 targets the old band let a 1e-11
  approximation tie a 1e-20 model, so the shorter approximation won the
  final choice and the best-so-far archive never replaced it.

bench_activation.py, 10 seeds, solved = test R^2 >= 0.99999 and R^2 >= 0.9999
for |x| in [4.2, 8]: with / without the moves, SiLU 10 / 8, GELU-erf
10 / 0, GELU-tanh 10 / 1 (found as the erf form, R^2 1-3e-8 apart), Mish 6 / 3,
soft switch 1 / 0 (of 9).  Regressions vs main: bench_harsh 89 = 89/100
and bench_jump (jump_scan_mutation) 58 = 58/80, identical because those
grammars have no squash operators. Smooth benchmark_afpo (5 seeds) 34 vs 33/40.

## What already exists (do not rebuild)

- Islands: `IslandRuntime` (3966), one full search state per island (population,
  Pareto archive, semantic + structural QD archives, Bayesian banks, fragment
  library, ADF registry, evaluation budget, dynamic pressure). Ring migration of
  local Pareto elites (`migrate_islands`, 4034). Per-island checkpointing
  (`island_snapshot` / `island_from_snapshot`, 3985-4030). Setup asks island
  count / migration interval / migrants (4491-4496); GUI mirrors it
  (`afpo_gui.py:279-301`). Minimum 8 models per island.
- AFPO age: `model.age = generation - birth_generation` (4152); age is an NSGA
  objective (`objective_schema ... mdl_bits,age`).
- Parent selection: epsilon-lexicase with informed down-sampling (3015-3033);
  QD parents blended in (`blend_dual_qd_parents`).
- `FragmentLibrary` (1709), `MutationPortfolio` adaptive operator weights
  (1676), `CasePopulation` difficulty/coverage row weights (1859),
  `DynamicPressureController` (2572), novelty injection of random trees.
- One generation = `evolve_generation(pop, generation, ...)` (4168); the main
  loop calls it once per island (≈4636) then migrates.

---

## Phase 1 — Optional AFPO stages ("vertical islands")

Goal: newcomers compete only with peers; a model meets the strongest models
only after climbing every stage. Modes: `off` (default, current behaviour),
`fitness` (HFC-style), `age` (ALPS-style), `both`.

### Design

- **Cell = (island, stage).** Each cell is its own `IslandRuntime`. Reason:
  archives, QD cells, fragment library and Bayesian banks are all parent
  sources; if a stage shared them with the top stage, top-stage models would
  leak back down as parents and destroy the fairness the stage exists to give.
  Cost: memory and per-cell overhead scale with islands × stages.
- **Population split:** `population` divided across islands × stages, still
  ≥ 8 models per cell (extend the existing check at 4492 / 4533 and the GUI).
  Optional later: larger share for the top stage.
- **Stage 0 is the only stage that creates brand-new random models** (beyond
  the existing novelty injection, which stays per cell). Upper stages are fed
  only by promotion and their own offspring.
- **Promotion ("vertical migration")** runs after every island has done its
  generation, next to the existing ring migration:
  - `fitness`: stage *k* has an admission threshold on true, unweighted
    training loss (`aggregate_loss`). A model in stage *k* that beats the
    threshold of stage *k+1* is copied up (NSGA selection at the receiver keeps
    its size fixed, as `migrate_islands` does). Thresholds adapt: after a warm-up,
    set stage *k*'s threshold to a quantile of the stage-above's loss
    distribution (e.g. the stage-above's median), recomputed every N
    generations, monotone non-increasing so a stage cannot fill with
    regressions. Top stage has no upper threshold.
  - `age`: stage *k* has an age limit (ALPS schedule: `age_gap * k^2`, or
    linear / Fibonacci as a choice). A model older than its stage's limit is
    offered to stage *k+1* (admitted only if it beats that stage's worst on
    NSGA rank) and removed from stage *k* either way. Stage 0 is fully
    re-seeded with random models every `age_gap` generations. Top stage has no
    limit.
  - `both`: age limits force models out; fitness thresholds decide whether they
    are admitted above. A model that is too old and not good enough dies.
- **AFPO inside each cell is unchanged** (age stays an objective). Note in docs
  that `age` stages layer a hard age structure on top of AFPO's soft one; the
  benchmark must show whether that helps or just duplicates it.
- **Islands × stages:** ring migration runs *per stage level* (stage *k* of
  island *i* → stage *k* of island *i+1*), so horizontal exchange never skips
  the ladder. Vertical promotion stays within an island. With 1 island this is
  pure stages; with 1 stage it is exactly today's islands.
- **Final model choice:** pool all cells, as the island pool does now (≈4660).

### Config, persistence, UI

- [x] Setup questions after island count: stage mode (`off/fitness/age/both`),
  stage count, `age_gap` + schedule (age/both), threshold quantile + refresh
  interval (fitness/both). Mirror in `afpo_gui.py` form and `check_form`.
- [x] `island_config` gains `stages: {mode, count, age_gap, schedule,
  threshold_quantile, refresh, thresholds, promotion_events}`. Old checkpoints
  without it load as `mode=off, count=1` (test this explicitly).
- [x] `island_states` stores cells with `(island, stage)` indices; resume
  rebuilds the grid; mismatch raises like the existing count check (4360).
- [x] Model card: stage config, thresholds over time, promotions per stage.
- [x] Telemetry: per-cell best loss, promotions up each rung, stage-0 reseeds;
  the GUI `Telemetry` class keys islands by archive id (`afpo_gui.py:339`) and
  must learn `(island, stage)`.

### Verification

- [x] Unit tests: threshold monotonicity; age-limit eviction and admission;
  stage 0 reseed timing; `off` mode leaves the population untouched;
  checkpoint round-trip of a 2×2 grid plus a staged 2-island run resumed from
  its checkpoint; old-checkpoint load. (Not done: a bit-identical seeded
  comparison against the pre-stage code, since equivalence collapse, on by
  default, deliberately changes seeded runs.)
- [x] Twin check: every place that iterates `islands` (final pooling,
  snapshots, ADF refresh, progress hook, migration) handles cells.
- [ ] Benchmark (see Phase 6) `off` vs `fitness` vs `age` vs `both`, 3+ seeds,
  equal evaluation budget (stages must not win by spending more evaluations).

---

## Phase 2 — Self-organising island specialisation (replaces hand-made roles)

Problem with `TODO.md` item 3: named roles ("tails", "rank/order", …) need prior
knowledge of the equation, and a fixed role list does not fit a variable island
count. Instead, let islands **discover** their niche from data, for any count.
All of this is selection-only: reported loss, archive admission, final pick and
validation stay on the true unweighted objective.

### 2a. Row responsibility (emergent row specialisation) — main mechanism

Mixture-of-experts / competitive-learning style. Every R generations:

1. Take each island's current best model; compute per-row error `e[i, r]`.
2. Responsibility `resp[i, r] = softmax_i(-e[i, r] / T)` (soft assignment of each
   row to the islands that currently handle it best). Temperature `T` from the
   median error so it is scale-free.
3. Island *i* selects parents (lexicase case sampling and `CasePopulation`
   weights) with row weights `mix * resp[i, r] + (1 - mix) * uniform`.

Positive feedback makes islands drift towards the rows they are already
relatively good at: specialisation emerges without naming it, and N islands
give up to N niches. Guards:
- [x] Island 0 is always a generalist (uniform weights).
- [x] `mix` capped (e.g. ≤ 0.7) so nobody forgets the full data.
- [x] Collapse detection: if two islands' responsibility vectors correlate
  above a threshold, re-randomise one's weights.
- [x] Symmetry breaking at start: random initial responsibilities.

### 2b. Parameter spread by island index (cheap diversity, any count)

Deterministically spread settings across islands using index / count:
parsimony tolerance, novelty rate, crossover rate, max nodes, operator subset
(each non-generalist island drops a random subset of operator groups). Log-spaced
so 2 islands and 20 islands both cover the range. Existing
`MutationPortfolio` / `DynamicPressureController` still adapt within each.

### 2c. Contribution test and retirement (replaces "role retirement")

- [x] Every W generations, test whether a specialist island contributed: its
  best model, combined with the global best by the existing affine /
  piecewise readout machinery (or a residual fit), improves **held-out**
  loss; or its fragments were admitted and used elsewhere.
- [x] No contribution for K windows → reset that island's responsibilities to
  random and re-draw its 2b parameters. Log it.

### 2d. Fragment migration

- [x] Extend `migrate_islands` to also send top fragments from each island's
  `FragmentLibrary` (bounded count, with provenance: source island, generation,
  admission evidence). Receiver admits only through its normal admission test.

### Verification

- [x] Unit tests: responsibilities sum to 1 per row; generalist stays uniform;
  weights never reach reported loss / HoF / final pick (assert on a run with and
  without 2a that reported losses for the same model are identical).
- [ ] Ablation: islands with 2a/2b/2c on vs off, same budget, 3+ seeds.

---

## Phase 3 — `TODO.md` items 1–2, retargeted to afpo

afpo already has semantic + structural CVT QD archives and a fragment library,
so these are **audit-then-extend**, not new builds.

- [x] Audit `FragmentLibrary.observe` admission against TODO item 2 criteria.
  Found and fixed (2026-09-29): (1) contribution was measured against
  predicting zero, so any fragment, even a constant one, was credited for
  shifting the mean; (2) contribution was scored on the rows it was fitted
  to, so an unrelated `sin(x1)` scored 36x the admission bar on chance
  correlation; (3) support counted models, so clones of one ancestor looked
  like independent evidence; (4) affine rescalings (`c*f`, `f+c`) were stored
  as separate fragments; (5) contribution was a never-decaying maximum. Now:
  cross-fitted held-out reduction beyond the best constant, constant-valued
  fragments rejected, support counts disjoint founder lineages, fragments keyed
  up to affine rescaling + algebraic equivalence (`fragment_key`), evolved
  contributions decay 5% per observation, bar raised to 5e-3; migrated
  fragments get a 10-observation probation.
- [x] Residual-signature QD archive (`ResidualQualityDiversityArchive`, a third
  repertoire): descriptor = share of error per target-magnitude bin (3 per
  numeric output) and per input region (4 quartiles of the first principal
  component); 64 cells. `--residual-archive on|off` (default on).
- [x] Scale-balanced selection-only lexicase (`--scale-balanced-selection
  on|off`, default off): equal total weight per target-magnitude quantile bin
  and asinh-compressed error comparison. Implemented in lexicase (the parent
  selector every run uses) rather than `CasePopulation`, which only acts on
  co-evolved minibatches above 512 rows.
- [x] Quality-and-coverage-aware archive parent choice: success rate x bounded
  quality rank (best cell at most e^2 the worst's weight, ties share a rank) x
  coverage bonus for rarely tried cells; the protected uniform share is kept.
  `--qd-parent-choice legacy` restores success-only weighting.

## Phase 4 — Later (after Phases 1–3 produce telemetry)

- [ ] Small-data structural stability: bootstrap / leave-one-out re-fits of the
  top frontier; report threshold intervals and regime support. The model card
  already has a `bootstrap` slot (3108-3118) that is currently unused.
- [ ] Learned mutation proposer: extend `MutationPortfolio` from a global
  win-rate to a contextual bandit keyed on parent structure + residual
  signature. Advisory only, with reserved exploration.

## Phase 5 — Features from earlier discussion and recent literature

Ordered by expected value / cost. Each is optional and independent.

- [ ] Dimensional analysis (units per column; penalise unit-inconsistent
  subtrees). Biggest search-space pruning for scientific data; PySR has it.
- [ ] User row weights and a pluggable loss (loss currently fixed to Huber via
  `robust_loss`).
- [ ] Sympy / LaTeX export of the chosen model.
- [x] Equivalence-aware deduplication: canonicalise with e-graphs or
  memoisation + locality-sensitive hashing so algebraically equal trees count
  once (EGG-SR, arXiv 2511.05849; Burlacu-style memoisation/LSH,
  arXiv 2512.01682). afpo currently dedups by `semantic_key` on predictions.
- [ ] Loop / accumulator operator (`loopsum`, `loopprod`, fold with carried
  state) for factorials, variable-depth receptive fields, Euler integration —
  see the design discussed in the 2026-09-29 session.
- [ ] Optional LLM-proposed skeletons (LLM-SR, ICLR 2025) as another injection
  source next to Bayesian proposals. Needs an API dependency — only with
  explicit approval.

## Phase 6 — Non-negotiable verification (all phases)

- [x] **An afpo benchmark harness does not exist yet** (`benchmark_evo68.py`
  drives evo68). Build `benchmark_afpo.py` first: deterministic cases for smooth
  polynomial/rational, interaction, piecewise, ratio-wrap (`MyTempos_v2.csv`),
  small-data, multi-order-of-magnitude targets; fixed seeds; equal evaluation
  budgets; records time-to-first-strong-model, loss/R² trajectory, final
  loss/complexity, wall-clock.
- [ ] 3+ seeds per case; compare against the previous baseline and an ablation
  with the new mechanism off. No material regression on simple targets.
- [ ] Every new memory/archive/threshold state is bounded, serialisable,
  resettable per run, visible in telemetry, and covered by checkpoint
  round-trip tests.
- [ ] Selection-only helpers never change reported loss, archive admission,
  final pick, or validation reporting.
- [ ] Exported standalone model code matches scored semantics for any new op.

## Open decisions

1. Stage cells as full `IslandRuntime`s (isolated, heavier — recommended) vs
   population slices sharing one runtime (lighter, but archives leak).
2. Fitness-stage thresholds: adaptive quantiles (recommended) vs fixed
   user-given loss levels.
3. Whether the top stage gets a larger share of the population.
