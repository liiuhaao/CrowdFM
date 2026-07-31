# Experiment Plan: CrowdSI-FM

The primary claim is no longer that a fixed response model can audit itself. CrowdSI-FM must show that a pretrained crowd foundation model can identify and safely adapt to a new dataset-level crowd mechanism without deployment gold labels.

## Paper claims

| ID | Claim | Required evidence |
|---|---|---|
| C1 | A separately exchangeable foundation architecture can infer a transferable dataset-level crowd mechanism. | Mechanism-family/parameter recovery, view consistency, posterior contraction, and held-out-world likelihood. |
| C2 | Compositional mechanism primitives generalize to unseen combinations better than a fixed CrowdFM or categorical family classifier. | Train-family/composition holdouts and interpolation/extrapolation results. |
| C3 | Test-time latent adaptation improves aggregation under mechanism shift without updating network weights. | Base vs always-adapt vs oracle-latent vs CrowdSI adaptation accuracy and response likelihood. |
| C4 | Task-disjoint e-value gating controls false adaptation under the fixed plug-in null. | False-adaptation curves at multiple alpha values, e-value calibration, and exact-null simulations. |
| C5 | Evidence gating preserves in-prior performance while recovering OOD performance. | Accuracy-gain/false-adaptation tradeoffs, real benchmark results, and adaptation-data scaling. |
| C6 | Modeling assignment and response generation jointly matters when observation is non-random. | Assignment-bias holdouts and ablations without the mask head. |

## Phase 0 — Local implementation validation

Completed in code:

- [x] `CrowdSIFM` with Gaussian mechanism posterior;
- [x] compositional mechanism basis;
- [x] mechanism-conditioned task, emission, and assignment heads;
- [x] shared-truth task-joint response likelihood;
- [x] task-disjoint cross-fitting;
- [x] latent-only variational adaptation;
- [x] cross-fitted e-value gate;
- [x] mechanism-diverse simulator;
- [x] training/evaluation entrypoints and tests.

Still required locally:

```bash
git pull --ff-only origin agent/cbr-audit-core
pytest -q
python train_crowdsi.py \
  config=config/crowdsi_train.yaml \
  epochs=2 \
  gradient_accumulation_steps=1 \
  output_dir=log/crowdsi_smoke
```

The complete suite is expected to contain the previous 21 tests plus 7 CrowdSI tests. Do not claim `28 passed` until it is executed locally.

## Phase 1 — Mechanism-model validation

Train on the full synthetic family vocabulary first, before mechanism holdouts.

Metrics:

- task-truth accuracy;
- masked response joint NLL per task;
- annotation Brier score;
- assignment AUROC/AUPRC;
- Gaussian posterior KL and entropy;
- primitive utilization and collapse diagnostics;
- mechanism-view symmetric KL;
- simulator mechanism-family linear-probe accuracy;
- continuous mechanism-parameter R2 from frozen mechanism latents.

Baselines:

1. original CrowdFM;
2. PredictiveCFM without a global mechanism latent;
3. CrowdSI with a single deterministic mechanism token;
4. CrowdSI Gaussian latent without compositional primitives;
5. CrowdSI without assignment modeling;
6. full CrowdSI-FM.

Decision rule: do not run adaptation claims until the full model matches or exceeds CrowdFM truth accuracy in-prior and predicts held-out response/assignment data better than fixed-mechanism baselines.

## Phase 2 — Exact-null e-value validation

Generate data directly from a frozen trained CrowdSI model at the plug-in mechanism mean. This isolates the mathematical gate from simulator misspecification.

For `alpha in {0.01,0.05,0.10}` report:

- empirical `Pr(E >= 1/alpha)`;
- confidence intervals over at least 1000 worlds;
- distributions of `log E_A->B`, `log E_B->A`, and combined `log E`;
- sensitivity to fold balance and predictive Monte Carlo sample count;
- leakage control using edge rather than task splitting.

The primary theorem applies to this fixed plug-in null. Simulator in-prior experiments are a model-calibration question, not an exact test of the theorem.

## Phase 3 — Held-out mechanism families

Train without one entire family, then test it:

- class-conditioned bias;
- coalition/shared response shocks;
- non-random assignment;
- strong task-difficulty shift;
- temporal worker drift;
- adversarial target-class behavior.

Compare:

- CrowdFM zero-shot;
- PredictiveCFM fixed mechanism;
- CrowdSI no adaptation;
- CrowdSI always adapt;
- CrowdSI e-value gated;
- per-dataset Dawid--Skene/GLAD/MACE;
- DGN and DGN-Adapt;
- oracle simulator mechanism and oracle gate.

Report:

- truth accuracy before/after adaptation;
- response joint NLL;
- e-value detection power;
- false-adaptation rate;
- posterior distance to simulator mechanism target;
- adaptation gain versus number of held-out tasks.

## Phase 4 — Compositional generalization

This is the most important novelty experiment. Train on primitive mechanisms individually but hold out their combinations.

Examples:

- class bias + assignment bias;
- coalition + difficulty shift;
- class bias + coalition;
- all three combined.

Compare the compositional basis against:

- one categorical mechanism embedding;
- an unconstrained global MLP token;
- a mixture-of-experts router with discrete top-1 selection;
- no mechanism latent.

Measure:

- aggregation accuracy;
- response likelihood;
- primitive weights versus known composition coefficients;
- interpolation to unseen severity mixtures;
- extrapolation beyond training severity ranges.

A main-conference claim requires improvement on held-out combinations, not merely recovery of training family IDs.

## Phase 5 — Non-ignorable assignment

Use worlds where assignment depends on worker ability, task difficulty, worker group, or target class. Preserve the same response mechanism while varying only assignment.

Ablations:

- ignore `O_ik`;
- treat missingness as random;
- assignment head trained but not conditioned on mechanism;
- full mechanism-conditioned assignment head.

Metrics:

- assignment likelihood/AUROC;
- task accuracy under assignment shift;
- mechanism posterior error;
- adaptation gain;
- e-value response when only the assignment process changes.

The current e-value implementation uses annotation responses only. A second assignment e-value should be added after the response-only core is validated.

## Phase 6 — Real-mask semi-synthetic evaluation

Take worker--task masks from real datasets and generate labels from known mechanisms. This separates unrealistic synthetic density from mechanism inference.

Required masks include at least six datasets spanning sparse/dense and binary/multiclass regimes. For each mask, evaluate in-prior, held-out-family, and held-out-composition worlds.

## Phase 7 — Real benchmarks

Use all available CrowdFM datasets. Adaptation receives no task truths. Gold labels are used only for retrospective evaluation.

Report:

- original CrowdFM accuracy;
- CrowdSI zero-shot accuracy;
- always-adapt accuracy;
- evidence-gated accuracy;
- per-dataset e-value and adaptation decision;
- masked response NLL;
- assignment prediction quality where the full matrix dimensions are known;
- runtime and adaptation steps.

Interpretation must remain conservative: high e-value means an alternative mechanism posterior predicts task-disjoint held-out annotations substantially better than the plug-in base law. It does not identify the unique real mechanism.

## Phase 8 — Theory experiments

Empirically test the theorem-facing quantities:

1. false adaptation under the exact plug-in null;
2. linear growth of `log E` with held-out task count under a fixed KL-separated alternative;
3. posterior contraction with deployment task count;
4. effect of generalized-Bayes temperature under misspecification;
5. primitive identifiability up to permutation and redundant bases.

## Required ablations

- no mechanism latent;
- deterministic vs Gaussian mechanism;
- categorical family embedding vs compositional basis;
- no mechanism consistency;
- no assignment head;
- edge-disjoint vs task-disjoint folds;
- point alternative vs posterior mixture alternative;
- always adapt vs e-value gate;
- backbone frozen vs joint pretraining;
- latent dimension and primitive count;
- adaptation KL weight, temperature, and task count.

## Reproducibility

Training seeds: `42,43,44`.

Evaluation/fold seeds: `42,43,44,45,46`.

Every result must record:

- Git commit and checkpoint;
- simulator family and continuous parameters;
- full configuration;
- base/adapt fold assignments;
- base and adapted mechanism posterior;
- directional and combined e-values;
- aggregation and predictive metrics;
- runtime and memory.

## Historical negative result

The confusion-estimator CbR experiments remain useful motivation: a deployment-fitted full confusion bridge was badly anti-conservative in sparse multiclass settings. PredictiveCFM corrected the missing generative output but remained a fixed-mechanism audit. CrowdSI-FM is the current primary direction because it changes aggregation through explicit deployment system identification rather than only detecting mismatch.
