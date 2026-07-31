# CbR Calibration History and Method Pivot

This file records why the first confusion-based implementation was retired and how the project progressed from fixed predictive auditing to CrowdSI-FM. Raw outputs remain under `log/` and are not committed.

## 1. Legacy hypothesis

The initial implementation used a frozen CrowdFM item posterior `q_k`, estimated a worker-specific class-conditional confusion matrix `P_i(a|c)` from nuisance tasks, and audited held-out pairwise disagreements. This implicitly assumed that a post-hoc deployment estimator could supply the response law missing from the discriminative CrowdFM checkpoint.

## 2. Fixed plug-in confusion sweep

At `alpha=0.05`:

| Configuration | Oracle q + oracle P | CrowdFM q + oracle P | Oracle q + estimated P | CrowdFM q + estimated P |
|---|---:|---:|---:|---:|
| small_binary | 0.00 | 0.00 | 0.00 | 0.00 |
| medium_multiclass | 0.15 | 0.10 | 1.00 | 1.00 |
| large_multiclass | 0.00 | 0.00 | 1.00 | 1.00 |
| sparse_imbalanced | 0.10 | 0.10 | 0.80 | 0.95 |

Aggregated over 80 worlds:

- oracle posterior + oracle confusion: `5/80 = 6.25%`;
- CrowdFM posterior + oracle confusion: `4/80 = 5.00%`;
- oracle posterior + estimated confusion: `56/80 = 70.00%`;
- CrowdFM posterior + estimated confusion: `59/80 = 73.75%`.

The residual/Monte Carlo core was approximately calibrated when the generating response law was supplied. The failure was the deployment confusion bridge.

## 3. Uncertainty corrections that failed

The following changes did not repair multiclass calibration:

1. parametric refit bootstrap;
2. Dirichlet posterior-predictive bootstrap;
3. symmetric prior-strength sweep over `0.1`, `0.01`, and `0.001`;
4. leave-one-worker-out global hierarchical shrinkage with strength `10`.

The full hierarchical sweep remained anti-conservative:

| Configuration | Oracle q + oracle P | CrowdFM q + oracle P | Oracle q + estimated P | CrowdFM q + estimated P |
|---|---:|---:|---:|---:|
| small_binary | 0.00 | 0.00 | 0.20 | 0.05 |
| medium_multiclass | 0.15 | 0.10 | 0.55 | 0.25 |
| large_multiclass | 0.00 | 0.00 | 0.75 | 0.65 |
| sparse_imbalanced | 0.10 | 0.10 | 0.55 | 0.20 |

Aggregated estimated-confusion rejection was `41/80 = 51.25%` with oracle `q` and `23/80 = 28.75%` with CrowdFM `q`. In the ten-class setting, entrywise confusion MAE was only about `0.023`, yet rejection remained `65--75%`; small coherent response-law bias is amplified by the spectral statistic.

## 4. Support sweep

A later sweep increased total tasks to raise mean worker/class nuisance support. The estimated disagreement became more accurate, but audit sample size increased at the same time.

| Setting | Mean support | Disagreement MAE | Estimated-P rejection |
|---|---:|---:|---:|
| K=5 | 5 | 0.0780 | 0.60 |
| K=5 | 20 | 0.0571 | 0.40 |
| K=5 | 50 | 0.0415 | 0.40 |
| K=10 | 5 | 0.0867 | 0.60 |
| K=10 | 20 | 0.0606 | 0.80 |

This experiment showed that the estimator error decreases, but it did not validate the method: nuisance estimation and audit power were scaled together. More importantly, even a consistent full-confusion estimator would remain an external model not produced by CrowdFM.

## 5. Conceptual diagnosis

The official CrowdFM checkpoint predicts task truth but does not expose

```math
Pr(A_{ik}=a | context, i, k).
```

The legacy implementation attempted to manufacture this quantity as

```math
\sum_c q_k(c)\widehat P_i(a|c).
```

That changes the object being audited. A rejection may indicate CrowdFM failure, confusion-family misspecification, estimator error, task-dependent worker behavior, or worker dependence. Consequently it is not a clean audit of the foundation model.

The conclusion is not “find a better prior.” It is:

> Do not infer a large post-hoc nuisance model and call it the predictive law of a frozen discriminative checkpoint.

## 6. Intermediate PredictiveCFM correction

The first principled correction trained a conditional emission head

```math
P_{ik}(a\mid c,G_C)
=
Pr(A_{ik}=a\mid Y_k=c,G_C,i,k),
```

and used the shared-truth predictive law

```math
Y_k\sim q_k,
\qquad
A_{ik}\mid Y_k\sim P_{ik}(\cdot\mid Y_k,G_C).
```

This removed deployment confusion estimation and fixed an important dependence error: held-out workers on the same task cannot be sampled independently from marginal response rows because they share one unknown truth.

`PredictiveCFM` remains a valid fixed-mechanism predictive-audit baseline, but it only detects incompatibility and defers. It does not identify a new mechanism or change aggregation constructively.

## 7. Current primary method: CrowdSI-FM

CrowdSI-FM introduces an explicit dataset-level mechanism posterior

```math
q_\phi(Z_{\mathcal D}\mid G),
```

and mechanism-conditioned truth, response, and assignment laws. At deployment it freezes network weights and updates only the low-dimensional mechanism posterior from held-out annotations.

Task-disjoint folds are used to avoid sharing latent truths across evidence folds. A posterior adapted on fold A is evaluated on fold B and vice versa. The averaged likelihood ratio is used as an e-value gate; adaptation is enabled only when the alternative mechanism posterior has sufficient independent predictive evidence over the fixed plug-in mechanism.

The project progression is therefore:

```text
post-hoc confusion audit
    -> learned fixed response-law audit
    -> amortized mechanism initialization + evidence-gated system identification.
```

See `docs/IDEA.md`, `docs/CROWDSI_SPEC.md`, and the `src/cfm/si/` modules.

## 8. Status of earlier files

The confusion-based files are retained only for reproducibility of the negative result:

- `src/cfm/audit/disagreement.py`;
- `src/cfm/audit/bootstrap.py`;
- `src/cfm/audit/pipeline.py`;
- `src/cfm/audit/calibration.py`;
- `run_cbr_calibration.py`;
- legacy calibration YAML files.

The `predictive_*` implementation and `PredictiveCFM` are retained as a fixed-mechanism ablation. New main experiments must use `CrowdSIFM`, `train_crowdsi.py`, and `evaluate_crowdsi.py`.

## 9. Next experiment

The next load-bearing step is local execution, not more confusion-prior tuning:

1. run the full unit suite and a two-step CrowdSI training smoke test;
2. establish in-prior truth/response/assignment performance;
3. validate e-value false-adaptation control under data generated from the fixed plug-in law;
4. hold out complete mechanism families and compositions;
5. compare zero-shot, always-adapt, and evidence-gated aggregation.
