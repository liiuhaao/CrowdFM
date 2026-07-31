# PredictiveCFM Baseline: Fixed-Mechanism Residual Audit

> **Status:** this document describes the intermediate fixed-mechanism baseline. The primary method is now CrowdSI-FM in `IDEA.md` and `CROWDSI_SPEC.md`.

## Research question

Given a crowd foundation model pretrained on synthetic annotation worlds and a new deployment annotation graph without gold labels, can cross-fitted held-out annotations determine whether the model's conditional annotator-emission law remains compatible with the deployment process, and can rejection-based defer improve selective aggregation?

## Baseline idea

The official CrowdFM backbone predicts the task posterior

```math
q_k(c)=\Pr(Y_k=c\mid G_C),
```

but does not expose a predictive law for held-out worker responses. PredictiveCFM adds an edge-conditioned emission head

```math
P_{ik}(a\mid c,G_C)
=\Pr(A_{ik}=a\mid Y_k=c,G_C,i,k).
```

At deployment, the audit hides labels from the model. For each audit task it treats all held-out workers as sharing the same unknown truth `Y_k`, computes item-conditioned marginal and pairwise-disagreement residuals, and calibrates them by Monte Carlo simulation that first samples one truth per task and then samples worker responses conditional on that truth.

The two checks are:

1. a direction-sensitive marginal categorical residual;
2. a signed worker-pair disagreement matrix summarized by its two-sided operator norm.

Separate p-values are combined by Bonferroni. Rejection triggers defer.

## Why it is no longer the main method

PredictiveCFM corrected the invalid post-hoc confusion bridge, but it still exposes one fixed mechanism learned at pretraining. When deployment differs, it can reject or defer but cannot identify a replacement mechanism and improve aggregation constructively.

CrowdSI-FM retains the conditional emission model but adds:

- an explicit dataset-level mechanism posterior;
- compositional mechanism primitives;
- assignment-process modeling;
- latent-only test-time Bayesian adaptation;
- task-disjoint e-value gating;
- adapted final truth aggregation.

## Historical predecessor

The still earlier implementation estimated full worker confusion matrices from sparse deployment labels and combined them with frozen CrowdFM task posteriors. Multiclass null rejection remained strongly inflated after refit bootstrap, posterior-predictive sampling, hierarchical shrinkage, and support diagnostics. The full progression is recorded in `CALIBRATION_RESULTS.md`.

## Interpretation

- PredictiveFM rejection means the fixed learned annotation law cannot reproduce the checked deployment responses.
- Non-rejection means compatibility with that fixed law, not proof that task labels are correct.
- It remains a useful ablation for determining whether explicit mechanism inference and adaptation add value beyond a learned response decoder.
