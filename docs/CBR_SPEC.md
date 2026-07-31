# Predictive Audit Implementation Specification

## 1. Model contract

The primary audit does not estimate a deployment confusion matrix. A compatible model must accept

```python
output = model(
    context_data,
    query_workers=query_workers,
    query_tasks=query_tasks,
)
```

and return

```python
output["hat_task_option"]               # [num_tasks, num_options]
output["hat_annotation_given_truth"]   # [num_query_edges, num_options, num_options]
```

The first tensor parameterizes

```math
q_k(c)=\Pr(Y_k=c\mid G_C).
```

The second parameterizes the edge-conditioned emission law

```math
P_{ik}(a\mid c,G_C)
=\Pr(A_{ik}=a\mid Y_k=c,G_C,i,k),
```

where the second axis is candidate truth `c` and the final axis is reported option `a`. The final axis is normalized by softmax. `PredictiveCFM` uses a scorer shared across truth/report option embeddings, preserving variable `K`.

The conditional annotation head must be trained by masked-annotation prediction. `run_predictive_audit` refuses a `PredictiveCFM` checkpoint whose trained-head marker is false.

## 2. Context/audit split

`make_annotation_audit_split` partitions every observed edge into exactly one of:

- context: visible to the model;
- audit: hidden from the model and used only after `q_k` and `P_ik` are fixed.

Each audited task retains at least:

- `min_context_workers` visible annotations;
- `min_audit_workers >= 2` hidden annotations.

Each worker remaining in the audit set has at least `min_worker_context_edges` visible annotations elsewhere. If support repair leaves a task with fewer than two audit labels, that task is returned entirely to context.

## 3. Training objective

For synthetic query edge `(i,k)` with known truth `Y_k`, train the corresponding emission row:

```math
\mathcal L_{emit}
=-\sum_{(i,k)\in H}\log P_{ik}(A_{ik}\mid Y_k,G_C).
```

When a task has no gold truth, train its marginal held-out response likelihood:

```math
r_{ik}(a)=\sum_c q_k(c)P_{ik}(a\mid c,G_C),
```

```math
\mathcal L_{emit}^{ungold}
=-\sum_{(i,k)\in H}\log r_{ik}(A_{ik}).
```

When synthetic truth is available, retain the original task classification objective and optimize

```math
\mathcal L
=\lambda_a\mathcal L_{emit}+\lambda_y\mathcal L_{truth}.
```

`predictive_training_loss` implements known-truth emission supervision and latent-truth marginalization without exposing held-out labels to the context graph.

## 4. Conditional predictive null

For every audit task, the null is

```math
Y_k\sim q_k,
```

```math
A_{ik}\mid Y_k=c,G_C
\sim P_{ik}(\cdot\mid c,G_C),
\qquad i\in H_k,
```

independently across workers only conditional on the same sampled `Y_k`.

This distinction is necessary. Marginal response rows from workers on the same task are generally dependent because they share the unknown truth. The Monte Carlo implementation therefore samples one truth per task, not one independent marginal answer per edge.

## 5. Marginal categorical statistic

Integrate out task truth:

```math
r_{ik}(a)=\sum_c q_k(c)P_{ik}(a\mid c,G_C).
```

For reported option `a`, define

```math
z_a
=
\frac{\sum_{(i,k)\in H}[\mathbf 1\{A_{ik}=a\}-r_{ik}(a)]}
{\sqrt{\sum_{(i,k)\in H}r_{ik}(a)(1-r_{ik}(a))+\epsilon}}.
```

The marginal statistic is

```math
T_{marg}=\lVert z\rVert_2.
```

This retains class direction. Its finite-instance null distribution is obtained by categorical simulation rather than a diagonal Gaussian approximation.

## 6. Item-conditioned pairwise residual

For workers `i,j` held out on task `k`, the model-implied disagreement probability is

```math
p_{ijk}^{dis}
=
1-
\sum_c q_k(c)
\sum_a P_{ik}(a\mid c,G_C)P_{jk}(a\mid c,G_C).
```

Let

```math
D_{ijk}=\mathbf 1\{A_{ik}\ne A_{jk}\}.
```

For every supported worker pair,

```math
R_{ij}
=
\frac{\sum_k m_{ijk}(D_{ijk}-p_{ijk}^{dis})}
{\sqrt{\sum_k m_{ijk}p_{ijk}^{dis}(1-p_{ijk}^{dis})+\epsilon}},
\qquad R_{ii}=0.
```

Pairs with fewer than `min_pair_count` shared audit tasks are masked. The dependence statistic is

```math
T_{dep}=\lVert R\rVert_{op}
=\max\{|\lambda_{max}(R)|,|\lambda_{min}(R)|\}.
```

The two-sided norm detects coherent excess disagreement and coherent excess agreement.

## 7. Conditional Monte Carlo test

Condition on the context graph, exact split and mask, `q_k`, and every `P_ik`. For each replicate:

1. sample one `Y_k^(b) ~ q_k` per audit task;
2. sample every held-out response from `P_ik(. | Y_k^(b), G_C)`;
3. recompute `T_marg` and `T_dep`.

For `s in {marg,dep}`:

```math
p_s
=
\frac{1+\sum_b\mathbf 1\{T_s^{(b)}\ge T_s^{obs}\}}{B+1}.
```

The joint p-value is

```math
p_{joint}=\min\{1,2\min(p_{marg},p_{dep})\}.
```

Under the fixed predictive null, observed and simulated audit arrays are exchangeable. Each component p-value is finite-sample conditionally valid, and Bonferroni controls the joint false-rejection probability. Since the smallest possible joint value is `2/(B+1)`, the pipeline rejects configurations whose Monte Carlo resolution cannot reach the requested `alpha`.

## 8. Output semantics

- `reject=True`: the learned held-out annotation law is rejected at `alpha`;
- `reject=False`: insufficient evidence against the checked law;
- neither result proves or disproves task-truth correctness.

The result returns component and joint p-values, statistics, task posteriors, emission probabilities, marginal response probabilities, support counts, residual matrices, split masks, and bootstrap statistics.

## 9. Legacy implementation

`pipeline.py`, `disagreement.py`, `bootstrap.py`, and `calibration.py` implement the retired post-hoc confusion audit. They remain importable only to reproduce the negative calibration study. New experiments must use the `predictive_*` modules.
