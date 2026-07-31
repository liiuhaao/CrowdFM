# CrowdSI-FM Implementation Specification

## 1. Model contract

Instantiate:

```python
model = CrowdSIFM(
    dim=d,
    layer=L,
    head=H,
    dropout=p,
    device=device,
    mechanism_latent_dim=d_z,
    num_mechanism_primitives=R,
)
```

A forward pass accepts an optional explicit mechanism latent:

```python
output = model(
    data,
    query_workers=workers,
    query_tasks=tasks,
    mechanism_latent=z,
    sample_mechanism=False,
)
```

Core outputs:

```text
hat_task_option_base          [N,K]
hat_task_option               [N,K]
mechanism_mean                [d_z]
mechanism_log_variance        [d_z]
mechanism_latent              [d_z]
mechanism_weights             [R]
mechanism_context             [d]
z_worker_si                   [M,d]
z_task_si                     [N,d]
hat_annotation_given_truth    [Q,K,K]  # when query edges are supplied
hat_assignment_logit          [Q]      # when query edges are supplied
```

`hat_annotation_given_truth[e,c,a]` is an unnormalized logit for

```math
P(A_e=a\mid Y_{k(e)}=c,G,Z).
```

The final response axis is normalized with `softmax` inside the likelihood.

## 2. Dataset-level mechanism encoder

`GaussianMechanismEncoder` uses five permutation-invariant summaries:

1. mean worker embedding;
2. mean task embedding;
3. mean option embedding;
4. mean encoded observed-edge embedding;
5. normalized worker/task degree statistics: mean, standard deviation, minimum, and maximum for each side of the bipartite graph.

The degree summary is explicit because normalized attention can otherwise erase assignment density. It lets the mechanism posterior observe sparse-selection structure even when the label values are unchanged.

The encoder returns a diagonal Gaussian. Edge order must not affect the posterior. Worker/task identity permutations remain equivariant through the CrowdFM backbone and invariant after pooling.

## 3. Compositional mechanism basis

The latent is routed to `R` primitives:

```math
\alpha=\operatorname{softmax}(WZ+b),
\qquad
C=\alpha^TB.
```

The primitive order is not identifiable and may permute between runs. Evaluation must align primitives or use permutation-invariant metrics.

## 4. Shared-truth likelihood

For every held-out task, all queried worker responses share one latent truth:

```math
\log p(A_{H_k}\mid G,Z)
=
\log\sum_c q_k(c\mid G,Z)
\prod_{e\in H_k}P_e(A_e\mid c,G,Z).
```

`joint_annotation_log_likelihood` implements this in log space. It must never be replaced by a sum of edgewise marginal log likelihoods when more than one held-out worker labels the same task.

## 5. Training split and losses

`crowdsi_training_loss` hides positive annotation edges before the forward pass. Positive assignment targets are therefore not present in the context graph. Negative targets are sampled from unobserved worker--task pairs.

The current objective contains:

```text
truth_loss
annotation_loss
assignment_loss
mechanism_kl
consistency_loss
```

Synthetic tasks use the known truth to select the emission row. A truth-free dataset uses the shared-truth joint likelihood.

A second independently masked view supplies the symmetric Gaussian KL consistency loss.

## 6. Test-time adaptation

`adapt_mechanism` freezes every network parameter and optimizes only:

```text
adapted_mean          [d_z]
adapted_log_variance  [d_z]
```

The objective is expected shared-truth log likelihood minus KL to the amortized posterior. `likelihood_temperature` and `kl_weight` implement generalized Bayes/shrinkage.

The model's train/eval state and every parameter's `requires_grad` flag are restored after adaptation.

## 7. Task-disjoint cross-fitting

`make_task_crossfit_split` first creates a context/audit edge split. It then assigns entire audit tasks to fold A or fold B.

Required invariants:

```text
context ∩ fold_A = empty
context ∩ fold_B = empty
fold_A ∩ fold_B = empty
tasks(fold_A) ∩ tasks(fold_B) = empty
```

Task-disjointness is required because workers on the same task share `Y_k`.

## 8. E-value gate

Let `z0` be the amortized posterior mean. Adapt `q_A` on fold A and `q_B` on fold B. Evaluate:

```math
E_{A\to B}
=
\frac{\int p(A_B\mid G_C,z)q_A(z)dz}
{p(A_B\mid G_C,z_0)},
```

and symmetrically `E_B->A`. Use

```math
E=\tfrac12(E_{A\to B}+E_{B\to A}).
```

The numerator integral is approximated by an arithmetic mixture of normalized predictive distributions; log likelihood uses `logsumexp - log(S)`.

Adaptation is statistically supported when

```math
E\ge1/\alpha.
```

The finite-sample e-value statement applies under the fixed plug-in null `z=z0`, conditioned on the context graph, with independent task units. It does not currently integrate uncertainty in the null denominator.

## 9. Final deployment aggregation

`run_crowdsi` performs:

1. context/audit/task-fold construction;
2. bidirectional adaptation and e-value calculation;
3. optional adaptation on all audit tasks if the gate opens;
4. final aggregation on the full observed graph at the selected mechanism mean.

`adapt_only_with_evidence=False` is an always-adapt ablation.

## 10. Simulator contract

`CrowdSISimulator` currently spans:

```text
irt
class_bias
coalition
assignment_bias
mixed
```

It stores `mechanism_target` only for evaluation and debugging. Primary mechanism learning remains generative/self-supervised. Final experiments must hold out complete families and compositions rather than train/test on the same family vocabulary.

## 11. Known incomplete components

- The assignment head is trained but the current e-value is response-only.
- The e-value denominator is a point mechanism rather than the full conditional null predictive.
- The simulator does not yet include temporal drift, legitimate subjective groups, or a low-rank Ising/Potts dependence family.
- Explicit mechanism identifiability constraints beyond the compositional simplex are not yet implemented.
- Query chunking is not implemented for the `Q*K*K` emission tensor.

These are research tasks, not hidden implementation details.
