# CrowdSI-FM: Amortized System Identification for Crowd Foundation Models

## One-sentence definition

CrowdSI-FM pretrains a permutation-equivariant model to infer the latent data-generating mechanism of an entire crowd dataset, updates only that low-dimensional mechanism at deployment, and changes aggregation only when task-disjoint held-out annotations provide sufficient predictive evidence.

## 1. Research problem

For workers `i in [M]`, tasks `k in [N]`, and options `a in [K]`, observe

```math
G=\{(i,k,A_{ik}):O_{ik}=1\},
```

where `O_ik` is the assignment indicator and `Y_k` is the unavailable task truth. CrowdFM learns one fixed map

```math
G\mapsto q_\theta(Y_k\mid G)
```

across synthetic worlds. A deployment dataset can nevertheless have a different worker population, task distribution, assignment policy, class-bias pattern, or dependence mechanism. A fixed forward pass has no explicit object to update when that data-generating mechanism changes.

The research question is:

> Can a crowd foundation model amortize an initialization for the deployment crowd mechanism, identify that mechanism from unlabeled annotations at test time, and adapt aggregation only when independent predictive evidence shows that adaptation improves over the zero-shot mechanism?

This is system identification and safe adaptation, not label-free certification of task truth.

## 2. Mechanism model

CrowdSI-FM uses the worker--task--edge factorization suggested by separately exchangeable arrays. A dataset has a global mechanism latent `Z_D`, worker/task representations `U_i,V_k`, truths `Y_k`, assignments `O_ik`, and responses `A_ik`:

```math
Z_{\mathcal D}\sim p(Z),
```

```math
O_{ik}\sim p_\theta(O_{ik}\mid U_i,V_k,Z_{\mathcal D}),
```

```math
Y_k\sim p_\theta(Y_k\mid V_k,Z_{\mathcal D}),
```

```math
A_{ik}\mid O_{ik}=1,Y_k
\sim p_\theta(A_{ik}\mid Y_k,U_i,V_k,Z_{\mathcal D}).
```

Worker and task identifiers are exchangeable; the architecture is invariant at dataset level and equivariant in worker/task outputs.

## 3. Amortized compositional mechanism posterior

The CrowdFM backbone produces worker, task, and option embeddings. The mechanism encoder pools five invariant summaries:

1. mean worker representation;
2. mean task representation;
3. mean option representation;
4. mean encoded annotation-edge representation;
5. normalized worker/task degree statistics—mean, standard deviation, minimum, and maximum on both sides of the bipartite graph.

The explicit degree summary is necessary because normalized graph attention can erase assignment density. The encoder returns

```math
q_\phi(Z_{\mathcal D}\mid G)
=\mathcal N(\mu_G,\operatorname{diag}(\sigma_G^2)).
```

A mechanism sample routes over learned primitives `B_1,...,B_R`:

```math
\alpha(Z)=\operatorname{softmax}(WZ+b),
\qquad
C(Z)=\sum_{r=1}^R\alpha_r(Z)B_r.
```

The intended benefit is generalization to unseen compositions of mechanisms rather than recognition of a finite simulator-family ID.

## 4. Mechanism-conditioned decoders

Conditioned on `C(Z)`, the model predicts:

### Task truth

```math
q_\theta(Y_k=c\mid G,Z).
```

### Edge-conditioned emission law

```math
P_{ik}^{Z}(a\mid c)
=\Pr_\theta(A_{ik}=a\mid Y_k=c,G,i,k,Z).
```

The scorer evaluates every candidate truth and reported option, preserving variable `K`.

### Assignment law

```math
\pi_{ik}^{Z}
=\Pr_\theta(O_{ik}=1\mid G,i,k,Z).
```

The mask is therefore part of the modeled crowd process rather than assumed missing completely at random.

## 5. Pretraining objective

For two independently masked views of one world, optimize

```math
\mathcal L
=\lambda_Y\mathcal L_Y
+\lambda_A\mathcal L_A
+\lambda_O\mathcal L_O
+\lambda_{KL}\mathcal L_{KL}
+\lambda_C\mathcal L_C.
```

Truth loss:

```math
\mathcal L_Y=-\sum_k\log q_\theta(Y_k\mid G_C,Z).
```

When synthetic truth is known, masked response loss is

```math
\mathcal L_A
=-\sum_{(i,k)\in H}\log P_{ik}^{Z}(A_{ik}\mid Y_k).
```

Without truth, use the task-joint likelihood

```math
\log p(A_{H_k}\mid G_C,Z)
=\log\sum_c q_\theta(c\mid G_C,Z)
\prod_{i:(i,k)\in H_k}P_{ik}^{Z}(A_{ik}\mid c).
```

The product remains inside one truth summation per task. Edgewise marginal likelihood would incorrectly discard dependence induced by the shared unknown `Y_k`.

Assignment loss uses hidden positive edges and sampled non-edges:

```math
\mathcal L_O
=-\sum_{i,k}\big[O_{ik}\log\pi_{ik}^{Z}+(1-O_{ik})\log(1-\pi_{ik}^{Z})\big].
```

Mechanism regularization is

```math
\mathcal L_{KL}=D_{KL}(q_\phi(Z\mid G_C)\Vert\mathcal N(0,I)),
```

and two graph views are aligned by symmetric Gaussian KL. The consistency term discourages a global latent that merely memorizes one mask realization.

## 6. Test-time system identification

The context graph gives

```math
q_0(Z)=\mathcal N(\mu_0,\operatorname{diag}(\sigma_0^2)).
```

All network weights are frozen. On adaptation tasks `H_A`, optimize only

```math
q_A(Z)=\mathcal N(\mu_A,\operatorname{diag}(\sigma_A^2))
```

using

```math
\max_{q_A}
\frac{1}{\eta}\mathbb E_{Z\sim q_A}
[\log p_\theta(A_{H_A}\mid G_C,Z)]
-\beta D_{KL}(q_A\Vert q_0).
```

This is low-dimensional generalized-Bayes system identification, not unconstrained test-time network fine-tuning.

## 7. Task-disjoint e-value gating

Audit tasks are divided into folds `H_A,H_B`; an entire task stays in one fold, because splitting workers from the same task would share the latent truth across folds.

Adapt `q_A` on A and evaluate on B:

```math
E_{A\to B}
=\frac{\int p_\theta(A_{H_B}\mid G_C,Z)q_A(Z)dZ}
{p_\theta(A_{H_B}\mid G_C,Z=\mu_0)}.
```

Define `E_B->A` symmetrically and combine

```math
E=\frac12(E_{A\to B}+E_{B\to A}).
```

Under the fixed plug-in null `Z=mu_0`, conditioned on the context graph and with independent task units, each numerator is a normalized predictive density selected using the opposite fold. Therefore

```math
\mathbb E_0[E]\le1,
\qquad
\Pr_0(E\ge1/\alpha)\le\alpha.
```

CrowdSI-FM enables adaptation only when `E >= 1/alpha`. It then refits the mechanism posterior on all audit tasks and performs final aggregation using the selected posterior mean.

The guarantee is deliberately narrow: it controls replacing the fixed plug-in mechanism under its own null. It neither repairs arbitrary misspecification nor certifies task truths.

## 8. Distinction from PredictiveCFM and legacy CbR

The confusion-based CbR path manufactured a deployment response law from a frozen discriminative checkpoint and failed sparse-multiclass calibration.

PredictiveCFM corrected that problem by learning a fixed emission law and testing held-out compatibility. It remains an audit/defer baseline.

CrowdSI-FM goes further:

- it represents a dataset mechanism explicitly;
- learns compositional mechanism primitives;
- models assignment and response generation;
- performs unlabeled test-time mechanism inference;
- uses predictive evidence to gate adaptation;
- changes final aggregation constructively.

## 9. Required main-conference evidence

A paper-level claim requires:

1. no in-prior loss relative to CrowdFM;
2. held-out mechanism-family generalization;
3. held-out composition generalization;
4. improved truth aggregation from latent adaptation;
5. controlled false adaptation under the exact plug-in null;
6. e-value growth under KL-separated alternatives;
7. assignment-shift gains from the mask model;
8. real-benchmark improvements without deployment gold labels.

The central experiment is not family classification. It is training on mechanism primitives individually and adapting to combinations never seen during pretraining.

## 10. Current limitations

- The current e-value is response-only; the assignment head is a training/system-identification signal.
- The null denominator is a point mechanism rather than the full conditional null predictive.
- Mechanism latents are identifiable only up to transformations without extra restrictions.
- The simulator does not yet cover temporal drift, legitimate subjective groups, or low-rank Ising/Potts dependence.
- Observationally equivalent truth/noise mechanisms remain inseparable without trusted labels, metadata, temporal structure, or intervention.
- Query chunking for the `Q*K*K` emission tensor remains engineering work.

## 11. Primary implementation

- `src/cfm/model/CrowdSIFM.py`;
- `src/cfm/data/crowdsi_simulator.py`;
- `src/cfm/si/likelihood.py`;
- `src/cfm/si/split.py`;
- `src/cfm/si/adaptation.py`;
- `src/cfm/si/training.py`;
- `src/cfm/si/pipeline.py`;
- `train_crowdsi.py`;
- `evaluate_crowdsi.py`;
- `tests/test_crowdsi.py`.
