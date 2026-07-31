# CrowdFM

Official implementation for the ICLR 2026 paper:
**Towards a Foundation Model for Crowdsourced Label Aggregation**.

![CrowdFM Overview](assets/CrowdFM.jpg)

## Installation

```bash
uv sync
source .venv/bin/activate
```

For development tests:

```bash
uv sync --extra dev
pytest -q
```

## Original CrowdFM usage

```bash
python evaluate.py
python train.py --backup
```

By default, evaluation loads `checkpoint.pt` and writes `log/perform.json`.

# CrowdSI-FM: crowd system identification and safe adaptation

The research branch changes the problem from fixed zero-shot aggregation to

```text
universal aggregation initialization
+ deployment crowd-mechanism identification
+ evidence-gated latent adaptation.
```

For a deployment annotation graph `G`, CrowdSI-FM infers a dataset-level mechanism posterior

```math
q_\phi(Z_{\mathcal D}\mid G),
```

where `Z_D` is represented compositionally through learned mechanism primitives. Conditioned on a mechanism latent, the model predicts:

```math
q_\theta(Y_k\mid G,Z_{\mathcal D}),
```

```math
P_\theta(A_{ik}=a\mid Y_k=c,G,Z_{\mathcal D},i,k),
```

and an annotation-assignment propensity

```math
\Pr_\theta(O_{ik}=1\mid G,Z_{\mathcal D},i,k).
```

The network weights remain fixed at deployment. Only the low-dimensional mechanism posterior is updated from held-out annotations.

## Evidence-gated adaptation

Audit tasks are divided into two task-disjoint folds. A posterior adapted on fold A is evaluated on fold B, and vice versa. Because an entire task stays in one fold, the evidence calculation does not split workers that share the same latent task truth.

For each direction, CrowdSI-FM compares the adapted mixture predictive likelihood with the fixed plug-in law at the amortized mechanism mean. The two likelihood ratios are averaged into an e-value. Adaptation is enabled only when

```math
E \geq 1/\alpha.
```

Under the fixed plug-in null and conditionally independent tasks, this controls erroneous adaptation by the standard e-value inequality. The guarantee concerns evidence against the base mechanism, not correctness of every aggregated truth.

## Train CrowdSI-FM

```bash
python train_crowdsi.py config=config/crowdsi_train.yaml
```

The default simulator spans IRT, class-biased, coalition, assignment-biased, and mixed crowd mechanisms. Training jointly optimizes:

- task-truth aggregation;
- masked conditional annotation generation;
- annotation-assignment prediction;
- a Gaussian mechanism prior;
- mechanism-posterior consistency across graph views.

The official CrowdFM checkpoint initializes the backbone. The new system-identification modules must be trained before deployment use.

## Evaluate and adapt

```bash
python evaluate_crowdsi.py \
  config=config/crowdsi_eval.yaml \
  checkpoint_path=log/crowdsi/10000.pt
```

Each dataset result reports the e-value, whether adaptation was statistically supported, whether adaptation was used, and—when truth is available—the base/adapted accuracy.

## Primary implementation

- `src/cfm/model/CrowdSIFM.py`: mechanism encoder, compositional basis, adapted truth head, conditional emission head, and assignment head;
- `src/cfm/data/crowdsi_simulator.py`: mechanism-diverse synthetic worlds;
- `src/cfm/si/likelihood.py`: shared-truth joint response likelihood;
- `src/cfm/si/split.py`: task-disjoint cross-fitting;
- `src/cfm/si/adaptation.py`: latent Bayesian adaptation and e-value gating;
- `src/cfm/si/training.py`: multi-objective pretraining;
- `src/cfm/si/pipeline.py`: evidence-gated deployment aggregation;
- `train_crowdsi.py` and `evaluate_crowdsi.py`: entrypoints.

## Earlier methods retained as ablations

`PredictiveCFM` is retained as a fixed-mechanism masked-annotation audit baseline. The still earlier confusion-estimator CbR path is retained only to reproduce its negative sparse-multiclass calibration results. Neither is the primary method.

## Documentation

- `docs/CROWDSI_AUTORESEARCH_PLAN.md`: single source of truth for the full idea, theory scope, implementation roadmap, experiment matrix, failure handling, AAAI criteria, and autoregressive code-agent protocol;
- `docs/IDEA.md`: formal CrowdSI-FM research problem and method;
- `docs/CROWDSI_SPEC.md`: implementation and mathematical contract;
- `docs/EXPERIMENT_PLAN.md`: mechanism-generalization and safe-adaptation experiments;
- `docs/CALIBRATION_RESULTS.md`: history of the retired confusion-based approach.

## Citation

```bibtex
@inproceedings{liu2026crowdfm,
    title={Towards a Foundation Model for Crowdsourced Label Aggregation},
    author={Liu, Hao and Liu, Jiacheng and Tang, Feilong and Chen, Long and Yu, Jiadi and Zhu, Yanmin and Dong, Qiwen and Yu, Yichuan and Hou, Xiaofeng},
    booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
    year={2026},
}
```