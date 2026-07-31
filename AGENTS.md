# CrowdFM + CbR Agent Instructions

Read `docs/CBR_IDEA.md`, `docs/CBR_SPEC.md`, and `docs/EXPERIMENT_PLAN.md` before changing CbR code.

## Statistical invariants

- Audit labels must never be visible to the model producing their item posterior.
- Estimate annotator confusion only from nuisance items.
- Predicted disagreement is item-conditioned through `q_k`; do not replace it with a global class prior.
- Use the signed two-sided operator norm, not only the largest eigenvalue.
- Conditional Monte Carlo replicates preserve the observed worker-item mask and keep nuisance estimates fixed.
- Do not claim that a small residual certifies label correctness. It only supports compatibility with the fitted predictive null.

## Engineering

- Python 3.10+, PyTorch, and pytest.
- Add type annotations to public CbR functions.
- Keep auditing code under `src/cfm/audit/` and experiment runners separate from model code.
- Every experiment must record seeds and its complete configuration.
- Run `pytest -q` before merging CbR changes.
