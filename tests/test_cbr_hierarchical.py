import torch

from cfm.audit.bootstrap import conditional_monte_carlo_test
from cfm.audit.disagreement import estimate_confusion_matrices


def _toy_inputs():
    workers = [0, 1, 2] * 4
    tasks = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    answers = [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1]
    triple = torch.tensor([workers, answers, tasks], dtype=torch.long)
    posterior = torch.tensor(
        [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]
    )
    return triple, posterior


def test_global_confusion_prior_is_finite_and_row_stochastic():
    triple, posterior = _toy_inputs()
    confusion = estimate_confusion_matrices(
        triple,
        posterior,
        num_worker=3,
        num_option=2,
        edge_mask=torch.ones(triple.shape[1], dtype=torch.bool),
        prior_strength=10.0,
        prior_mode="global",
    )
    assert torch.isfinite(confusion).all()
    assert torch.all(confusion > 0)
    assert torch.allclose(confusion.sum(dim=-1), torch.ones(3, 2), atol=1e-6)


def test_global_prior_posterior_predictive_is_reproducible():
    triple, posterior = _toy_inputs()
    nuisance_mask = triple[2] < 2
    audit_mask = triple[2] >= 2
    confusion = estimate_confusion_matrices(
        triple,
        posterior,
        num_worker=3,
        num_option=2,
        edge_mask=nuisance_mask,
        prior_strength=10.0,
        prior_mode="global",
    )
    kwargs = dict(
        triple=triple,
        task_posterior=posterior,
        confusion=confusion,
        audit_edge_mask=audit_mask,
        nuisance_edge_mask=nuisance_mask,
        posterior_predictive_confusion=True,
        prior_strength=10.0,
        prior_mode="global",
        num_bootstrap=9,
        seed=777,
    )
    first = conditional_monte_carlo_test(**kwargs)
    second = conditional_monte_carlo_test(**kwargs)
    assert torch.equal(first.bootstrap_statistics, second.bootstrap_statistics)
    assert first.p_value == second.p_value
    assert 0.0 < first.p_value <= 1.0
