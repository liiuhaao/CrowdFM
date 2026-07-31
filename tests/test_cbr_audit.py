import pytest
import torch

from cfm.audit.bootstrap import conditional_monte_carlo_test
from cfm.audit.disagreement import (
    build_residual_matrix,
    confusion_posterior_parameters,
    estimate_confusion_matrices,
    item_conditioned_disagreement,
    spectral_statistic,
)
from cfm.audit.split import make_crossfit_split
from cfm.utils import normalize_seeds


def _toy_triple() -> torch.Tensor:
    # Four tasks, three workers per task, binary labels.
    workers = [0, 1, 2] * 4
    tasks = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    answers = [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1]
    return torch.tensor([workers, answers, tasks], dtype=torch.long)


def test_seed_override_normalization():
    assert normalize_seeds(42) == [42]
    assert normalize_seeds([42, 43]) == [42, 43]
    assert normalize_seeds("[42]") == [42]
    assert normalize_seeds("42,43") == [42, 43]
    with pytest.raises(ValueError):
        normalize_seeds("")


def test_crossfit_masks_are_disjoint_and_complete():
    triple = _toy_triple()
    split = make_crossfit_split(
        triple,
        num_task=4,
        audit_fraction=0.5,
        context_fraction=1 / 3,
        seed=7,
    )
    masks = [split.nuisance_edge_mask, split.context_edge_mask, split.audit_edge_mask]
    assert torch.all(masks[0] | masks[1] | masks[2])
    assert not torch.any(masks[0] & masks[1])
    assert not torch.any(masks[0] & masks[2])
    assert not torch.any(masks[1] & masks[2])
    assert torch.equal(split.model_input_edge_mask, masks[0] | masks[1])


def test_confusion_estimate_is_row_stochastic():
    triple = _toy_triple()
    posterior = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    edge_mask = torch.ones(triple.shape[1], dtype=torch.bool)
    parameters = confusion_posterior_parameters(
        triple,
        posterior,
        num_worker=3,
        num_option=2,
        edge_mask=edge_mask,
    )
    confusion = estimate_confusion_matrices(
        triple,
        posterior,
        num_worker=3,
        num_option=2,
        edge_mask=edge_mask,
    )
    assert parameters.shape == (3, 2, 2)
    assert torch.all(parameters > 0)
    assert confusion.shape == (3, 2, 2)
    assert torch.allclose(confusion.sum(-1), torch.ones(3, 2), atol=1e-6)
    assert torch.all(confusion > 0)


def test_uniform_noise_recovers_closed_form_disagreement():
    error = torch.tensor([0.2, 0.35])
    confusion = torch.stack(
        [
            torch.tensor([[1 - error[0], error[0]], [error[0], 1 - error[0]]]),
            torch.tensor([[1 - error[1], error[1]], [error[1], 1 - error[1]]]),
        ]
    )
    posterior = torch.tensor([[0.3, 0.7]])
    predicted = item_conditioned_disagreement(
        posterior,
        confusion,
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([0]),
    )[0]
    expected = error[0] + error[1] - 2 * error[0] * error[1]
    assert torch.allclose(predicted, expected, atol=1e-6)


def test_spectral_statistic_is_two_sided():
    residual = torch.tensor([[0.0, -3.0], [-3.0, 0.0]])
    assert spectral_statistic(residual) == 3.0


def test_residual_and_bootstrap_are_finite_and_reproducible():
    triple = _toy_triple()
    posterior = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
    confusion = torch.tensor(
        [
            [[0.8, 0.2], [0.2, 0.8]],
            [[0.75, 0.25], [0.25, 0.75]],
            [[0.7, 0.3], [0.3, 0.7]],
        ]
    )
    audit_mask = torch.ones(triple.shape[1], dtype=torch.bool)
    residual = build_residual_matrix(
        triple,
        posterior,
        confusion,
        audit_edge_mask=audit_mask,
    )
    assert torch.isfinite(residual.residual).all()
    assert residual.statistic >= 0

    first = conditional_monte_carlo_test(
        triple,
        posterior,
        confusion,
        audit_edge_mask=audit_mask,
        num_bootstrap=9,
        seed=123,
    )
    second = conditional_monte_carlo_test(
        triple,
        posterior,
        confusion,
        audit_edge_mask=audit_mask,
        num_bootstrap=9,
        seed=123,
    )
    assert torch.equal(first.bootstrap_statistics, second.bootstrap_statistics)
    assert first.p_value == second.p_value
    assert 0.0 < first.p_value <= 1.0


def test_confusion_refit_bootstrap_is_finite_and_reproducible():
    triple = _toy_triple()
    posterior = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
    nuisance_mask = triple[2] < 2
    audit_mask = triple[2] >= 2
    confusion = estimate_confusion_matrices(
        triple,
        posterior,
        num_worker=3,
        num_option=2,
        edge_mask=nuisance_mask,
        prior_strength=1.0,
    )

    kwargs = dict(
        triple=triple,
        task_posterior=posterior,
        confusion=confusion,
        audit_edge_mask=audit_mask,
        nuisance_edge_mask=nuisance_mask,
        refit_confusion=True,
        prior_strength=1.0,
        num_bootstrap=9,
        seed=321,
    )
    first = conditional_monte_carlo_test(**kwargs)
    second = conditional_monte_carlo_test(**kwargs)

    assert torch.isfinite(first.bootstrap_statistics).all()
    assert torch.equal(first.bootstrap_statistics, second.bootstrap_statistics)
    assert first.p_value == second.p_value
    assert 0.0 < first.p_value <= 1.0


def test_posterior_predictive_bootstrap_is_finite_and_reproducible():
    triple = _toy_triple()
    posterior = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
    nuisance_mask = triple[2] < 2
    audit_mask = triple[2] >= 2
    confusion = estimate_confusion_matrices(
        triple,
        posterior,
        num_worker=3,
        num_option=2,
        edge_mask=nuisance_mask,
        prior_strength=1.0,
    )

    kwargs = dict(
        triple=triple,
        task_posterior=posterior,
        confusion=confusion,
        audit_edge_mask=audit_mask,
        nuisance_edge_mask=nuisance_mask,
        posterior_predictive_confusion=True,
        prior_strength=1.0,
        num_bootstrap=9,
        seed=654,
    )
    first = conditional_monte_carlo_test(**kwargs)
    second = conditional_monte_carlo_test(**kwargs)

    assert torch.isfinite(first.bootstrap_statistics).all()
    assert torch.equal(first.bootstrap_statistics, second.bootstrap_statistics)
    assert first.p_value == second.p_value
    assert 0.0 < first.p_value <= 1.0
