import torch

from cfm.audit.calibration import CALIBRATION_VARIANTS, run_calibration_world
from cfm.audit.pipeline import CBRConfig
from cfm.audit.synthetic import SyntheticWorldConfig, generate_synthetic_world


class UniformModel(torch.nn.Module):
    def forward(self, data):
        return {
            "hat_task_option": torch.zeros(
                data.num_task,
                data.num_option,
                device=data.triple.device,
            )
        }


def test_synthetic_world_is_deterministic_and_well_formed():
    config = SyntheticWorldConfig(
        name="toy",
        num_worker=6,
        num_task=12,
        num_option=3,
        labels_per_task=4,
    )
    first = generate_synthetic_world(config, seed=7)
    second = generate_synthetic_world(config, seed=7)

    assert torch.equal(first.data.triple, second.data.triple)
    assert torch.equal(first.truth, second.truth)
    assert torch.allclose(
        first.true_confusion.sum(dim=-1),
        torch.ones(6, 3),
        atol=1e-6,
    )
    assert torch.all(first.data.task_degree == 4)


def test_four_way_calibration_smoke():
    config = SyntheticWorldConfig(
        name="toy",
        num_worker=6,
        num_task=16,
        num_option=2,
        labels_per_task=4,
    )
    world = generate_synthetic_world(config, seed=11)
    model = UniformModel()
    cbr_config = CBRConfig(num_bootstrap=9)
    results = {}

    for variant in CALIBRATION_VARIANTS:
        result = run_calibration_world(
            model,
            world,
            variant=variant,
            config=cbr_config,
            seed=3,
        )
        results[variant] = result
        assert 0.0 < result.p_value <= 1.0
        assert result.num_audit_edges > 0
        assert result.num_supported_pairs > 0

    oracle = results["oracle_q_oracle_p"]
    assert oracle.audit_posterior_accuracy == 1.0
    assert oracle.confusion_mae == 0.0
    assert (
        results["model_q_oracle_p"].audit_posterior_accuracy
        == results["model_q_estimated_p"].audit_posterior_accuracy
    )
