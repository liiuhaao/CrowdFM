import math

import torch

from cfm.data.crowd_data import CrowdData
from cfm.data.crowdsi_simulator import CrowdSISimulator
from cfm.model.CrowdSIFM import CrowdSIFM, GaussianMechanismEncoder
from cfm.si.adaptation import AdaptationConfig, crossfit_e_value
from cfm.si.likelihood import joint_annotation_log_likelihood
from cfm.si.pipeline import CrowdSIPipelineConfig, run_crowdsi
from cfm.si.split import make_task_crossfit_split
from cfm.si.training import CrowdSITrainingConfig, crowdsi_training_loss


def _toy_data() -> CrowdData:
    workers = []
    answers = []
    tasks = []
    task_y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)
    for task, truth in enumerate(task_y.tolist()):
        for worker in range(4):
            workers.append(worker)
            tasks.append(task)
            answers.append(truth if worker < 3 else (truth + 1) % 3)
    data = CrowdData(
        dim=4,
        num_worker=4,
        num_task=8,
        num_option=3,
        triple=torch.tensor([workers, answers, tasks], dtype=torch.long),
    )
    data.task_y = task_y
    data.setup()
    return data


def _model() -> CrowdSIFM:
    return CrowdSIFM(
        dim=4,
        layer=1,
        head=1,
        dropout=0.0,
        device="cpu",
        mechanism_latent_dim=3,
        num_mechanism_primitives=4,
    )


def test_crowdsi_output_contract_and_variable_k():
    data = _toy_data()
    model = _model()
    output = model(
        data,
        query_workers=torch.tensor([0, 3]),
        query_tasks=torch.tensor([1, 5]),
        sample_mechanism=False,
    )
    assert output["hat_task_option"].shape == (data.num_task, data.num_option)
    assert output["hat_annotation_given_truth"].shape == (2, 3, 3)
    assert output["hat_assignment_logit"].shape == (2,)
    assert output["mechanism_mean"].shape == (3,)
    assert torch.allclose(output["mechanism_weights"].sum(), torch.tensor(1.0))


def test_mechanism_encoder_is_edge_permutation_invariant():
    torch.manual_seed(3)
    encoder = GaussianMechanismEncoder(dim=4, latent_dim=3)
    z_worker = torch.randn(4, 4)
    z_task = torch.randn(5, 4)
    z_option = torch.randn(3, 4)
    triple = torch.tensor(
        [[0, 1, 2, 3, 0, 2], [0, 1, 2, 0, 2, 1], [0, 1, 2, 3, 4, 4]],
        dtype=torch.long,
    )
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    first = encoder(z_worker, z_task, z_option, triple)
    second = encoder(z_worker, z_task, z_option, triple[:, permutation])
    assert torch.allclose(first[0], second[0], atol=1e-6)
    assert torch.allclose(first[1], second[1], atol=1e-6)


def test_joint_likelihood_preserves_one_shared_truth_per_task():
    task_probability = torch.tensor([[0.6, 0.4]], dtype=torch.float64)
    task_logits = task_probability.log()
    emissions = torch.tensor(
        [
            [[0.9, 0.1], [0.2, 0.8]],
            [[0.7, 0.3], [0.4, 0.6]],
        ],
        dtype=torch.float64,
    )
    emission_logits = emissions.log()
    answers = torch.tensor([0, 1])
    query_tasks = torch.tensor([0, 0])
    observed = joint_annotation_log_likelihood(
        task_logits,
        emission_logits,
        answers,
        query_tasks,
    )
    expected = math.log(0.6 * 0.9 * 0.3 + 0.4 * 0.2 * 0.6)
    assert abs(float(observed.item()) - expected) < 1e-8


def test_task_crossfit_folds_do_not_share_task_truths():
    data = _toy_data()
    split = make_task_crossfit_split(
        data.triple,
        data.num_task,
        num_worker=data.num_worker,
        context_fraction=0.5,
        min_worker_context_edges=1,
        seed=7,
    )
    tasks = data.triple[2]
    fold_a_tasks = torch.unique(tasks[split.fold_a_edge_mask])
    fold_b_tasks = torch.unique(tasks[split.fold_b_edge_mask])
    assert not torch.any(torch.isin(fold_a_tasks, fold_b_tasks))
    assert not torch.any(split.context_edge_mask & split.audit_edge_mask)


def test_crowdsi_training_loss_backpropagates_to_system_id_heads():
    data = _toy_data()
    model = _model()
    result = crowdsi_training_loss(
        model,
        data,
        config=CrowdSITrainingConfig(
            context_fraction=0.5,
            min_worker_context_edges=0,
            assignment_negative_ratio=0.5,
        ),
        seed=11,
    )
    result["loss"].backward()
    modules = [
        model.mechanism_encoder,
        model.emission_head,
        model.assignment_head,
        model.truth_scorer,
    ]
    for module in modules:
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        assert gradients
        assert any(gradient is not None for gradient in gradients)
    assert torch.isfinite(result["loss"])


def test_crossfit_e_value_runs_with_latent_only_adaptation():
    data = _toy_data()
    model = _model()
    result = crossfit_e_value(
        model,
        data,
        adaptation_config=AdaptationConfig(
            steps=2,
            learning_rate=0.01,
            num_elbo_samples=1,
        ),
        alpha=0.5,
        num_predictive_samples=2,
        context_fraction=0.5,
        seed=13,
    )
    assert math.isfinite(result.log_e_value)
    assert result.e_value > 0
    assert result.threshold == 2.0
    assert result.posterior_a.mean.shape == result.base_posterior.mean.shape


def test_crowdsi_pipeline_and_mechanism_diverse_simulator():
    simulator = CrowdSISimulator(
        dim=4,
        num_worker_range=(6, 6),
        num_task_range=(8, 8),
        num_option_range=(3, 3),
        num_answer_each_task_range=(4, 4),
        min_answers_per_task=3,
        mechanism_families=["mixed"],
    )
    simulated = simulator.generate()
    assert simulated.mechanism_family_name == "mixed"
    assert simulated.mechanism_target.shape == (8,)
    assert torch.all(simulated.task_degree >= 3)

    data = _toy_data()
    model = _model()
    model.mark_crowdsi_trained()
    result = run_crowdsi(
        model,
        data,
        config=CrowdSIPipelineConfig(
            alpha=0.5,
            num_predictive_samples=1,
            context_fraction=0.5,
            adaptation_steps=1,
            adaptation_num_elbo_samples=1,
        ),
        seed=17,
    )
    assert result["task_posterior"].shape == (data.num_task, data.num_option)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["e_value"] > 0
