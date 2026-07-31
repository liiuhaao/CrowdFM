"""Deployment-time auditing utilities for CrowdFM.

The primary API is the masked-annotation predictive audit exported below. The
older confusion-matrix CbR implementation remains available for reproducing the
negative calibration study, but it is no longer the recommended method.
"""

from .predictive import (
    PredictiveResidualResult,
    build_predictive_residual,
    marginal_categorical_statistic,
    marginal_response_probabilities,
    normalize_annotation_probabilities,
    spectral_statistic as predictive_spectral_statistic,
    standardized_categorical_residuals,
)
from .predictive_bootstrap import (
    PredictiveMonteCarloResult,
    conditional_predictive_test,
)
from .predictive_pipeline import PredictiveAuditConfig, run_predictive_audit
from .predictive_split import AnnotationAuditSplit, make_annotation_audit_split
from .predictive_training import PredictiveTrainingConfig, predictive_training_loss

# Legacy confusion-based CbR exports retained for result reproduction.
from .bootstrap import MonteCarloResult, conditional_monte_carlo_test
from .calibration import (
    CALIBRATION_VARIANTS,
    CalibrationWorldResult,
    run_calibration_world,
)
from .disagreement import (
    ResidualResult,
    build_residual_matrix,
    confusion_posterior_parameters,
    estimate_confusion_matrices,
    item_conditioned_disagreement,
    spectral_statistic,
)
from .pipeline import CBRConfig, run_cbr_audit
from .split import CrossFitSplit, make_crossfit_split
from .synthetic import (
    SyntheticWorld,
    SyntheticWorldConfig,
    generate_synthetic_world,
)

__all__ = [
    # Primary predictive-audit API.
    "AnnotationAuditSplit",
    "PredictiveAuditConfig",
    "PredictiveMonteCarloResult",
    "PredictiveResidualResult",
    "PredictiveTrainingConfig",
    "build_predictive_residual",
    "conditional_predictive_test",
    "make_annotation_audit_split",
    "marginal_categorical_statistic",
    "marginal_response_probabilities",
    "normalize_annotation_probabilities",
    "predictive_spectral_statistic",
    "predictive_training_loss",
    "run_predictive_audit",
    "standardized_categorical_residuals",
    # Legacy confusion-based API.
    "CALIBRATION_VARIANTS",
    "CBRConfig",
    "CalibrationWorldResult",
    "CrossFitSplit",
    "MonteCarloResult",
    "ResidualResult",
    "SyntheticWorld",
    "SyntheticWorldConfig",
    "build_residual_matrix",
    "conditional_monte_carlo_test",
    "confusion_posterior_parameters",
    "estimate_confusion_matrices",
    "generate_synthetic_world",
    "item_conditioned_disagreement",
    "make_crossfit_split",
    "run_calibration_world",
    "run_cbr_audit",
    "spectral_statistic",
]
