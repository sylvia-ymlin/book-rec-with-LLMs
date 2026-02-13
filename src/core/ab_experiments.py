"""
Minimal A/B testing framework for online experimentation.

Assigns users to control/treatment based on hash. Logs experiment variant
for metric analysis. Kept simple for research prototype use.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Optional

from src.infra.utils import setup_logger

logger = setup_logger(__name__)

# Experiment definitions: experiment_id -> (control_config, treatment_config)
# Each config is a dict of param names to values.
EXPERIMENTS: dict[str, tuple[dict, dict]] = {
    "diversity_rerank": (
        {"enable_diversity_rerank": False},  # control
        {"enable_diversity_rerank": True},   # treatment
    ),
}


def get_variant(user_id: str, experiment_id: str, salt: str = "ab") -> str:
    """
    Assign user to control or treatment based on stable hash.

    Args:
        user_id: User identifier (stable across requests).
        experiment_id: Experiment name (e.g. "diversity_rerank").
        salt: Salt for hash (enables re-randomization if needed).

    Returns:
        "control" or "treatment"
    """
    if experiment_id not in EXPERIMENTS:
        return "control"
    key = f"{salt}:{experiment_id}:{user_id}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return "treatment" if (h % 100) < 50 else "control"


def get_experiment_config(
    user_id: str,
    experiment_id: str,
    variant_override: Optional[str] = None,
) -> dict:
    """
    Get config dict for the assigned variant.

    Args:
        user_id: User identifier.
        experiment_id: Experiment name.
        variant_override: If set ("control"|"treatment"), skip assignment.
            Used when API caller forces a variant for testing.

    Returns:
        Config dict (e.g. {"enable_diversity_rerank": True}).
    """
    if experiment_id not in EXPERIMENTS:
        return {}
    control_cfg, treatment_cfg = EXPERIMENTS[experiment_id]
    if variant_override in ("control", "treatment"):
        variant = variant_override
    else:
        variant = get_variant(user_id, experiment_id)
    cfg = control_cfg if variant == "control" else treatment_cfg
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"AB experiment={experiment_id} user={user_id} variant={variant} config={cfg}")
    return cfg.copy()


def log_experiment(
    experiment_id: str,
    user_id: str,
    variant: str,
    extra: Optional[dict] = None,
) -> None:
    """
    Log experiment exposure for offline metric join.

    In production, this would emit to a metrics pipeline (e.g. Prometheus,
    data warehouse). For research prototype, simple logger.info.
    """
    msg = f"ab_experiment experiment_id={experiment_id} user_id={user_id} variant={variant}"
    if extra:
        msg += " " + " ".join(f"{k}={v}" for k, v in extra.items())
    logger.info(msg)
