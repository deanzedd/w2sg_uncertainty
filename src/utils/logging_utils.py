"""
Logging utilities — configures rich console + file logging + WandB.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import wandb
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler


def setup_logging(cfg: DictConfig, log_level: str = "INFO") -> None:
    """
    Set up logging: rich console handler + optional file handler.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    handlers = [
        RichHandler(rich_tracebacks=True, show_time=True, show_path=False),
    ]

    # File handler
    output_dir = cfg.get("output_root", "outputs")
    log_file = os.path.join(output_dir, "experiment.log")
    os.makedirs(output_dir, exist_ok=True)
    handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )

    # Suppress noisy third-party loggers
    for noisy in ["transformers", "datasets", "accelerate", "peft"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")


def init_wandb(cfg: DictConfig, tags: Optional[list] = None) -> None:
    """Initialize Weights & Biases run."""
    if not cfg.get("use_wandb", True):
        return

    run_name = cfg.get("wandb_run_name", None)
    project = cfg.get("wandb_project", "w2sg_uncertainty")

    wandb.init(
        project=project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=tags or [],
    )
    logging.getLogger(__name__).info(
        f"WandB initialized: project={project}, run={run_name}"
    )


def finish_wandb() -> None:
    """Finish the WandB run."""
    try:
        wandb.finish()
    except Exception:
        pass
