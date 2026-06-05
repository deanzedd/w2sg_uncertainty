"""
Config utilities — load and merge OmegaConf configs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf


_CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"


def load_config(config_path: str, overrides: Optional[list] = None) -> DictConfig:
    """
    Load an experiment config, merging with base.yaml.

    Args:
        config_path:  path to experiment YAML (absolute or relative to configs/)
        overrides:    list of "key=value" strings to override specific fields

    Returns:
        merged DictConfig
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = _CONFIGS_DIR / config_path

    # Load base config
    base_cfg = OmegaConf.load(_CONFIGS_DIR / "base.yaml")

    # Load experiment config
    exp_cfg = OmegaConf.load(config_path)

    # Remove 'defaults' key if present (we handle it manually)
    if "defaults" in exp_cfg:
        exp_dict = OmegaConf.to_container(exp_cfg, resolve=False)
        exp_dict.pop("defaults", None)
        exp_cfg = OmegaConf.create(exp_dict)

    # Merge: base <- experiment
    cfg = OmegaConf.merge(base_cfg, exp_cfg)

    # Apply CLI overrides
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg


def print_config(cfg: DictConfig) -> None:
    """Pretty-print config."""
    print(OmegaConf.to_yaml(cfg))
