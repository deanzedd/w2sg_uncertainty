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
        # CU1 fix: only prepend _CONFIGS_DIR if the path doesn't already
        # resolve from the current working directory.
        #
        # Use-cases handled correctly:
        #   (a) "--config configs/cwpo_hh_rlhf.yaml"  → exists at CWD → use as-is
        #   (b) "--config cwpo_hh_rlhf.yaml"           → not at CWD → prepend _CONFIGS_DIR
        #   (c) "--config /abs/path/to/config.yaml"    → absolute → skip this block
        #
        # Previous behaviour prepended unconditionally, turning (a) into
        # "configs/configs/cwpo_hh_rlhf.yaml" which does not exist.
        if not config_path.exists():
            config_path = _CONFIGS_DIR / config_path

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"  Searched relative to CWD and in configs/ directory.\n"
            f"  Available configs: {[p.name for p in _CONFIGS_DIR.glob('*.yaml')]}"
        )

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
