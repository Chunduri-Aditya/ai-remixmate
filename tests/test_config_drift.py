"""
tests/test_config_drift.py — config.yaml ↔ typed-dataclass drift guard.

_dict_to_config()._fill() in scripts/core/config.py copies only YAML keys that
are declared as fields on the matching dataclass and silently drops everything
else. Consumers read via guarded getattr with defaults, so a dropped key never
errors — the YAML value just stops having any effect.

Regression this guards against: dj.masking_eq_enabled / masking_eq_strength /
masking_eq_max_atten_db existed in config.yaml but not on DJConfig, making the
masking-EQ kill switch a no-op. The generative section had no dataclass at all,
so model_manager's max_vram_gb / musicgen_model overrides were dead too.
"""

import dataclasses
import typing
from pathlib import Path

import pytest
import yaml

from scripts.core.config import AppConfig, _dict_to_config

_CONFIG_YAML = Path(__file__).parents[1] / "config.yaml"

RAW: dict = yaml.safe_load(_CONFIG_YAML.read_text(encoding="utf-8")) or {}
_APP_HINTS = typing.get_type_hints(AppConfig)  # resolves string annotations


@pytest.mark.parametrize("section", sorted(RAW))
def test_yaml_section_has_dataclass(section):
    """Every top-level config.yaml section must be a declared AppConfig field."""
    assert section in _APP_HINTS, (
        f"config.yaml section [{section}] has no field on AppConfig — "
        f"_dict_to_config() drops the whole section silently"
    )
    assert dataclasses.is_dataclass(_APP_HINTS[section]), (
        f"AppConfig.{section} is not a dataclass — _fill() will not recurse into it"
    )


@pytest.mark.parametrize("section", sorted(RAW))
def test_yaml_keys_are_dataclass_fields(section):
    """Every key inside a config.yaml section must be a declared dataclass field."""
    if section not in _APP_HINTS:
        pytest.skip("section missing entirely — reported by test_yaml_section_has_dataclass")
    section_cls = _APP_HINTS[section]
    declared = {f.name for f in dataclasses.fields(section_cls)}
    missing = sorted(set(RAW[section]) - declared)
    assert not missing, (
        f"config.yaml [{section}] keys with no {section_cls.__name__} field "
        f"(silently dropped by _dict_to_config()): {missing}"
    )


def test_yaml_values_reach_typed_config():
    """Values from config.yaml must actually land on the cfg object."""
    cfg = _dict_to_config(RAW)
    for section, keys in RAW.items():
        sub = getattr(cfg, section)
        for key, expected in keys.items():
            if expected is None:  # _fill() skips None (e.g. logging.file: null)
                continue
            actual = getattr(sub, key)
            assert actual == expected, (
                f"cfg.{section}.{key} = {actual!r}, config.yaml says {expected!r}"
            )


def test_masking_eq_kill_switch_reaches_cfg():
    """dj.masking_eq_* overrides must reach cfg.dj (the A/B kill switch)."""
    cfg = _dict_to_config(
        {
            "dj": {
                "masking_eq_enabled": False,
                "masking_eq_strength": 1.5,
                "masking_eq_max_atten_db": 9.0,
            }
        }
    )
    assert cfg.dj.masking_eq_enabled is False
    assert cfg.dj.masking_eq_strength == 1.5
    assert cfg.dj.masking_eq_max_atten_db == 9.0
