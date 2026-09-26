"""Configuration loading.

The YAML file is loaded into a read-only attribute tree so modules can write
``cfg.avoidance.safe_distance`` instead of nested dict lookups. Overrides can
be applied with dotted keys (used by tests and the simulator).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "atlas.yaml"


class Config(Mapping):
    """Immutable attribute-access view over a nested dict."""

    def __init__(self, data: Mapping[str, Any]):
        object.__setattr__(self, "_data", dict(data))

    def __getattr__(self, key: str) -> Any:
        try:
            return self._wrap(self._data[key])
        except KeyError as exc:
            raise AttributeError(f"config has no key '{key}'") from exc

    def __setattr__(self, key, value):
        raise AttributeError("Config is read-only; use with_overrides()")

    def __getitem__(self, key):
        return self._wrap(self._data[key])

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def get(self, key, default=None):
        return self._wrap(self._data.get(key, default))

    @staticmethod
    def _wrap(value):
        if isinstance(value, Mapping) and not isinstance(value, Config):
            return Config(value)
        if isinstance(value, list):
            return [Config(v) if isinstance(v, Mapping) else v for v in value]
        return value

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Config":
        """Return a copy with dotted-key overrides, e.g. {"avoidance.safe_distance": 3}."""
        data = self.to_dict()
        for dotted, value in overrides.items():
            node = data
            parts = dotted.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = value
        return Config(data)


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = Config(data)
    return cfg.with_overrides(overrides) if overrides else cfg
