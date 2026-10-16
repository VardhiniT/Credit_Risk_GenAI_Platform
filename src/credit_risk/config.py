"""Load the YAML feature contract. One place, used by every stage."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    check_no_forbidden(cfg)
    return cfg


def model_inputs(cfg):
    f = cfg["features"]
    return list(f["numeric"]) + list(f["categorical"]) + list(f["derived"])


def check_no_forbidden(cfg):
    """Leakage / fairness guard: fail loudly if a forbidden column is a model input."""
    bad = set(model_inputs(cfg)) & set(cfg["forbidden_features"])
    if bad:
        raise ValueError(f"Forbidden columns used as model inputs: {sorted(bad)}")
