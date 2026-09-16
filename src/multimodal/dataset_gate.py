"""Fail-closed dataset gate for the commercial Dental Vision project."""
import json
from pathlib import Path

ALLOWED = {"allowed", "commercial_allowed"}


def load_policy(path="configs/dataset_policy.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_training_allowed(dataset_name: str, policy_path="configs/dataset_policy.json"):
    policy = load_policy(policy_path)
    datasets = policy.get("datasets", {})
    if dataset_name not in datasets:
        raise RuntimeError(f"Dataset {dataset_name!r} is not registered; training blocked.")
    status = datasets[dataset_name].get("commercial_training", "unknown")
    if status not in ALLOWED:
        raise RuntimeError(
            f"Commercial training blocked for {dataset_name}: status={status}. "
            "Verify license/access terms or obtain permission first."
        )
    return datasets[dataset_name]
