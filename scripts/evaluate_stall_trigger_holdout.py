"""Evaluate the frozen stall trigger and decide from matched held-out episodes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean

RESULTS = Path("docs/results")
PROTOCOL = RESULTS / "stall-trigger-holdout/protocol.json"
CHECKPOINTS = RESULTS / "pose-aware-recovery/recovery-comparison.json"
POLICIES = ("search_off", "stall_trigger")
DISTURBANCES = ("lateral_offset_x", "lateral_offset_y", "friction_scale", "tool_mass_scale")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, result: dict) -> None:
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(checkpoint: str, model_path: Path, output: Path) -> None:
    import numpy as np
    from evaluate_pose_estimator_transfer import CONDITIONS
    from stable_baselines3 import SAC

    from panda_assembly.config import load_config
    from panda_assembly.contact_search import SpiralResidualPolicy, StallTriggerConfig
    from panda_assembly.residual_rl import evaluate_residual_policy, save_episode_results

    protocol = load_json(PROTOCOL)
    model_hash = sha256(model_path)
    if model_hash != load_json(CHECKPOINTS)["model_sha256"][checkpoint]:
        raise ValueError(f"archived checkpoint hash mismatch for {checkpoint}")
    episode_count = protocol["episodes_per_checkpoint"]
    first_seed = protocol["reset_seed_start"][checkpoint]
    config = load_config(Path("configs/menagerie.toml"))
    config = replace(
        config,
        peg_task=replace(config.peg_task, insert_time=protocol["insert_time_s"]),
        residual_rl=replace(config.residual_rl, episode_duration_s=protocol["episode_duration_s"]),
    )
    agent = SAC.load(model_path, device="cpu")

    def learned(observation):
        action, _ = agent.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    params = protocol["candidate"].copy()
    stall = StallTriggerConfig(
        **{
            key: params.pop(key)
            for key in ("contact_duration_s", "minimum_progress_m", "retry_margin_s")
        }
    )
    search = SpiralResidualPolicy(
        learned,
        max_position_offset_m=np.asarray(config.residual_rl.max_position_offset_m),
        episode_duration_s=config.residual_rl.episode_duration_s,
        force_scale_n=config.force_safety.force_limit_n,
        entry_mode="compensated",
        stall_trigger=stall,
        **params,
    )
    estimators = {key: estimator for key, _, estimator in CONDITIONS}
    for condition in protocol["condition_order"]:
        for key, policy in zip(POLICIES, (learned, search), strict=True):
            results = evaluate_residual_policy(
                config,
                policy,
                episode_count,
                seed=first_seed,
                socket_clearance_m=protocol["socket_clearance_m"],
                pose_estimator_config=estimators[condition],
            )
            save_episode_results(results, output / condition / key)
        print(f"{checkpoint}: {condition} complete", flush=True)
    write_json(
        output / "manifest.json",
        {
            "checkpoint": checkpoint,
            "checkpoint_sha256": model_hash,
            "protocol_sha256": sha256(PROTOCOL),
            "reset_seeds": list(range(first_seed, first_seed + episode_count)),
            "estimators": {
                condition: asdict(estimators[condition])
                for condition in protocol["condition_order"]
            },
        },
    )


def read_episodes(path: Path, expected: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["episode"]))
    if [int(row["episode"]) for row in rows] != list(range(expected)):
        raise ValueError(f"{path}: episode IDs differ from protocol")
    return rows


def flag(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"invalid episode flag {value!r}")
    return value == "True"


def collect_pairs(directory: Path, protocol: dict) -> tuple[list[dict], dict]:
    pairs: list[dict] = []
    manifests = {}
    expected = protocol["episodes_per_checkpoint"]
    checkpoints = load_json(CHECKPOINTS)["model_sha256"]
    historical_seeds = {int(seed) + episode for seed in checkpoints for episode in range(20)}
    used_seeds: set[int] = set()
    for checkpoint, first_seed in protocol["reset_seed_start"].items():
        folder = directory / checkpoint
        manifest = load_json(folder / "manifest.json")
        seed_range = list(range(first_seed, first_seed + expected))
        if (
            manifest["checkpoint"] != checkpoint
            or manifest["checkpoint_sha256"] != checkpoints[checkpoint]
            or manifest["protocol_sha256"] != sha256(PROTOCOL)
            or manifest["reset_seeds"] != seed_range
            or used_seeds.intersection(seed_range)
            or historical_seeds.intersection(seed_range)
        ):
            raise ValueError(f"{checkpoint}: held-out provenance check failed")
        used_seeds.update(seed_range)
        manifests[checkpoint] = manifest
        expected_disturbance: dict[int, tuple] = {}
        for condition in protocol["condition_order"]:
            control, candidate = (
                read_episodes(folder / condition / policy / "episodes.csv", expected)
                for policy in POLICIES
            )
            for reset_seed, a, b in zip(seed_range, control, candidate, strict=True):
                vector = tuple(float(a[field]) for field in DISTURBANCES)
                if vector != tuple(float(b[field]) for field in DISTURBANCES):
                    raise ValueError(f"{condition}/{reset_seed}: policies received different cases")
                if vector != expected_disturbance.setdefault(reset_seed, vector):
                    raise ValueError(f"{condition}/{reset_seed}: disturbance changed by condition")
                record = {
                    "condition": condition,
                    "checkpoint": int(checkpoint),
                    "reset_seed": reset_seed,
                    **dict(zip(DISTURBANCES, vector, strict=True)),
                }
                for key, row in zip(POLICIES, (a, b), strict=True):
                    for field in ("success", "safety_violation", "search_triggered"):
                        record[f"{key}_{field}"] = flag(row[field])
                    record[f"{key}_peak_force_n"] = float(row["max_contact_force_n"])
                    record[f"{key}_completion_time_s"] = (
                        float(row["completion_time_s"]) if row["completion_time_s"] else None
                    )
                if record["search_off_search_triggered"]:
                    raise ValueError("search-off baseline triggered")
                pairs.append(record)
    all_estimator_configs = [manifest["estimators"] for manifest in manifests.values()]
    if any(config != all_estimator_configs[0] for config in all_estimator_configs[1:]):
        raise ValueError("pose-estimator configurations differ across checkpoints")
    return pairs, manifests


def summarize(pairs: list[dict]) -> dict:
    rescued = sum(not row["search_off_success"] and row["stall_trigger_success"] for row in pairs)
    harmed = sum(row["search_off_success"] and not row["stall_trigger_success"] for row in pairs)
    added_safety = sum(
        not row["search_off_safety_violation"] and row["stall_trigger_safety_violation"]
        for row in pairs
    )
    force_deltas = [
        row["stall_trigger_peak_force_n"] - row["search_off_peak_force_n"] for row in pairs
    ]
    return {
        "pairs": len(pairs),
        "rescued": rescued,
        "harmed": harmed,
        "net_rescues": rescued - harmed,
        "new_safety_terminations": added_safety,
        "mean_paired_peak_force_delta_n": mean(force_deltas),
        "max_paired_peak_force_increase_n": max(0.0, max(force_deltas)),
        "policies": {
            key: {
                "successes": sum(row[f"{key}_success"] for row in pairs),
                "search_triggers": sum(row[f"{key}_search_triggered"] for row in pairs),
                "safety_terminations": sum(row[f"{key}_safety_violation"] for row in pairs),
                "mean_episode_peak_force_n": mean(row[f"{key}_peak_force_n"] for row in pairs),
                "maximum_episode_peak_force_n": max(row[f"{key}_peak_force_n"] for row in pairs),
            }
            for key in POLICIES
        },
    }


def apply_decision(conditions: dict, protocol: dict) -> dict:
    if set(conditions) != set(protocol["condition_order"]):
        raise ValueError("decision requires every prespecified sensing condition")
    limits = protocol["acceptance"]
    primary = protocol["primary_condition"]
    checks = {}
    for condition, result in conditions.items():
        threshold = (
            limits["minimum_primary_net_rescues"]
            if condition == primary
            else limits["minimum_net_rescues_in_other_conditions"]
        )
        checks[condition] = {
            "net_rescue_requirement_met": result["net_rescues"] >= threshold,
            "new_safety_limit_met": (
                result["new_safety_terminations"] <= limits["maximum_new_safety_terminations"]
            ),
        }
    passed = all(all(values.values()) for values in checks.values())
    return {
        "accepted": passed,
        "policy": "stall_trigger" if passed else "search_off",
        "checks": checks,
    }


def aggregate(directory: Path, output: Path) -> None:
    protocol = load_json(PROTOCOL)
    pairs, manifests = collect_pairs(directory, protocol)
    conditions = {}
    expected = len(manifests) * protocol["episodes_per_checkpoint"]
    for condition in protocol["condition_order"]:
        subset = [row for row in pairs if row["condition"] == condition]
        distinct = {tuple(row[name] for name in DISTURBANCES) for row in subset}
        if len(subset) != expected or len(distinct) != expected:
            raise ValueError(f"{condition}: held-out case count or uniqueness mismatch")
        conditions[condition] = {
            **summarize(subset),
            "distinct_disturbances": len(distinct),
            "per_checkpoint": {
                checkpoint: summarize(
                    [row for row in subset if row["checkpoint"] == int(checkpoint)]
                )
                for checkpoint in manifests
            },
        }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "paired-episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    result = {
        "protocol_sha256": sha256(PROTOCOL),
        "paired_cases": len(pairs),
        "total_policy_episodes": 2 * len(pairs),
        "manifests": manifests,
        "conditions": conditions,
        "decision": apply_decision(conditions, protocol),
    }
    write_json(output / "stall-trigger-holdout.json", result)
    print(json.dumps(result["decision"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("evaluate")
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser("aggregate")
    summary.add_argument("--input", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args.checkpoint, args.model, args.output)
    else:
        aggregate(args.input, args.output)


if __name__ == "__main__":
    main()
