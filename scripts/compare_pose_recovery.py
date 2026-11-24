"""Compare pose recovery variants on matched checkpoints and initial disturbances."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

POLICIES = ("learned", "learned-search", "search-depth", "search-compensated")
SEEDS = (20261119, 20261120, 20261121)
DISTURBANCES = ("lateral_offset_x", "lateral_offset_y", "friction_scale", "tool_mass_scale")


def read_episodes(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        episodes = sorted(csv.DictReader(handle), key=lambda row: int(row["episode"]))
    if [int(row["episode"]) for row in episodes] != list(range(20)):
        raise ValueError(f"{path}: expected episode IDs 0 through 19")
    return episodes


def success(row: dict[str, str]) -> bool:
    return row["success"] == "True"


def paired(baseline: list, candidate: list) -> dict[str, int]:
    pairs = list(zip(baseline, candidate, strict=True))
    return {
        "rescued": sum(not success(a) and success(b) for a, b in pairs),
        "harmed": sum(success(a) and not success(b) for a, b in pairs),
    }


def compare(root: Path, reference: Path) -> dict:
    conditions: dict = {}
    hashes = {}
    for seed in SEEDS:
        source = root / str(seed)
        prior = reference / str(seed)
        summary = json.loads((source / "pose_estimator_summary.json").read_text(encoding="utf-8"))
        previous = json.loads((prior / "pose_estimator_summary.json").read_text(encoding="utf-8"))
        for key in (
            "seed",
            "episodes",
            "socket_clearance_m",
            "contact_detection_force_n",
            "insert_time_s",
            "episode_duration_s",
        ):
            if summary[key] != previous[key]:
                raise ValueError(f"{seed}: {key} differs from the prior experiment")
        if any(summary["search"][key] != value for key, value in previous["search"].items()):
            raise ValueError(f"{seed}: search parameters differ from the prior experiment")
        if set(summary["conditions"]) != set(previous["conditions"]):
            raise ValueError(f"{seed}: inconsistent condition set")
        hashes[seed] = summary["model_sha256"]
        for key, condition in summary["conditions"].items():
            if condition["pose_estimator"] != previous["conditions"][key]["pose_estimator"]:
                raise ValueError(f"{seed}/{key}: inconsistent estimator parameters")
            record = conditions.setdefault(
                key,
                {
                    "label": condition["label"],
                    "episodes": {policy: [] for policy in POLICIES},
                    "reference": {policy: [] for policy in POLICIES[:2]},
                },
            )
            rows = {
                policy: read_episodes(source / key / policy / "episodes.csv") for policy in POLICIES
            }
            original = {
                policy: read_episodes(prior / key / policy / "episodes.csv")
                for policy in POLICIES[:2]
            }
            expected = [[float(row[field]) for field in DISTURBANCES] for row in rows["learned"]]
            for policy_rows in [*rows.values(), *original.values()]:
                observed = [[float(row[field]) for field in DISTURBANCES] for row in policy_rows]
                if observed != expected:
                    raise ValueError(f"{seed}/{key}: mismatched paired disturbances")
            for policy in POLICIES:
                record["episodes"][policy].extend(rows[policy])
            for policy in POLICIES[:2]:
                record["reference"][policy].extend(original[policy])

    results = {}
    for key, record in conditions.items():
        rows = record["episodes"]
        results[key] = {
            "label": record["label"],
            "policies": {
                policy: {
                    "episodes": len(episodes),
                    "successes": sum(map(success, episodes)),
                    "safety_terminations": sum(
                        row["safety_violation"] == "True" for row in episodes
                    ),
                    "mean_peak_force_n": mean(
                        float(row["max_contact_force_n"]) for row in episodes
                    ),
                    "search_triggers": sum(row["search_triggered"] == "True" for row in episodes),
                    "successes_by_seed": [
                        sum(map(success, episodes[i : i + 20])) for i in range(0, len(episodes), 20)
                    ],
                }
                for policy, episodes in rows.items()
            },
            "compensated_vs_legacy": paired(rows["learned-search"], rows["search-compensated"]),
            "compensated_vs_no_search": paired(rows["learned"], rows["search-compensated"]),
            "compensated_vs_depth": paired(rows["search-depth"], rows["search-compensated"]),
            "baseline_changes_vs_previous_run": {
                policy: paired(prior, rows[policy]) for policy, prior in record["reference"].items()
            },
        }
    return {"seeds": SEEDS, "model_sha256": hashes, "conditions": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("docs/results/pose-estimator-transfer/seed-results"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.input, args.reference)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
