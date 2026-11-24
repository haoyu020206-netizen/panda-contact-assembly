"""Develop a stall trigger on archived combined-disturbance episodes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import numpy as np

POLICIES = ("learned", "search-compensated", "search-stall")


def trigger_context(rows: list[dict], baseline_rows: list[dict], horizon_s: float) -> dict | None:
    row = next((sample for sample in rows if sample["search_triggered"]), None)
    if row is None:
        return None
    trigger_s = row["time_s"]
    tolerance_s = 1e-9  # Include the sample at the window boundary despite clock rounding.
    reference_rows = [
        sample
        for sample in baseline_rows
        if trigger_s - tolerance_s <= sample["time_s"] <= trigger_s + horizon_s + tolerance_s
    ]
    progress_mm = None
    if reference_rows:
        depth_mm = reference_rows[0]["depth_mm"]
        progress_mm = max(sample["depth_mm"] - depth_mm for sample in reference_rows)
    return {
        "time_s": trigger_s,
        "force_n": row["contact_force_n"],
        "depth_mm": row["depth_mm"],
        "pose_age_ms": row["pose_age_ms"],
        "remaining_s": row["remaining_s"],
        "search_off_progress_next_window_mm": progress_mm,
    }


def run(model_path: Path, reference: Path, output: Path) -> dict:
    from render_paired_recovery_demo import (
        CHECKPOINT_SHA256,
        FORCE_TRIGGER_N,
        MAX_RECOVERY_POSE_AGE_S,
        SEED,
        record,
        replay_environment,
        verify,
    )
    from stable_baselines3 import SAC

    from panda_assembly.contact_search import SpiralResidualPolicy, StallTriggerConfig

    if hashlib.sha256(model_path.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("checkpoint differs from the archived 20261119 policy")
    model = SAC.load(model_path, device="cpu")

    def learned(observation):
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    env = replay_environment()
    config = env.config
    stall = StallTriggerConfig(
        contact_duration_s=config.contact_recovery.jam_timeout_s,
        minimum_progress_m=config.contact_recovery.jam_progress_m,
    )
    searches = [
        SpiralResidualPolicy(
            learned,
            max_position_offset_m=np.asarray(config.residual_rl.max_position_offset_m),
            episode_duration_s=config.residual_rl.episode_duration_s,
            force_scale_n=config.force_safety.force_limit_n,
            trigger_force_n=FORCE_TRIGGER_N,
            search_radius_m=0.007,
            entry_depth_m=config.contact_recovery.entry_depth_m,
            entry_mode="compensated",
            max_pose_age_s=MAX_RECOVERY_POSE_AGE_S,
            stall_trigger=gate,
        )
        for gate in (None, stall)
    ]
    archive_path = reference / "learned" / "episodes.csv"
    with archive_path.open(newline="", encoding="utf-8") as handle:
        episode_ids = [int(row["episode"]) for row in csv.DictReader(handle)]
    if episode_ids != list(range(20)):
        raise ValueError("expected the 20 archived development episodes")

    episodes = []
    try:
        for episode in episode_ids:
            replay = [
                record(env, learned, search, reset_seed=SEED + episode, trigger_diagnostics=True)
                for search in (None, *searches)
            ]
            archive_deltas = {
                policy: verify(
                    trace,
                    reference / policy / "episodes.csv",
                    episode=episode,
                    strict_metrics=False,
                )
                for policy, trace in zip(POLICIES[:2], replay[:2], strict=True)
            }
            off, old, candidate = replay
            if off.summary["success"] and not old.summary["success"]:
                group = "search_harmed_baseline_success"
            elif not off.summary["success"] and not old.summary["success"]:
                group = "both_failed"
            else:
                group = "other"
            for key, trace in zip(POLICIES, replay, strict=True):
                target = output / "traces" / f"episode-{episode:02d}" / f"{key}.csv"
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(trace.rows[0]))
                    writer.writeheader()
                    writer.writerows(trace.rows)
            episodes.append(
                {
                    "episode": episode,
                    "reset_seed": SEED + episode,
                    "archive_deltas": archive_deltas,
                    "group": group,
                    "outcomes": {
                        key: trace.summary for key, trace in zip(POLICIES, replay, strict=True)
                    },
                    "old_trigger": trigger_context(old.rows, off.rows, stall.contact_duration_s),
                    "stall_trigger": trigger_context(
                        candidate.rows, off.rows, stall.contact_duration_s
                    ),
                    "stall_decisions": dict(
                        Counter(row["trigger_decision"] for row in candidate.rows)
                    ),
                }
            )
    finally:
        env.close()

    results = {}
    for key in POLICIES:
        cases = [row["outcomes"][key] for row in episodes]
        baseline = [row["outcomes"]["learned"] for row in episodes]
        pairs = list(zip(baseline, cases, strict=True))
        results[key] = {
            "successes": sum(case["success"] for case in cases),
            "triggers": sum(case["trigger_time_s"] is not None for case in cases),
            "safety_terminations": sum(case["safety_violation"] for case in cases),
            "mean_episode_peak_force_n": mean(case["peak_force_n"] for case in cases),
            "maximum_episode_peak_force_n": max(case["peak_force_n"] for case in cases),
            "rescued_vs_search_off": sum(not a["success"] and b["success"] for a, b in pairs),
            "harmed_vs_search_off": sum(a["success"] and not b["success"] for a, b in pairs),
            "new_safety_terminations_vs_search_off": sum(
                not a["safety_violation"] and b["safety_violation"] for a, b in pairs
            ),
        }
    recovery = searches[1]
    return {
        "evaluation": "historical development cases",
        "reference_run": 45476321699,
        "checkpoint_seed": SEED,
        "model_sha256": CHECKPOINT_SHA256,
        "condition": "camera-combined",
        "case_count": len(episodes),
        "candidate": {
            **asdict(stall),
            "force_threshold_n": FORCE_TRIGGER_N,
            "maximum_pose_age_s": MAX_RECOVERY_POSE_AGE_S,
            "minimum_remaining_s": (
                recovery.retract_duration_s + recovery.search_duration_s + stall.retry_margin_s
            ),
        },
        "policies": results,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.model, args.reference, args.output)
    (args.output / "stall-trigger-development.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["policies"], indent=2))


if __name__ == "__main__":
    main()
