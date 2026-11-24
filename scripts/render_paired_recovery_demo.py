"""Replay and render the verified combined-disturbance recovery counterexample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import matplotlib
import mujoco
import numpy as np
from evaluate_pose_estimator_transfer import CONDITIONS, MAX_RECOVERY_POSE_AGE_S
from PIL import Image, ImageDraw, ImageFont

from panda_assembly.config import load_config
from panda_assembly.contact_search import SpiralResidualPolicy
from panda_assembly.residual_rl import ResidualPegInsertionEnv

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

SEED = 20261119
EPISODE = 8
CHECKPOINT_SHA256 = "c82dfc9cef31b52fb8219e57cb87df156653f2db5a2f5558e2889f7f3efacccd"
WIDTH, HEIGHT, FPS = 1280, 720, 25
PANEL_WIDTH, PANEL_HEIGHT = 590, 320
PANEL_X = (40, 650)
BG = "#111923"
PANEL = "#1b2734"
WHITE = "#f1f5fa"
MUTED = "#afbcc9"
COLORS = ("#70dfc1", "#ffbb76")
FORCE_TRIGGER_N = 5.0
DEPTH_TARGET_MM = 50.0
POLICIES = ("learned", "search-compensated")


@dataclass
class Replay:
    rows: list[dict]
    qpos: np.ndarray
    summary: dict

    @property
    def times(self) -> np.ndarray:
        return np.asarray([row["time_s"] for row in self.rows])


def record(
    env: ResidualPegInsertionEnv,
    learned,
    search: SpiralResidualPolicy | None,
    *,
    reset_seed: int = SEED + EPISODE,
    trigger_diagnostics: bool = False,
) -> Replay:
    observation, info = env.reset(seed=reset_seed)
    if search is not None:
        search.reset_episode()
    rows: list[dict] = []
    states = []
    trigger_s = None
    finished = False
    while True:
        pose = env.last_pose_estimate
        assert pose is not None
        time_s = float(info["time_s"])
        if not finished:
            if search is None:
                action = learned(observation)
            else:
                action = search.act_with_pose(
                    observation, pose, time_s=time_s, tip_z_m=env.last_observed_tip_z_m
                )
                if trigger_s is None and search.episode_metrics()["search_triggered"]:
                    trigger_s = time_s
        states.append(env.data.qpos.copy())
        rows.append(
            {
                "time_s": time_s,
                "contact_force_n": info["contact_force_n"],
                "running_peak_force_n": info["max_contact_force_n"],
                "depth_mm": 1000.0 * info["insertion_depth_m"],
                "lateral_error_mm": 1000.0 * info["xy_error_m"],
                "pose_age_ms": 1000.0 * pose.age_s,
                "search_triggered": trigger_s is not None,
            }
        )
        if trigger_diagnostics:
            rows[-1].update(
                tip_z_mm=1000.0 * env.last_observed_tip_z_m,
                remaining_s=max(0.0, env.rl.episode_duration_s - time_s),
                trigger_decision=search.trigger_decision if search is not None else "off",
            )
        if finished:
            break
        observation, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        finished = terminated or truncated
    summary = {
        "success": info["is_success"],
        "safety_violation": info["safety_violation"],
        "completion_time_s": info["completion_time_s"],
        "peak_force_n": info["max_contact_force_n"],
        "final_depth_mm": float(rows[-1]["depth_mm"]),
        "trigger_time_s": trigger_s,
        "end_time_s": info["time_s"],
        "disturbance": {
            key: info[key]
            for key in ("lateral_offset_x", "lateral_offset_y", "friction_scale", "tool_mass_scale")
        },
    }
    return Replay(rows, np.asarray(states), summary)


def verify(
    replay: Replay, path: Path, *, episode: int = EPISODE, strict_metrics: bool = True
) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        archived = next(row for row in csv.DictReader(handle) if int(row["episode"]) == episode)
    result = replay.summary
    flags = {
        "success": result["success"],
        "safety_violation": result["safety_violation"],
        "search_triggered": result["trigger_time_s"] is not None,
    }
    for key, value in flags.items():
        if (archived[key] == "True") != value:
            raise ValueError(f"{path} episode {episode}: replay differs on {key}")
    for key, value in result["disturbance"].items():
        if not math.isclose(float(archived[key]), value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{path} episode {episode}: replay disturbance differs on {key}")
    metrics = {
        "max_contact_force_n": result["peak_force_n"],
        "final_insertion_depth_m": result["final_depth_mm"] / 1000.0,
    }
    if result["success"]:
        metrics["completion_time_s"] = result["completion_time_s"]
    for key, value in metrics.items():
        tolerance = 0.0021 if key == "completion_time_s" else 1e-6
        if strict_metrics and not math.isclose(
            float(archived[key]), value, rel_tol=1e-5, abs_tol=tolerance
        ):
            raise ValueError(f"{path} episode {episode}: replay differs on {key}")
    return {key: value - float(archived[key]) for key, value in metrics.items()}


@lru_cache(maxsize=24)
def font(size: int, *, bold: bool = False):
    face = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(face, size)


def label(draw, x, y, value, size=18, color=WHITE, *, bold=False):
    draw.text((x, y), value, fill=color, font=font(size, bold=bold))


def plot(traces: tuple[Replay, Replay], end_s: float, trigger_s: float):
    figure, axes = plt.subplots(1, 2, figsize=(12, 1.6), dpi=100, facecolor=BG)
    figure.subplots_adjust(left=0.046, right=0.98, bottom=0.28, top=0.82, wspace=0.19)
    for axis, key, title, threshold in zip(
        axes,
        ("contact_force_n", "depth_mm"),
        ("Contact force (N) · sampled at 50 Hz", "Insertion depth (mm)"),
        (FORCE_TRIGGER_N, DEPTH_TARGET_MM),
        strict=True,
    ):
        axis.set_facecolor(PANEL)
        for trace, color in zip(traces, COLORS, strict=True):
            axis.plot(trace.times, [row[key] for row in trace.rows], color=color, lw=1.25)
        axis.axhline(threshold, color=MUTED, ls=":", lw=0.8)
        axis.axvline(trigger_s, color=COLORS[1], ls="--", lw=0.9)
        axis.set_title(title, color=WHITE, fontsize=11, loc="left", pad=7)
        axis.set_xlim(0, end_s)
        if key == "depth_mm":
            axis.set_ylim(bottom=-20)  # Focus on the socket mouth and insertion travel.
        axis.set_xlabel("Simulation time (s)", color=MUTED, fontsize=9)
        axis.tick_params(colors=MUTED, labelsize=9, length=2)
        axis.grid(alpha=0.14)
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.canvas.draw()
    bounds = [axis.bbox.extents for axis in axes]
    plot_image = Image.fromarray(np.asarray(figure.canvas.buffer_rgba()).copy()).convert("RGB")
    plt.close(figure)
    return plot_image, bounds


def render(env: ResidualPegInsertionEnv, traces: tuple[Replay, Replay], out: Path) -> None:
    times = [trace.times for trace in traces]
    end_s = float(max(t[-1] for t in times))
    trigger_s = traces[1].summary["trigger_time_s"]
    assert trigger_s is not None
    chart, bounds = plot(traces, end_s, trigger_s)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 0.67
    camera.azimuth = 140.0
    camera.elevation = -21.0
    camera.lookat[:] = env.hole_position + np.array([-0.045, 0.0, 0.095])
    renderer = mujoco.Renderer(env.model, height=PANEL_HEIGHT, width=PANEL_WIDTH)

    @lru_cache(maxsize=8)
    def scene(side: int, index: int) -> Image.Image:
        env.data.qpos[:] = traces[side].qpos[index]
        mujoco.mj_forward(env.model, env.data)
        renderer.update_scene(env.data, camera=camera)
        return Image.fromarray(renderer.render())

    def frame(time_s: float, chapter: str, caption: str, *, show_outcome: bool = False):
        image = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(image)
        label(draw, 40, 18, "FRANKA PANDA  /  MUJOCO SIMULATION", 15, MUTED, bold=True)
        label(draw, 40, 42, "One insertion. Two recovery decisions.", 34, bold=True)
        draw.rounded_rectangle((951, 41, 1240, 81), 10, fill=PANEL)
        label(draw, 967, 51, f"{chapter}   |   t = {time_s:05.2f} s", 16, WHITE, bold=True)
        for side, (trace, x, color) in enumerate(zip(traces, PANEL_X, COLORS, strict=True)):
            index = max(0, int(np.searchsorted(times[side], time_s, side="right") - 1))
            image.paste(scene(side, index), (x, 130))
            draw.rectangle((x, 94, x + PANEL_WIDTH, 130), fill=PANEL)
            draw.rectangle((x, 94, x + 4, 130), fill=color)
            heading = ("A  ·  SEARCH OFF", "B  ·  COMPENSATED SEARCH")[side]
            label(draw, x + 15, 101, heading, 19, color, bold=True)
            row = trace.rows[index]
            draw.rectangle((x, 450, x + PANEL_WIDTH, 484), fill=PANEL)
            if show_outcome or time_s >= times[side][-1]:
                value = (
                    f"SUCCESS at {trace.summary['completion_time_s']:.2f} s"
                    if trace.summary["success"]
                    else f"TIMEOUT · final depth {trace.summary['final_depth_mm']:.1f} mm"
                )
            else:
                state = "SEARCH TRIGGERED" if row["search_triggered"] else "TRACKING"
                value = (
                    f"{state}   |   {row['contact_force_n']:.1f} N   |   {row['depth_mm']:.1f} mm"
                )
            label(draw, x + 15, 456, value, 16, color, bold=True)
        chart_y = 490
        image.paste(chart, (40, chart_y))
        for left, bottom, right, top in bounds:
            cursor_x = 40 + left + (right - left) * time_s / end_s
            draw.line(
                (cursor_x, chart_y + chart.height - top, cursor_x, chart_y + chart.height - bottom),
                fill=WHITE,
                width=2,
            )
        draw.rounded_rectangle((40, 657, 1240, 705), 10, fill=PANEL)
        label(draw, 56, 670, caption, 18)
        return image

    detail_start = max(0.0, trigger_s - 1.0)
    detail_end = min(end_s, max(trigger_s + 3.0, float(times[0][-1]) + 0.5))
    chapters = (
        (7, 0.0, 0.0, "Setup", "Same checkpoint and disturbance · 3 mm radial clearance."),
        (
            24,
            0.0,
            end_s,
            "Full replay",
            "25 Hz pose estimates · 120 ms delay · 30% frame loss · correlated noise.",
        ),
        (
            19,
            detail_start,
            detail_end,
            "Slow replay",
            f"Dashed orange marker: compensated search triggers at {trigger_s:.2f} s.",
        ),
        (
            12,
            end_s,
            end_s,
            "Result",
            "Same peak force: 58.0 N. Search off inserted successfully; the search run timed out.",
        ),
    )
    ffmpeg = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out / "paired-recovery.mp4"),
    ]
    try:
        poster = frame(end_s, "Result", chapters[-1][-1], show_outcome=True)
        poster.save(out / "paired-recovery-poster.png")
        with subprocess.Popen(ffmpeg, stdin=subprocess.PIPE) as encoder:
            assert encoder.stdin is not None
            for duration, start, end, chapter, caption in chapters:
                if start == end:
                    pixels = frame(start, chapter, caption).tobytes()
                    for _ in range(duration * FPS):
                        encoder.stdin.write(pixels)
                else:
                    for time_s in np.linspace(start, end, duration * FPS):
                        encoder.stdin.write(frame(float(time_s), chapter, caption).tobytes())
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("ffmpeg failed to encode the replay")
    finally:
        renderer.close()


def replay_environment() -> ResidualPegInsertionEnv:
    config = load_config(Path("configs/menagerie.toml"))
    config = replace(
        config,
        peg_task=replace(config.peg_task, insert_time=12.0),
        residual_rl=replace(config.residual_rl, episode_duration_s=18.0),
    )
    estimator = next(config for key, _, config in CONDITIONS if key == "camera-combined")
    return ResidualPegInsertionEnv(
        config, pose_estimator_config=estimator, socket_clearance_m=0.003
    )


def record_replays(env: ResidualPegInsertionEnv, args: argparse.Namespace) -> None:
    from stable_baselines3 import SAC

    if hashlib.sha256(args.model.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("checkpoint does not match the archived 20261119 SAC model")
    args.output.mkdir(parents=True, exist_ok=True)
    config = env.config
    model = SAC.load(args.model, device="cpu")

    def learned(obs):
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    search = SpiralResidualPolicy(
        learned,
        max_position_offset_m=np.asarray(config.residual_rl.max_position_offset_m),
        episode_duration_s=config.residual_rl.episode_duration_s,
        force_scale_n=config.force_safety.force_limit_n,
        trigger_force_n=FORCE_TRIGGER_N,
        search_radius_m=0.007,
        entry_depth_m=config.contact_recovery.entry_depth_m,
        entry_mode="compensated",
        max_pose_age_s=MAX_RECOVERY_POSE_AGE_S,
    )
    traces = (record(env, learned, None), record(env, learned, search))
    for policy, trace in zip(POLICIES, traces, strict=True):
        verify(trace, args.reference / policy / "episodes.csv")
        with (args.output / f"{policy}-timeseries.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trace.rows[0]))
            writer.writeheader()
            writer.writerows(trace.rows)
    np.savez_compressed(
        args.output / "joint-states.npz",
        **dict(zip(POLICIES, [trace.qpos for trace in traces], strict=True)),
    )
    result = {
        "condition": "camera-combined",
        "checkpoint_seed": SEED,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "episode": EPISODE,
        "reset_seed": SEED + EPISODE,
        "benchmark_run": 45476321699,
        "search_off": traces[0].summary,
        "compensated_search": traces[1].summary,
    }
    (args.output / "replay-summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


def load_replay(directory: Path, policy: str, summary: dict) -> Replay:
    with (directory / f"{policy}-timeseries.csv").open(newline="", encoding="utf-8") as handle:
        rows = [
            {
                key: val == "True" if key == "search_triggered" else float(val)
                for key, val in row.items()
            }
            for row in csv.DictReader(handle)
        ]
    with np.load(directory / "joint-states.npz", allow_pickle=False) as states:
        return Replay(rows, states[policy], summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record_parser = commands.add_parser("record")
    record_parser.add_argument("--model", type=Path, required=True)
    record_parser.add_argument("--reference", type=Path, required=True)
    record_parser.add_argument("--output", type=Path, required=True)
    render_parser = commands.add_parser("render")
    render_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    env = replay_environment()
    try:
        if args.command == "record":
            record_replays(env, args)
        else:
            result = json.loads((args.output / "replay-summary.json").read_text(encoding="utf-8"))
            traces = (
                load_replay(args.output, POLICIES[0], result["search_off"]),
                load_replay(args.output, POLICIES[1], result["compensated_search"]),
            )
            env.reset(seed=SEED + EPISODE)
            render(env, traces, args.output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
