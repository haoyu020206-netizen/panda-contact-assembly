"""Re-cut the verified replay for a timed live explanation, using its logged telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

WIDTH, HEIGHT, FPS = 1280, 720, 25
SCENE_WIDTH, SCENE_HEIGHT = 1200, 320
SCENE_BYTES = SCENE_WIDTH * SCENE_HEIGHT * 3
SOURCE_FRAMES = 24 * FPS
BACKGROUND, SURFACE = "#101922", "#1c2b39"
WHITE, MUTED = "#f3f7fa", "#acbfcd"
COLORS = ("#73e4c4", "#ffbc83")
START_S = 9.0


@lru_cache(maxsize=20)
def font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
    style = font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "regular")
    return ImageFont.truetype(font_manager.findfont(style), size)


def label(draw, x, y, message, size, color=WHITE, *, bold=False) -> None:
    draw.text((x, y), message, font=font(size, bold), fill=color)


def read_trace(path: Path) -> dict[str, np.ndarray]:
    fields = ("time_s", "depth_mm", "contact_force_n")
    with path.open(newline="", encoding="utf-8") as handle:
        values = [[float(row[key]) for key in fields] for row in csv.DictReader(handle)]
    return dict(zip(fields, np.asarray(values).T, strict=True))


def make_chart(traces, trigger_s, success_s, end_s):
    figure, axes = plt.subplots(1, 2, figsize=(12, 1.62), dpi=100, facecolor=BACKGROUND)
    figure.subplots_adjust(left=0.04, right=0.985, bottom=0.24, top=0.81, wspace=0.19)
    presentations = (
        ("depth_mm", "Insertion depth near socket  /  mm", 50, (-15, 65)),
        ("contact_force_n", "Contact force  /  N  ·  50 Hz samples", 5, (0, None)),
    )
    for axis, (field, title, threshold, limits) in zip(axes, presentations, strict=True):
        axis.set_facecolor(SURFACE)
        for trace, color in zip(traces, COLORS, strict=True):
            axis.plot(trace["time_s"], trace[field], lw=1.5, color=color)
        axis.axhline(threshold, color=MUTED, ls=":", alpha=0.65, lw=0.9)
        axis.axvline(trigger_s, color=COLORS[1], ls="--", lw=1.25)
        axis.axvline(success_s, color=COLORS[0], ls="--", lw=1.25)
        axis.set_title(title, loc="left", color=WHITE, fontsize=12, pad=7)
        axis.set_xlim(START_S, end_s)
        axis.set_ylim(*limits)
        axis.set_xlabel("Simulation time (s)", color=MUTED, fontsize=9, labelpad=1)
        axis.tick_params(colors=MUTED, labelsize=9, length=2)
        axis.grid(alpha=0.14)
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.canvas.draw()
    bounds = [tuple(axis.bbox.extents) for axis in axes]
    image = Image.fromarray(np.array(figure.canvas.buffer_rgba())).convert("RGB")
    plt.close(figure)
    return image, bounds


def scene_command(executable, source, start, frames):
    return [
        str(executable),
        "-v",
        "error",
        "-ss",
        str(start),
        "-i",
        str(source),
        "-frames:v",
        str(frames),
        "-vf",
        f"crop={SCENE_WIDTH}:{SCENE_HEIGHT}:40:130",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]


def render(example: Path, output: Path, ffmpeg: str) -> None:
    summary = json.loads((example / "replay-summary.json").read_text(encoding="utf-8"))
    off, on = (summary[key] for key in ("search_off", "compensated_search"))
    trigger_s = round(on["trigger_time_s"], 3)
    completion_s = round(off["completion_time_s"], 3)
    end_s = round(on["end_time_s"], 3)
    bench_file = example.parent / "pose-aware-recovery" / "recovery-comparison.json"
    comparison = json.loads(bench_file.read_text(encoding="utf-8"))
    policies = comparison["conditions"]["camera-combined"]["policies"]
    successes = tuple(
        f"{policies[key]['successes']}/{policies[key]['episodes']}"
        for key in ("learned", "search-compensated")
    )
    filenames = ("learned-timeseries.csv", "search-compensated-timeseries.csv")
    traces = tuple(read_trace(example / name) for name in filenames)
    chart, chart_bounds = make_chart(traces, trigger_s, completion_s, end_s)
    source = example / "paired-recovery.mp4"
    last_scene = subprocess.check_output(scene_command(ffmpeg, source, 54, 1))
    if len(last_scene) != SCENE_BYTES:
        raise ValueError("could not extract the final source frame")

    def compose(scene, time_s, section, headline, caption, *, highlight=False):
        image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(image)
        label(draw, 40, 14, f"FRANKA PANDA  /  MUJOCO  /  {section}", 15, MUTED, bold=True)
        label(draw, 40, 37, headline, 32, bold=True)
        draw.rounded_rectangle((1004, 37, 1240, 77), 9, fill=SURFACE)
        label(draw, 1022, 47, f"SIM TIME  {time_s:05.2f} s", 17, bold=True)
        image.paste(Image.frombytes("RGB", (SCENE_WIDTH, SCENE_HEIGHT), scene), (40, 123))
        for index, (x, color) in enumerate(zip((40, 650), COLORS, strict=True)):
            draw.rectangle((x, 88, x + 590, 123), fill=SURFACE)
            draw.rectangle((x, 88, x + 4, 123), fill=color)
            policy = ("A   SEARCH OFF", "B   COMPENSATED SEARCH")[index]
            label(draw, x + 15, 95, policy, 19, color, bold=True)
            draw.rectangle((x, 443, x + 590, 480), fill=SURFACE)
            if index == 0 and time_s >= completion_s:
                status = f"SUCCESS  ·  {completion_s:.2f} s  ·  {off['final_depth_mm']:.1f} mm"
            elif index == 1 and time_s >= end_s:
                status = f"TIMEOUT  ·  {end_s:.2f} s  ·  {on['final_depth_mm']:.1f} mm"
            else:
                trace = traces[index]
                row = int(np.searchsorted(trace["time_s"], time_s, side="right") - 1)
                state = "SEARCH ACTIVE" if index == 1 and time_s >= trigger_s else "TRACKING"
                status = f"{state}  ·  DEPTH {trace['depth_mm'][row]:.1f} mm"
            label(draw, x + 15, 451, status, 17, color, bold=True)
        chart_y = 483
        image.paste(chart, (40, chart_y))
        if not highlight:
            for left, bottom, right, top in chart_bounds:
                position = (time_s - START_S) / (end_s - START_S)
                cursor = 40 + left + position * (right - left)
                draw.line(
                    (cursor, chart_y + chart.height - top, cursor, chart_y + chart.height - bottom),
                    fill=WHITE,
                    width=2,
                )
        draw.rounded_rectangle((40, 650, 1240, 704), 10, fill=SURFACE)
        label(draw, 57, 665, caption, 20, bold=True)
        return image

    peak = f"{off['peak_force_n']:.1f} N"
    chapters = (
        (
            8,
            end_s,
            end_s,
            "RESULT PREVIEW",
            "Same disturbance. Two outcomes.",
            "Same SAC checkpoint and disturbance. Only spiral recovery changes.",
        ),
        (
            8,
            9.2,
            trigger_s,
            "SETUP",
            "Delayed pose and a narrow socket",
            "3 mm radial clearance  ·  120 ms pose delay  ·  30% frame loss  ·  correlated error",
        ),
        (
            18,
            trigger_s,
            12.2,
            "SLOW MOTION",
            f"Search triggers at {trigger_s:.2f} s",
            "Orange dashed line: search starts   |   Mint dashed line: insertion succeeds",
        ),
        (
            9,
            12.2,
            end_s,
            "OUTCOME",
            "The search run stays near the socket opening",
            f"Both runs record a {peak} peak at the physics step rate.",
        ),
        (
            11,
            end_s,
            end_s,
            "BENCHMARK",
            "When should the robot start searching?",
            f"Combined disturbance  ·  {successes[0]} succeed without search  ·  "
            f"{successes[1]} with compensated search",
        ),
    )
    output.mkdir(parents=True, exist_ok=True)
    movie = output / "paired-recovery-presentation.mp4"
    if movie.exists():
        raise FileExistsError(movie)
    compose(last_scene, end_s, *chapters[0][3:], highlight=True).save(
        output / "presentation-cover.png"
    )
    encoding = [
        ffmpeg,
        "-v",
        "error",
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
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(movie),
    ]
    with (
        subprocess.Popen(
            scene_command(ffmpeg, source, 7, SOURCE_FRAMES), stdout=subprocess.PIPE
        ) as decoder,
        subprocess.Popen(encoding, stdin=subprocess.PIPE) as encoder,
    ):
        assert decoder.stdout is not None and encoder.stdin is not None
        index, scene = -1, b""
        for seconds, start, end, section, headline, caption in chapters:
            if start == end:
                still = compose(last_scene, start, section, headline, caption, highlight=True)
                pixels = still.tobytes()
                for _ in range(seconds * FPS):
                    encoder.stdin.write(pixels)
                continue
            for time_s in np.linspace(start, end, seconds * FPS, endpoint=False):
                frame = round(float(time_s) / end_s * (SOURCE_FRAMES - 1))
                while index < frame:
                    scene = decoder.stdout.read(SCENE_BYTES)
                    if len(scene) != SCENE_BYTES:
                        raise ValueError("source ended before the requested frame")
                    index += 1
                image = compose(scene, float(time_s), section, headline, caption)
                encoder.stdin.write(image.tobytes())
        # Drain the last source frames before checking both processes.
        decoder.stdout.read()
        encoder.stdin.close()
        if decoder.wait() or encoder.wait():
            raise RuntimeError("ffmpeg failed to create the presentation cut")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, default=Path("docs/results/paired-recovery-demo"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    render(args.example, args.output, args.ffmpeg)


if __name__ == "__main__":
    main()
