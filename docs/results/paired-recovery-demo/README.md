# Paired Panda recovery replay

[54-second presentation cut](presentation/paired-recovery-presentation.mp4) ·
[Timed Chinese and English speaker notes](presentation/speaker-notes.md) ·
[Presentation cover](presentation/presentation-cover.png)

The presentation cut uses the verified replay and CSV traces. With FFmpeg on the path
and the project plotting dependencies installed, run
`python scripts/edit_paired_recovery_presentation.py --output runs/presentation`
from the repository root.

This 62-second MuJoCo replay compares the same SAC checkpoint and the same disturbance with
search off (left) and compensated search enabled (right). It shows one counterexample from the
[four-policy recovery benchmark](../pose-aware-recovery/recovery-comparison.json).

- **Setup, 0–7 s** — The tested socket has 3 mm radial clearance. The simulated relative-pose input updates at 25 Hz with 120 ms delay, 30% configured frame loss and correlated error.
- **Full replay, 7–31 s** — Both views share a simulation clock. A finished trial holds its last frame while the other trial runs.
- **Slow replay, 31–50 s** — The orange dashed line marks the recovery trigger at 10.56 s of simulation time. Force is sampled at the 50 Hz control rate. The depth plot focuses on the last 20 mm above the opening and the insertion range.
- **Result, 50–62 s** — Search off completes in 11.892 s at 57.8 mm depth. Compensated search times out at 18 s at 0.1 mm depth. Both simulations record a 57.99 N peak at the physics step rate.

[MP4](paired-recovery.mp4) · [Still frame](paired-recovery-poster.png) ·
[Replay summary](replay-summary.json) · [Search-off trace](learned-timeseries.csv) ·
[Compensated-search trace](search-compensated-timeseries.csv)

Replay identity is checkpoint seed `20261119`, episode index `8`, environment reset seed
`20261127`, and condition `camera-combined`. The render workflow checks the original checkpoint
hash and compares both reruns against the benchmark archive before producing the video.
It stores the [joint state samples](joint-states.npz) alongside the time series for frame replay.
[Verified cloud render](https://github.com/haoyu020206-netizen/panda-contact-assembly/actions/runs/45485892239).
