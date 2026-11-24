# Held-out evaluation of the stall trigger

The [protocol](protocol.json) was committed before any holdout episode ran. It freezes the
second-stage candidate at 5 N sustained contact, 0.6 s without 1 mm downward tip travel,
the 160 ms pose-age limit and the 5 s recovery time budget.

The evaluator loads the three existing SAC checkpoints, verifies their SHA-256 hashes
and assigns 20 separate new reset seeds to each. Each of the 60 disturbance vectors is
reused across the three sensing conditions and the two policies, for 180 matched
comparisons and 360 policy episodes. The pairing check verifies fixture offset,
friction and mass in every comparison. The 3 mm radial clearance and simulator setup
follow the development experiment.

The primary condition combines 120 ms latency, 30% configured frame dropout and
correlated position error. Acceptance requires a net rescue in this condition,
nonnegative net rescues in the ideal-pose and 120 ms latency conditions, and zero new
paired force-limit terminations across all conditions. A rescue is a candidate success
on a case where search off failed. A harm is the reverse outcome. Peak force is reported
as a performance metric.

| Sensing condition | Search off | Stall trigger | Search interventions | Rescued / harmed | Force-limit terminations, off / trigger | Mean peak force, off / trigger |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ideal pose | 52/60 | 52/60 | 1 | 0 / 0 | 0 / 0 | 15.0 / 15.0 N |
| 120 ms latency | 49/60 | 49/60 | 10 | 0 / 0 | 1 / 1 | 49.1 / 49.1 N |
| Combined disturbance | 46/60 | 46/60 | 7 | 0 / 0 | 4 / 4 | 68.9 / 68.9 N |

Every matched pair had the same success and safety termination flags. No episode showed
a peak-force increase with the stall trigger. The 18 interventions changed the final
residual offset in their paired replays, and all 18 ended unsuccessfully. In the primary
condition the result was zero net rescues. The prespecified decision therefore selects
search off. The optional candidate remains available for subsequent recovery experiments.

[Summary](stall-trigger-holdout.json) · [180 paired records](paired-episodes.csv) ·
[Cloud run and raw per-policy records](https://github.com/haoyu020206-netizen/panda-contact-assembly/actions/runs/45496804966).
Local aggregation of the downloads reproduced the cloud JSON and CSV after line-ending
normalization.
