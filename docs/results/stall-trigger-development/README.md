# Spiral recovery trigger development

The development set is the 20 archived combined-disturbance episodes for checkpoint
`20261119`. The replay checks both baseline policies' outcome flags and disturbances
against the archive. It records deviations in completion time, peak force and final depth.
All three policies use the same checkpoint, simulator settings, initial disturbances
and pose-estimator configuration.

The candidate uses the existing compensated spiral wrapper and changes when recovery can
start. Contact force must remain at or above 5 N for 0.6 s without at least 1 mm of downward
tip travel, measured from robot kinematics. Releasing contact or advancing resets the
window. Triggering also requires a usable compensated depth below the 4 mm entry threshold
and at least 5 s remaining. This time allowance covers the 0.5 s unload phase, the 4 s
spiral radius ramp and a 0.5 s retry margin.

The replay logs control-rate force, true simulated depth, tip height, lateral error,
pose estimate age, time remaining and the candidate's trigger decision. True simulated
depth is used for diagnosis. The trigger uses the timestamped estimate and the robot
tip kinematics.

| Policy | Success | Search triggers | Rescued / harmed vs search off | Safety terminations | Mean episode peak force |
| --- | ---: | ---: | ---: | ---: | ---: |
| Search off | 14/20 | 0 | 0 / 0 | 1 | 63.7 N |
| Compensated search, immediate trigger | 8/20 | 11 | 0 / 6 | 1 | 62.0 N |
| Compensated search, stall trigger | 14/20 | 5 | 0 / 0 | 1 | 63.7 N |

In the six harmed episodes, the immediate trigger fired with 7.40–7.46 s left.
In the matched search-off replay, depth advanced 6.1–29.2 mm from that instant over
the next 0.6 s. For the five triggered episodes that failed with both baselines, the
same measure was 0.04–0.41 mm. The stall trigger intervened in those five episodes
and all five still failed. The remaining paired failure ended at the force limit
before search started. None of the policies produced an extra safety termination
relative to search off.

On this development set, the gate prevented interruptions that the immediate trigger
caused. Search off and the stall trigger produced the same episode success outcomes.
These cases were selected from a single existing checkpoint and condition. The candidate
is available through the optional `stall_trigger` argument and awaits evaluation on
separate disturbances before any default change.

[Per-episode results](stall-trigger-development.json) ·
[Cloud run and 60 control-rate traces](https://github.com/haoyu020206-netizen/panda-contact-assembly/actions/runs/45490241913).
The trigger-context fields in the committed summary were recomputed from those traces
with a 1 ns tolerance at the 0.6 s window boundary.
