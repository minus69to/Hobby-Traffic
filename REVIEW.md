# REVIEW (final self-review, M9)

_Written at the end of the run (2026-11-05, after the VM restart described in PROGRESS.md)._

## What is strong

- **One control path, one shield.** Fixed-time, max-pressure, PPO and the deployed MQTT service all
  go through `SafetyShield`, which also produces the action mask. Safety does not depend on the
  policy being good, and the property-based tests (`tests/test_shield.py`) exercise random action
  sequences and random timing parameters.
- **Honest numbers end to end.** README, report and website are rendered/built from `results/`;
  `scripts/check_numbers.py` fails if a document or the web bundle drifts from the results files.
  Confidence intervals are paired bootstraps over shared evaluation seeds; shield interventions are
  reported even when zero.
- **The simulation is deliberately hard.** Sublane weaving, six classes including 12 km/h rickshaws,
  signalised pedestrian crossings with real interaction, an incident and sensor noise.
- **Reproducibility.** Plain-XML networks built by netconvert, seeded demand, YAML configs, disjoint
  seed sets, `make reproduce` for a 15-minute end-to-end run, CI with a smoke training.
- **Deployment story is real code, not slides.** ONNX export with parity and quantisation numbers,
  pydantic-versioned MQTT schemas, watchdog fallback, Prometheus sidecar, SUMO bridge, integration test.
- **The visualisation is built from simulation data.** Road geometry comes from the SUMO network,
  replays are compact binaries decoded in the browser, the HUD reads the replay manifest.

## What a hiring manager or reviewer would question

- **The headline result is negative.** With two seeds and 300k steps on CPU the safe PPO policy does
  not beat fixed-time on vehicle delay and is far behind max-pressure; the only consistent gain is
  pedestrian waiting time. The repository reports this as is. A reviewer will rightly ask whether the
  reward, the budget or the observation is the limiting factor; the imitation warm start (appendix A.2)
  is the only ablation that ran.
- **Only two seeds; the planned third seed did not train** (crash after a code edit, then a VM
  restart); the grid and forecasting components are implemented but were never executed.
- **Oversaturated demand.** Roughly a third of the peak-hour vehicles never complete their trip under
  any controller, so waiting-time means of completed trips are selective; throughput and insertion
  delay are reported next to them, but the scenario design could be criticised as too heavy.

- **Synthetic everything.** No real counts, no calibration, no field data. The project says so in
  every artefact, but the numbers are only as meaningful as the assumed demand.
- **Max-pressure baseline weakness.** The textbook arg-max never serves the shared through+right
  lane's right-turn phase voluntarily (see `DECISIONS.md` 8), so it looks weak off-peak. A reviewer
  may ask for a lane-blocking-aware variant; several were tried without a dominant winner.
- **Training budget.** 300k steps per seed on CPU is small for PPO in traffic control; the learning
  curves are noisy and a longer run would likely change the ranking vs max-pressure at the peak.
- **Reward shaping.** The waiting-time difference reward telescopes; weights were chosen by scale
  reasoning, not by an ablation.
- **Pedestrian modelling.** Striping model only, with the jam-time setting needed to avoid deadlock.
- **Replays use a 0.5 s step, evaluation 1 s.** HUD numbers come from the replay run; the tables
  from the evaluation runs. Both are stated, but it is a second source of numbers.
- **Grid experiments** are secondary and, if present, are a single seed.

## Adversarial review outcome

A workflow of 7 reviewers (one per subsystem) and 3 refuters per finding ran before the final
evaluation: 39 findings, 31 confirmed and fixed (see the commit "Apply 31 confirmed review
findings"). Refuted and therefore left as is: mask/shield disagreement at max-green (deliberate: the
shield also forces the change), forced max-green ignoring the walk minimum (only reachable with a
silent controller after 60 s of green), the non-atomic route-file write (workers generate identical
files), 1e-9 division floor in the paired bootstrap (the baseline mean is never that small for
reported metrics), the `independent` grid ablation naming (documented as "no neighbour information"),
GNN ONNX export (not needed; the grid runs are single-seed feasibility checks), and two cosmetic web
items. Two confirmed items could not be applied to runs already in progress, see `DECISIONS.md` 11.

## Known bugs and rough edges

- Bloom (post-processing) is disabled in split view because `EffectComposer` cannot be scissored per
  pane in `@react-three/drei`'s `View`.
- Headless SwiftShader rendering is slow; the media capture renders at reduced frame rates.
- `libsumo` allows only one simulation per process, so every worker is a separate process and the
  `grid2x2` training uses the custom `MultiJunctionVecEnv` instead of SB3's `SubprocVecEnv`.
- The Docker files were not executed in the development VM (no daemon).

## Next steps

1. Calibrate demand and vehicle parameters against real counts for one Dhaka junction.
2. Longer training with several reward variants (pure pressure, class-weighted waiting) and a proper
   hyper-parameter sweep; report learning curves with multiple seeds for every variant.
3. Lane-blocking-aware max-pressure and SUMO's actuated controller as additional baselines.
4. Corridor coordination on the grid with the parameter-shared and graph policies.
5. A hardware-in-the-loop test of the MQTT service on a small ARM board.

## Status checklist (filled at M9)

- [x] `make lint test` passes (ruff, mypy, 32 unit tests; slow and integration tests also green)
- [x] `make reproduce` runs end to end (small budget, exit 0)
- [x] `npm run build` and the Playwright smoke test pass with no console errors
- [x] `scripts/check_numbers.py` passes (README, report and web bundle vs results/)
- [x] property-based tests rerun with `HYPOTHESIS_PROFILE=thorough` (2,000 examples): zero violations
- [x] repository searched for anything resembling proprietary deployment material: nothing found
- [ ] `ALL DELIVERABLES PUSHED` written to PROGRESS.md (last step)
