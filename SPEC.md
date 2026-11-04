You are an autonomous senior ML engineer, simulation engineer and creative technologist. The repository you are in is `dhaka-rl-signals`. The owner is asleep and will not answer questions. Read this whole specification first, save it verbatim as `SPEC.md` in the repo, commit it, and then execute it end to end.

# 0. Operating rules (read twice)

1. Autonomy. Never stop to ask a question. When something is ambiguous, choose the most defensible option, record the choice and reason in `DECISIONS.md`, and continue.
2. Time. Current time is about 2026-11-04 16:00 UTC. Everything important must be committed and pushed by **2026-11-05 07:30 UTC**. Follow the milestone timeboxes. If a milestone overruns by more than 30 minutes, apply its fallback, log it, and move on.
3. Priority if time runs short (protect in this order): (1) simulation plus trained safe RL controller plus evaluation numbers, (2) the 3D web replay with RL vs fixed-time split view and the landing page, (3) README with hero GIF, video and results, (4) edge export and MQTT controller service, (5) technical report, (6) stretch goals.
4. Persistence. Commit and push to `main` at least every 45 minutes and at the end of each milestone. Keep `PROGRESS.md` as a timestamped (UTC) log. If interrupted, the next instruction will be "Read SPEC.md and PROGRESS.md and continue"; make that possible.
5. Long jobs. Training, evaluation and rendering run in the background with logs in `logs/`, checkpoints, and polling. Never sit idle: while training runs, build the visualization and the service.
6. Honesty. Never invent numbers. Every number shown in README, report or website must be read from files in `results/` produced by code in this repo (the website imports them at build time). Simulation demand and vehicle parameters are synthetic assumptions inspired by Dhaka traffic; say so everywhere. Never claim real-world validation.
7. Independence. Build everything from scratch with public tools. Do not use, reproduce or imitate any proprietary code, data, protocol, message format or configuration from any real traffic-signal deployment, including any project the owner works on.
8. Compute. CPU only: 4 vCPU, 16 GB RAM, no GPU, about 30 GB disk. Internet available for pip, npm, apt and GitHub. No paid APIs. Keep the git repo small (under about 150 MB): ignore model checkpoints except the final small policy files, and compress replay data.
9. Quality bar. Interview-grade code: `src/` package, type hints, docstrings, `ruff` and `mypy`-clean Python, ESLint and TypeScript strict on the web app, tests, configs in YAML, deterministic seeds, versions recorded in `ENVIRONMENT.md`.

# 1. Product vision

A visitor opens a link and sees a living, stylised 3D Dhaka intersection at golden hour: buses, cars, motorcycles weaving between lanes, green CNG auto-rickshaws and colourful cycle-rickshaws, pedestrians on zebra crossings, signal lights changing. A split screen shows the same rush hour under a fixed-time signal on the left and the safe RL controller on the right, with live counters showing queues shrinking on the right. Scrolling down tells the story: the problem, how the RL agent works, how safety is guaranteed, results charts, and edge-deployment numbers. The repo behind it is clean, tested and reproducible.

# 2. Milestones with timeboxes (UTC)

| ID | Box | Milestone |
|---|---|---|
| M0 | 16:00 to 16:40 | Scaffold, environment, SUMO speed test |
| M1 | 16:40 to 18:30 | Networks, Dhaka mixed-traffic demand, baselines |
| M2 | 18:30 to 19:30 | RL environment, safety shield, tests |
| M3 | 19:30 to 03:00 (background) | PPO training, single intersection then grid |
| M4 | 19:30 to 01:30 (parallel) | 3D web app and landing page |
| M5 | 23:00 to 02:30 (parallel) | Edge export, MQTT controller service, Docker |
| M6 | 03:00 to 04:30 | Evaluation, robustness, replay data export |
| M7 | 04:30 to 05:45 | Media capture (GIF, video, screenshots), Pages deploy |
| M8 | 05:00 to 06:45 | README, technical report |
| M9 | 06:45 to 07:30 | Final self-review and push |

# 3. M0: scaffold

```
dhaka-rl-signals/
  SPEC.md PROGRESS.md DECISIONS.md ENVIRONMENT.md README.md LICENSE (MIT)
  Makefile pyproject.toml docker-compose.yml Dockerfile .github/workflows/
  configs/ (sim, train, eval, service yaml)
  sim/networks/ sim/demand/ sim/vtypes/
  src/drs/ (env, shield, controllers, train, eval, export, service, replay)
  tests/ scripts/ results/ figures/ logs/ models/ docs/
  web/ (Vite + React + TypeScript app)
```

- Python deps: eclipse-sumo and libsumo (pip), sumo-rl, gymnasium, stable-baselines3, sb3-contrib (MaskablePPO), torch CPU, onnx, onnxruntime, numpy, pandas, scipy, matplotlib, pyyaml, fastapi, uvicorn, aiomqtt (or paho-mqtt), prometheus-client, pydantic, hypothesis, pytest, ruff, mypy, playwright. System: ffmpeg, mosquitto (apt, or docker image eclipse-mosquitto). Node LTS for web.
- Use libsumo (`LIBSUMO_AS_TRACI=1`) for speed. Measure simulation steps per second for 1 hour of simulated peak traffic on the single intersection and record it in `ENVIRONMENT.md`; size the training budget from it.

# 4. M1: networks and Dhaka-like mixed traffic

## 4.1 Networks (plain SUMO XML: nodes, edges, connections, traffic-light programs, built with netconvert)
- `single`: one 4-arm signalised junction, 3 lanes per approach (one right-turn, one through, one left-turn plus through; Bangladesh drives on the left, so set `--lefthand` in netconvert), approach length 300 m, sidewalks and zebra crossings on all arms.
- `grid2x2`: four such junctions 250 m apart forming a corridor grid.
- Four-phase programs with yellow 3 s and all-red 2 s; pedestrian walk phases tied to the conflicting vehicle phases.

## 4.2 Vehicle types (`sim/vtypes/dhaka.add.xml`) and sublane model
Use the SUMO sublane model (`--lateral-resolution 0.8`) so small vehicles weave and share lanes, the defining feature of Dhaka traffic. Define vTypes with documented, plausible parameters (length, width, maxSpeed, accel, decel, sigma, lcStrategic, lcPushy, latAlignment, minGapLat):
- `car` (private car), `bus` (standard bus, also a red double-decker variant), `truck`, `motorcycle` (high lateral agility), `cng` (auto-rickshaw: short, slow, agile), `rickshaw` (cycle-rickshaw: very slow, about 12 km/h, wide), `pedestrian`.
- Default mix (an assumption; state it): cars 22 percent, motorcycles 25 percent, CNG 20 percent, rickshaws 15 percent, buses 12 percent, trucks 6 percent. Put the mix in config so it can be changed.

## 4.3 Demand
Generate route files with flows per vehicle class and turning ratios, seeded:
- `offpeak`, `morning_peak` (heavier inbound), `evening_peak` (heavier outbound), `incident` (morning peak with one approach lane blocked for 15 minutes), `surge` (+20 percent over peak), and pedestrian flows on all crossings.
- Episode length 3,600 s simulated, warm-up 300 s excluded from metrics.
- Training seeds and evaluation seeds must be disjoint; record them.

## 4.4 Baseline controllers
- `fixed_time`: Webster-style cycle from the configured demand (document the calculation).
- `actuated`: SUMO actuated or a max-pressure controller with the same safety constraints.

Fallback for M1: drop `grid2x2` to M3 stretch and do everything on `single` first.

# 5. M2: RL environment and safety shield

- Gymnasium environment over sumo-rl or a custom TraCI/libsumo wrapper. Decision interval 5 s.
- Observation per junction: normalised queue length and max waiting time per incoming lane, vehicle counts per class per approach (so the agent can see slow rickshaws), current phase one-hot, time in phase, pedestrian waiting counts per crossing; for the grid add neighbour pressure.
- Action: choose next green phase (or keep current).
- Reward: negative weighted sum of total waiting time change and pressure, plus pedestrian waiting penalty and a small switch penalty; weights in config; document reward design in DECISIONS.md.
- **Safety by construction**:
  1. Action masking (sb3-contrib MaskablePPO): a phase change is invalid before `min_green` (10 s), forced after `max_green` (60 s), and impossible during yellow, all-red or a pedestrian walk minimum (8 s).
  2. Runtime safety shield (`src/drs/shield.py`): an explicit finite-state machine that every controller (RL, baselines, the deployed service) must pass through. It inserts yellow and all-red, enforces timers, rejects conflicting greens, and logs every intervention.
  3. Tests: unit tests for each rule, plus property-based tests with `hypothesis` that generate random action sequences and assert that no unsafe state is ever reached (no conflicting greens, no green shorter than min, clearance always present).

# 6. M3: training

- MaskablePPO, MLP policy (2 x 128, tanh), n_steps 1024, batch 256, n_epochs 10, lr 3e-4 with linear decay, gamma 0.99, gae_lambda 0.95, clip 0.2, ent_coef 0.01, 4 parallel environments (SubprocVecEnv), observation normalisation (VecNormalize, saved with the policy).
- Curriculum: offpeak, then peaks, then a mixture including incident.
- Seeds 13, 42, 2026. Budget from the M0 speed test; target at least 300k environment steps per seed on `single`, more if time allows.
- Grid: parameter-shared policy across the 4 junctions with a junction-id embedding, trained after the single-junction runs start converging.
- Log with TensorBoard plus CSV; save the best checkpoint by evaluation return on a validation seed every N steps. Record wall-clock time.
- Fallback: if training is slow, run 2 seeds on `single` and 1 on `grid2x2`, and say so.

# 7. M6: evaluation

- Controllers: fixed_time, actuated or max-pressure, PPO (per seed). Scenarios: offpeak, morning_peak, evening_peak, incident, surge, and noisy sensors (Gaussian noise plus 5 percent dropped detector readings).
- Metrics from SUMO tripinfo and summary outputs: mean and 95th percentile travel time, mean waiting time, mean queue length, throughput, per-class waiting time (rickshaw and bus waits matter for fairness), pedestrian mean wait, number of shield interventions (must be reported even if zero), phase switches per hour.
- 10 evaluation episodes per scenario per controller with disjoint seeds; mean, std, 95 percent bootstrap CIs; percentage change vs fixed-time.
- Outputs: `results/eval_summary.json`, `results/main_table.csv`, figures (bar charts with CIs, wait-over-time curves, per-class fairness chart), all generated by `make eval report`.
- Replay export: for the website, export synchronized FCD (positions, angles, speeds, vehicle class) and TLS state for the same demand seed under fixed-time and PPO for `morning_peak` (single junction, required), `incident` (required) and `grid2x2` morning peak (if available). Write `scripts/make_replay.py` that samples at 0.5 s, quantizes positions to 5 cm, delta-encodes, and writes gzip-compressed JSON or a compact binary format plus a JSON manifest with scenario metadata and per-second aggregate metrics (queue per approach, vehicles waiting, average wait). Target under 10 MB per scenario. Include schema tests.

# 8. M4: 3D web app and landing page (`web/`)

## 8.1 Stack
Vite, React 18, TypeScript (strict), three.js through @react-three/fiber and @react-three/drei, @react-three/postprocessing for bloom, Zustand for state, a light chart library (uPlot or Recharts), @fontsource/noto-sans-bengali for Bengali signage text. No external runtime APIs; everything static.

## 8.2 Scene (procedural, no downloaded 3D assets, so licences stay clean)
- Ground, asphalt roads with correct left-hand lane markings, stop lines, zebra crossings, kerbs and sidewalks generated from the SUMO network geometry exported to JSON (`scripts/export_geometry.py`), so the 3D roads match the simulation exactly.
- Seeded procedural city blocks around the junction: mid-rise buildings with varied heights and pastel colours typical of Dhaka, balconies, water tanks on roofs, shop shutters at street level, a few billboards and shop signs with fictional Bengali text (for example "চা স্টল", "ফার্মেসি"), tangled overhead wires between poles, trees, street lamps, a tea stall, a rickshaw stand. Keep everything low-poly and stylised.
- Traffic-signal poles at every approach with red, yellow and green lamps driven by the replayed TLS state, plus pedestrian signals.
- Vehicles as low-poly models built from primitives, one InstancedMesh per class: car (several colours), red double-decker bus and single bus, truck, motorcycle with rider, green CNG auto-rickshaw with cage, colourful cycle-rickshaw with hood, pedestrians as simple capsules with walking bob. Smooth interpolation between samples, heading from angle, brake lights when decelerating, headlights at night.
- Lighting and mood: golden-hour default, day and night toggle (night uses emissive windows, street lamps and bloom), soft shadows on the main light only, gentle fog, optional rain toggle with a simple particle system for a monsoon look.
- Colour mode toggle: realistic colours or heat colouring by waiting time (cool to hot), plus a lane heatmap overlay of queue density.

## 8.3 Interaction and HUD
- Camera presets with smooth transitions: aerial isometric, top-down, street-level, and follow-a-vehicle (click a vehicle to follow it and show its class, speed and waiting time).
- Playback: play, pause, speed 0.5x to 8x, timeline scrubber with markers for the incident window, scenario selector, controller selector.
- **Split view**: fixed-time on the left and RL on the right, same demand and synchronized clock, a draggable divider, and shared camera.
- HUD panels (glassmorphism, readable on dark and light): vehicles waiting, average wait, queue per approach, current phase and timer, shield-intervention badge, and small live line charts comparing both controllers. All numbers come from the replay manifest, never hard-coded.

## 8.4 Landing page (single page, scroll story)
1. Hero: the live 3D split view autoplaying morning peak, title "Safe Deep RL Traffic Signals for Dhaka-like Mixed Traffic", subtitle, and two buttons (Explore the simulation, Read the report).
2. Key result cards generated at build time from `results/eval_summary.json` (for example average waiting time change vs fixed-time with CI, pedestrian wait change, shield interventions).
3. How it works: animated diagram of simulation, PPO agent, action mask, safety shield, controller.
4. Safety: the state machine drawn as a diagram plus the property-based testing explanation.
5. Results: interactive charts by scenario and vehicle class.
6. Edge deployment: ONNX int8 size and latency numbers from results.
7. Limitations and future work, author section (Souvik Mandol, links to GitHub and portfolio), footer with licences.

## 8.5 Performance and quality
- Target smooth playback (aim for 60 fps, acceptable above 30 fps) on a normal laptop integrated GPU: instancing, frustum culling, level of detail for buildings, capped shadow map, lazy-loading of replay files with a progress bar.
- Responsive: on phones show a single view with a toggle between controllers.
- Accessibility: keyboard controls, reduced-motion option, alt text for figures.
- ESLint and `tsc --noEmit` clean, a Playwright smoke test that loads the built site, waits for the canvas, and asserts no console errors.
- Fallback for M4: drop rain, follow-camera and grid scenario before dropping split view or HUD.

# 9. M5: edge export and controller service

- Export the PPO policy (with observation normalisation folded in) to ONNX (opset 17); verify action parity with PyTorch on 10,000 random observations (report agreement). Apply onnxruntime dynamic int8 quantization; report size, action agreement vs float, and single-thread CPU latency p50 and p95 over 10,000 calls (single thread to approximate a small edge CPU; say this is an approximation, not a Jetson measurement).
- Controller service (`src/drs/service/`): asyncio app that subscribes to detector messages on MQTT topics `drs/junction/<id>/detectors`, builds observations, runs the ONNX policy, passes the decision through the same safety shield, and publishes phase commands to `drs/junction/<id>/command` and status to `drs/junction/<id>/status`. JSON message schemas with pydantic and versioning. Watchdog: if the model fails, inference exceeds a deadline, or detector data is stale for more than N seconds, fall back to fixed-time and publish an alarm. Optional TLS with self-signed certificates generated by a script (document it; keep certs out of git).
- FastAPI sidecar: `/health`, `/metrics` (Prometheus: decisions, latency histogram, fallbacks, shield interventions).
- Simulator bridge: a script that runs SUMO and exchanges detector data and commands with the service over MQTT, so the full loop can be demonstrated; integration test runs it for 5 simulated minutes.
- Docker: `Dockerfile` for the service, `docker-compose.yml` with mosquitto, service, simulator bridge and the static web app (nginx). Document `docker compose up`.

# 10. M7: media and deployment

- Use Playwright with headless Chromium and software WebGL (`--use-gl=swiftshader` or `--use-angle=swiftshader`) to load the built site, set the camera presets, and capture frames at fixed time steps (not real time) for: a 30 to 40 second split-view morning-peak video, a 10 second incident clip, and a night shot. Encode with ffmpeg to MP4 (H.264) and an optimised GIF (palette-based, under about 8 MB) for the README hero. Save 4 PNG screenshots (aerial golden hour split view, street-level with rickshaws and CNGs, night with bloom, heatmap mode) in `docs/media/`.
- GitHub Actions workflow `pages.yml` that builds `web/` and deploys to GitHub Pages with the correct base path for a project site. Document in README the one manual step: repo Settings, Pages, Source "GitHub Actions" (and that GitHub Pages needs a public repo on the free plan).
- CI workflow `ci.yml`: ruff, mypy, pytest (unit plus a 2-minute smoke training), web lint, type check and build.

# 11. M8: README and technical report

- README.md: hero GIF at the top; one-paragraph pitch; live demo link (`https://<owner>.github.io/dhaka-rl-signals/` pattern); headline results table read from results; architecture diagram in Mermaid; how safety is guaranteed; how to run (`make setup`, `make train`, `make eval`, `make web`, `docker compose up`, `make reproduce` for a 15-minute small-budget run); project structure; tests and CI badges; limitations (synthetic demand inspired by Dhaka, no real-world data or validation, CPU training budget, simplified pedestrian model); licence; citation block.
- `docs/report.md` and, if `tectonic` installs, `docs/report.pdf` from LaTeX: a 5 to 8 page technical report written like a short paper: abstract, problem and motivation (mixed traffic, lane-less weaving, pedestrians), related work (verify every citation by fetching its official page, record in `docs/CITATIONS_VERIFIED.md`; candidates: sumo-rl, PressLight, MPLight or FRAP, CoLight, max-pressure control, safe RL with action masking or shielding, Proximal Policy Optimization), simulation design, method (observation, action, reward, masking, shield FSM), experimental setup, results with tables and figures, robustness and fairness across vehicle classes, edge deployment benchmarks, visualization design, limitations, future work (real detector data, sim-to-real, multi-agent coordination, transfer to other cities). Author: Souvik Mandol. No other authors.

# 12. M9: final self-review (do not skip)

1. `make lint test` passes; `make reproduce` runs end to end on the small budget; `npm run build` succeeds; the Playwright smoke test passes with no console errors.
2. `scripts/check_numbers.py` verifies that every number in README, report and the website's built JSON matches `results/`.
3. Safety: rerun the property-based tests with more examples and confirm zero violations; report shield interventions honestly.
4. Search the repo for anything that could look like proprietary material from a real deployment and remove it.
5. Write `REVIEW.md`: what is strong, what a hiring manager or reviewer would question, known bugs, and the next steps.
6. Push everything. Final line in PROGRESS.md: `ALL DELIVERABLES PUSHED`.

# 13. Stretch goals (only after M9)

Grid coordination comparison (independent vs parameter-shared vs neighbour-aware); a "what-if" panel in the web app that lets visitors change the vehicle mix and see precomputed results; a live mode where the web app connects over WebSocket to the running simulator bridge; per-class fairness-aware reward ablation; a short narrated demo script in `docs/demo_script.md` for a 2-minute video.

# 14. Known environment limits (handle without asking)
- Docker daemon may not be available in this VM. If `docker` does not work, still write Dockerfile and docker-compose.yml, but run mosquitto as a local process (apt or pip-installed broker) for tests, and state in README that Docker files were not executed here.
- Pushing files under .github/workflows/ may be rejected by the GitHub proxy (missing workflow permission). If a push fails because of workflow files, move them to docs/github-workflows/ with a note in README telling the owner to copy them into .github/workflows/ manually, and push everything else.
- Headless WebGL capture may be slow or fail. Try software rendering first; if video capture fails after two attempts, produce PNG screenshots plus a GIF built from a smaller number of frames, and record the issue in PROGRESS.md.

# 15. Additional ML/DL components (after the core RL results exist; protect the priority order)
A. Traffic forecasting (supervised DL): log per-lane queue and arrival time series from simulations on training seeds; train a small GRU and a Temporal Convolutional Network in PyTorch (CPU) to forecast queue length 30 s and 60 s ahead per lane. Report MAE and RMSE vs naive baselines (last value, moving average) on held-out seeds with CIs. Then add the forecasts to the PPO observation ("forecast-augmented PPO") and compare against plain PPO in evaluation.
B. Imitation learning warm start: generate state-action pairs from the max-pressure controller, train the policy network by behaviour cloning (cross-entropy, masked actions), then fine-tune with MaskablePPO. Compare learning curves (return vs environment steps and wall-clock) for PPO from scratch vs BC+PPO.
C. Graph neural network for the 2x2 grid: represent junctions as nodes and roads as edges; implement a small message-passing or graph-attention policy encoder in plain PyTorch (no heavy dependency) with parameter sharing; compare independent PPO, shared MLP PPO and GNN PPO on the grid.
For each: add configs, tests, results in results/, figures, a section in the technical report, a card on the website ("Forecasting", "Imitation learning", "Graph coordination"), and keep all numbers generated from results files. If time is short, do B first (cheapest, speeds up training), then A, then C.
