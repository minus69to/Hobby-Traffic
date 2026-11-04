# PROGRESS

Timestamped (UTC) log of work on `dhaka-rl-signals`. If interrupted, read `SPEC.md` and this file and continue.

- 2026-11-04 16:20 M0 started. Repo had only a stub README. Saved SPEC.md verbatim. Python 3.13, Node 22, 4 vCPU, 16 GB RAM, docker CLI present but no daemon (`docker ps` fails) so Docker files will be written but not executed here. Started pip installs (torch CPU, eclipse-sumo 1.28, libsumo, sb3-contrib) and apt install of mosquitto in the background.
- 2026-11-04 16:35 M0 done. Networks (`single`, `grid2x2`) build with netconvert (lefthand, crossings, sidewalks). Speed test: 3600 s of morning peak on `single` with the sublane model and pedestrians runs at ~100 sim steps/s per process (results/speedtest.json). Budget: 300k env steps/seed x 3 seeds ~ 3.2 h wall on 4 cores.
- 2026-11-04 16:50 M1/M2 done: demand generator (6 scenarios, seeded flows per class, incident lane blockage, pedestrian flows), four-phase plan built from the net file, safety shield FSM with mask, 15 unit + hypothesis tests passing. Committed and pushed.
- 2026-11-04 17:05 M3 started: `python -m drs.train --config configs/train.yaml` running in background (logs/train_single.log), seeds 13, 42, 2026 sequentially, 4 envs each. Small-budget smoke run (configs/train_small.yaml) verified end to end in 33 s.
