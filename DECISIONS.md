# DECISIONS

Ambiguities resolved autonomously, with the reason. Newest at the bottom.

1. **Branch.** The spec says push to `main`; the session's instructions name the feature branch `claude/great-goodall-2fl9mt`. All work is pushed to that branch (the owner can fast-forward `main`). Reason: the branch instruction is the more specific one.
2. **Docker.** No Docker daemon is available in this VM. `Dockerfile` and `docker-compose.yml` are written and linted by eye but not executed; mosquitto runs as a local apt-installed process for tests. Stated in README.
3. **Traffic-light control path.** Instead of relying on SUMO's built-in programs, every controller (fixed-time, max-pressure, PPO, the MQTT service) produces a *phase request*, the safety shield FSM turns it into a red/yellow/green state string, and that string is written to SUMO every step with `setRedYellowGreenState`. Reason: one code path, one shield, exact parity between training, evaluation and deployment.
