"""Simulator bridge: runs SUMO and exchanges detector data and commands with the service over MQTT.

Run ``python -m drs.service.bridge --config configs/service.yaml --scenario morning_peak --seconds 300``.
Every simulated second the bridge publishes a :class:`DetectorMessage`, waits (up to ``--wait``
seconds of wall-clock) for the matching :class:`CommandMessage` and applies its lamp state.  If no
command arrives in time, the previous state is kept and a miss is counted (the shield inside the
service keeps the sequence safe; the bridge never invents states).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import aiomqtt

from drs.config import SimConfig, load_yaml
from drs.demand import ensure_routes
from drs.service.app import apply_override, payload_bytes, tls_context
from drs.service.schemas import CommandMessage, CrossingReading, DetectorMessage, LaneReading
from drs.sumo_sim import VEHICLE_CLASSES, SumoSim
from drs.topology import DIRS


def detector_message(sim: SumoSim, jid: str, t: float) -> DetectorMessage:
    s = sim.junction_state(jid)
    table = sim.tables[jid]
    lanes = []
    for i, lane in enumerate(s.lanes):
        arm = sim._lane_arm[jid][lane]
        arm_idx = DIRS.index(arm)
        n_lanes_arm = sum(1 for ln in s.lanes if sim._lane_arm[jid][ln] == arm)
        out_lane = next((lk.to_lane for lk in table.links if lk.from_lane == lane and lk.turn != "c"), None)
        lanes.append(
            LaneReading(
                lane_id=lane,
                halting=float(s.lane_halting[i]),
                max_wait=float(s.lane_max_wait[i]),
                count=float(s.lane_count[i]),
                class_counts={
                    c: float(s.class_counts[arm_idx, k]) / n_lanes_arm for k, c in enumerate(VEHICLE_CLASSES)
                },
                out_halting=float(s.out_lane_halting.get(out_lane, 0.0)) if out_lane else 0.0,
            )
        )
    crossings = [CrossingReading(crossing_id=f":{jid}_w{k}", waiting=float(s.ped_waiting[k])) for k in range(4)]
    return DetectorMessage(junction_id=jid, timestamp=t, lanes=lanes, crossings=crossings)


async def run_bridge(
    scfg: dict[str, Any], scenario: str, seconds: int, seed: int, wait_s: float, log_path: Path | None = None
) -> dict[str, Any]:
    cfg = SimConfig.load(scfg["sim_config"])
    network = str(scfg.get("network", "single"))
    jid = str(scfg["junction_id"])
    route = ensure_routes(cfg, network, scenario, seed)
    sim = SumoSim(cfg, network, route, seed, episode_seconds=seconds)
    sim.start()
    m = scfg["mqtt"]
    topics = {k: str(v).format(id=jid) for k, v in scfg["topics"].items()}
    stats: dict[str, Any] = {"steps": 0, "commands": 0, "misses": 0, "states": [], "alarms": []}
    last_state = sim.tables[jid].phases[0].state
    t0 = time.perf_counter()
    async with aiomqtt.Client(str(m["host"]), int(m["port"]), tls_context=tls_context(m)) as client:
        await client.subscribe(topics["command"], qos=1)
        await client.subscribe(topics["alarm"], qos=1)
        queue: asyncio.Queue[CommandMessage] = asyncio.Queue()

        async def reader() -> None:
            async for message in client.messages:
                try:
                    payload = payload_bytes(message.payload)
                    if str(message.topic) == topics["alarm"]:
                        stats["alarms"].append(json.loads(payload))
                        continue
                    queue.put_nowait(CommandMessage.model_validate_json(payload))
                except Exception as e:
                    stats["bad_messages"] = stats.get("bad_messages", 0) + 1
                    print(f"[bridge] ignoring message on {message.topic}: {e}")

        task = asyncio.create_task(reader())
        try:
            for step in range(seconds):
                t = float(step)
                await client.publish(topics["detectors"], detector_message(sim, jid, t).model_dump_json(), qos=1)
                if task.done():
                    raise RuntimeError("bridge reader task ended unexpectedly") from task.exception()
                try:
                    loop = asyncio.get_running_loop()
                    deadline = loop.time() + wait_s
                    cmd = await asyncio.wait_for(queue.get(), timeout=wait_s)
                    while cmd.timestamp < t:  # drain stale commands while waiting for the current one
                        remaining = deadline - loop.time()
                        if remaining <= 0:
                            raise TimeoutError
                        cmd = await asyncio.wait_for(queue.get(), timeout=remaining)
                    last_state = cmd.state
                    stats["commands"] += 1
                except TimeoutError:
                    stats["misses"] += 1
                sim.set_state(jid, last_state)
                stats["states"].append(last_state)
                sim.step()
                stats["steps"] += 1
        finally:
            task.cancel()
            sim.close()
    stats["wall_s"] = round(time.perf_counter() - t0, 1)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps({k: v for k, v in stats.items() if k != "states"}, indent=1), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/service.yaml")
    parser.add_argument("--scenario", default="morning_peak")
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--wait", type=float, default=2.0)
    parser.add_argument("--override", default=None)
    args = parser.parse_args()
    scfg = apply_override(load_yaml(args.config), args.override)
    stats = asyncio.run(run_bridge(scfg, args.scenario, args.seconds, args.seed, args.wait))
    print({k: v for k, v in stats.items() if k != "states"})


if __name__ == "__main__":
    main()
