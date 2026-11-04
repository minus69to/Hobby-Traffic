"""asyncio MQTT controller service.

Run ``python -m drs.service.app --config configs/service.yaml``.  Subscribes to detector messages,
runs the ONNX policy through the safety shield, publishes commands/status/alarms and serves the
FastAPI sidecar (``/health``, ``/metrics``).  Optional TLS: set ``mqtt.tls: true`` and point the
``cafile``/``certfile``/``keyfile`` entries at certificates made by ``scripts/gen_certs.sh``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import ssl
from typing import Any

import aiomqtt
import uvicorn

from drs.config import REPO_ROOT, SimConfig, load_yaml
from drs.service.core import ControllerCore, CoreConfig, CoreOutputs
from drs.service.schemas import DetectorMessage
from drs.service.sidecar import Metrics, make_app


def apply_override(scfg: dict[str, Any], override: str | None) -> dict[str, Any]:
    if override:
        for k, v in json.loads(override).items():
            if isinstance(v, dict) and isinstance(scfg.get(k), dict):
                scfg[k].update(v)
            else:
                scfg[k] = v
    return scfg


def core_from_config(scfg: dict[str, Any]) -> ControllerCore:
    sim_cfg = SimConfig.load(scfg["sim_config"])
    wd = scfg.get("watchdog", {})
    ccfg = CoreConfig(
        junction_id=str(scfg["junction_id"]),
        network=str(scfg.get("network", "single")),
        decision_interval=float(sim_cfg.sumo.decision_interval),
        stale_after_s=float(wd.get("stale_after_s", 15)),
        inference_deadline_ms=float(wd.get("inference_deadline_ms", 50)),
    )
    onnx = scfg.get("model_onnx")
    meta = scfg.get("model_meta")
    if onnx and not (REPO_ROOT / onnx).exists():
        onnx = meta = None
    return ControllerCore(sim_cfg, ccfg, onnx, meta)


def tls_context(m: dict[str, Any]) -> ssl.SSLContext | None:
    if not m.get("tls"):
        return None
    ctx = ssl.create_default_context(cafile=str(REPO_ROOT / m["cafile"]))
    ctx.load_cert_chain(str(REPO_ROOT / m["certfile"]), str(REPO_ROOT / m["keyfile"]))
    ctx.check_hostname = False
    return ctx


def payload_bytes(payload: Any) -> bytes | str:
    if isinstance(payload, bytes | str):
        return payload
    return bytes(payload)


class Service:
    def __init__(self, scfg: dict[str, Any]) -> None:
        self.scfg = scfg
        self.core = core_from_config(scfg)
        self.metrics = Metrics()
        jid = self.core.ccfg.junction_id
        self.topics = {k: str(v).format(id=jid) for k, v in scfg["topics"].items()}
        self.stop = asyncio.Event()

    async def publish_outputs(self, client: aiomqtt.Client, out: CoreOutputs) -> None:
        for c in out.commands:
            await client.publish(self.topics["command"], c.model_dump_json(), qos=1)
        if out.status is not None:
            await client.publish(self.topics["status"], out.status.model_dump_json(), qos=0)
        for a in out.alarms:
            await client.publish(self.topics["alarm"], a.model_dump_json(), qos=1)
            if a.kind != "recovered":
                self.metrics.fallbacks.labels(a.kind).inc()
        self.metrics.decisions.labels(self.core.controller).inc(len(out.commands))
        for ms in self.core.latencies[-1:]:
            self.metrics.latency.observe(ms / 1000.0)

    async def mqtt_loop(self) -> None:
        m = self.scfg["mqtt"]
        async with aiomqtt.Client(str(m["host"]), int(m["port"]), tls_context=tls_context(m)) as client:
            await client.subscribe(self.topics["detectors"], qos=1)
            watchdog = asyncio.create_task(self.watchdog_loop(client))
            try:
                async for message in client.messages:
                    if self.stop.is_set():
                        break
                    try:
                        msg = DetectorMessage.model_validate_json(payload_bytes(message.payload))
                    except Exception as e:
                        print(f"[service] bad detector message: {e}")
                        continue
                    out = self.core.on_detectors(msg)
                    await self.publish_outputs(client, out)
            finally:
                watchdog.cancel()

    async def watchdog_loop(self, client: aiomqtt.Client) -> None:
        tick = float(self.scfg.get("watchdog", {}).get("tick_s", 1.0))
        while not self.stop.is_set():
            await asyncio.sleep(tick)
            out = self.core.watchdog()
            if out.commands or out.alarms:
                await self.publish_outputs(client, out)

    async def http_loop(self) -> None:
        h = self.scfg.get("http", {})
        config = uvicorn.Config(
            make_app(self.core, self.metrics),
            host=str(h.get("host", "0.0.0.0")),
            port=int(h.get("port", 8080)),
            log_level="warning",
        )
        await uvicorn.Server(config).serve()

    async def run(self) -> None:
        print(
            f"[service] junction {self.core.ccfg.junction_id}: "
            f"model_loaded={self.core.model_loaded} controller={self.core.controller}",
            flush=True,
        )
        await asyncio.gather(self.mqtt_loop(), self.http_loop())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/service.yaml")
    parser.add_argument(
        "--override", default=None, help='JSON with config overrides, e.g. \'{"mqtt": {"port": 1884}}\''
    )
    args = parser.parse_args()
    asyncio.run(Service(apply_override(load_yaml(args.config), args.override)).run())


if __name__ == "__main__":
    main()
