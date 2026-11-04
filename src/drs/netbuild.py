"""Build the SUMO networks (plain XML + netconvert) and the Dhaka vehicle-type file.

Run ``python -m drs.netbuild --config configs/sim.yaml``.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from xml.sax.saxutils import quoteattr

from drs.config import REPO_ROOT, SimConfig, sumo_binary
from drs.topology import NetworkSpec, build_spec

NET_DIR = REPO_ROOT / "sim" / "networks"
VTYPE_DIR = REPO_ROOT / "sim" / "vtypes"


def write_plain_xml(spec: NetworkSpec, out_dir: Path) -> tuple[Path, Path]:
    """Write ``<name>.nod.xml`` and ``<name>.edg.xml`` for `spec`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nod = out_dir / f"{spec.name}.nod.xml"
    edg = out_dir / f"{spec.name}.edg.xml"
    with nod.open("w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n<nodes>\n')
        for n in spec.nodes.values():
            ntype = 'traffic_light" tlType="static' if n.signalised else "priority"
            fh.write(f'  <node id="{n.id}" x="{n.x:.2f}" y="{n.y:.2f}" type="{ntype}"/>\n')
        fh.write("</nodes>\n")
    with edg.open("w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n<edges>\n')
        for eid, frm, to in spec.edges:
            fh.write(
                f'  <edge id="{eid}" from="{frm}" to="{to}" numLanes="{spec.lanes_per_approach}" '
                f'speed="{spec.speed_limit:.2f}" sidewalkWidth="{spec.sidewalk_width:.2f}"/>\n'
            )
        fh.write("</edges>\n")
    return nod, edg


def run_netconvert(spec: NetworkSpec, out_dir: Path, lefthand: bool = True) -> Path:
    """Run netconvert and return the path of the generated ``.net.xml``."""
    nod, edg = write_plain_xml(spec, out_dir)
    net = out_dir / f"{spec.name}.net.xml"
    cmd = [
        sumo_binary("netconvert"),
        "--node-files",
        str(nod),
        "--edge-files",
        str(edg),
        "--output-file",
        str(net),
        "--no-turnarounds",
        "true",
        "--crossings.guess",
        "true",
        "--walkingareas",
        "true",
        "--junctions.corner-detail",
        "5",
        "--default.junctions.radius",
        "12",
        "--tls.default-type",
        "static",
        "--no-internal-links",
        "false",
        "--junctions.join",
        "false",
    ]
    if lefthand:
        cmd += ["--lefthand", "true"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return net


def write_vtypes(cfg: SimConfig, out_path: Path) -> Path:
    """Write ``dhaka.add.xml`` with one ``vType`` per vehicle class from the config."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    attrs_order = [
        "vClass",
        "length",
        "width",
        "maxSpeed",
        "accel",
        "decel",
        "sigma",
        "minGapLat",
        "latAlignment",
        "lcPushy",
        "lcStrategic",
        "color",
    ]
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fh.write("<!-- Synthetic vehicle types inspired by Dhaka traffic. Parameters are assumptions,\n")
        fh.write("     not measurements. See configs/sim.yaml and DECISIONS.md. -->\n<additional>\n")
        for name, params in cfg.vtypes.items():
            parts = [f"id={quoteattr(name)}"]
            for key in attrs_order:
                if key in params:
                    parts.append(f"{key}={quoteattr(str(params[key]))}")
            if name != "pedestrian":
                parts.append('guiShape="passenger"' if name == "car" else f"guiShape={quoteattr(_gui_shape(name))}")
            fh.write("  <vType " + " ".join(parts) + "/>\n")
        fh.write("</additional>\n")
    return out_path


def _gui_shape(name: str) -> str:
    return {
        "bus": "bus",
        "bus_dd": "bus",
        "truck": "truck",
        "motorcycle": "motorcycle",
        "cng": "passenger/hatchback",
        "rickshaw": "moped",
    }.get(name, "passenger")


def build_all(cfg: SimConfig) -> list[Path]:
    """Build every network in the config plus the vehicle types file."""
    out = []
    for name in cfg.networks:
        spec = build_spec(name, cfg.networks)
        out.append(run_netconvert(spec, NET_DIR, lefthand=cfg.sumo.lefthand))
    out.append(write_vtypes(cfg, VTYPE_DIR / "dhaka.add.xml"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sim.yaml")
    args = parser.parse_args()
    cfg = SimConfig.load(args.config)
    for p in build_all(cfg):
        print(f"wrote {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
