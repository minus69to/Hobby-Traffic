import { useMemo } from 'react';
import * as THREE from 'three';
import type { Geometry, Lane } from '../replay/geometry';
import { pointAt, polygonGeometry, polylineLength, ribbonGeometry } from './util';

const ASPHALT = '#3b3a40';
const ASPHALT_NIGHT = '#2a2a33';
const SIDEWALK = '#b9ad9c';
const KERB = '#8f877b';

function mergeGeoms(geoms: THREE.BufferGeometry[]): THREE.BufferGeometry | null {
  if (geoms.length === 0) return null;
  const merged = new THREE.BufferGeometry();
  const pos: number[] = [];
  const idx: number[] = [];
  let base = 0;
  for (const g of geoms) {
    const p = g.getAttribute('position');
    for (let i = 0; i < p.count; i++) pos.push(p.getX(i), p.getY(i), p.getZ(i));
    const ix = g.getIndex();
    if (ix) for (let i = 0; i < ix.count; i++) idx.push(ix.getX(i) + base);
    else for (let i = 0; i < p.count; i++) idx.push(i + base);
    base += p.count;
  }
  merged.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  merged.setIndex(idx);
  merged.computeVertexNormals();
  return merged;
}

/** Dashed lane separator boxes along a lane's left edge (SUMO coords). */
function dashes(lane: Lane, side: number, dashLen = 3, gap = 3, width = 0.15): THREE.BufferGeometry[] {
  const out: THREE.BufferGeometry[] = [];
  const L = polylineLength(lane.shape);
  const hw = lane.width / 2;
  for (let s = 10; s < L - 12; s += dashLen + gap) {
    const a = pointAt(lane.shape, s);
    const b = pointAt(lane.shape, s + dashLen);
    const nx = -a.ty * side;
    const ny = a.tx * side;
    out.push(
      ribbonGeometry(
        [
          [a.x + nx * hw, a.y + ny * hw],
          [b.x + nx * hw, b.y + ny * hw],
        ],
        width,
        0.02,
      ),
    );
  }
  return out;
}

function zebra(shape: [number, number][], width: number): THREE.BufferGeometry[] {
  const out: THREE.BufferGeometry[] = [];
  const L = polylineLength(shape);
  const stripe = 0.6;
  for (let s = 0.4; s < L - 0.4; s += stripe * 2) {
    const a = pointAt(shape, s);
    const b = pointAt(shape, Math.min(s + stripe, L));
    out.push(
      ribbonGeometry(
        [
          [a.x, a.y],
          [b.x, b.y],
        ],
        width * 0.9,
        0.025,
      ),
    );
  }
  return out;
}

export function Roads({ geometry, night, queueByLane }: { geometry: Geometry; night: boolean; queueByLane: Map<string, number> | null }) {
  const built = useMemo(() => {
    const roadGeoms: THREE.BufferGeometry[] = [];
    const walkGeoms: THREE.BufferGeometry[] = [];
    const kerbGeoms: THREE.BufferGeometry[] = [];
    const markGeoms: THREE.BufferGeometry[] = [];
    const yellowGeoms: THREE.BufferGeometry[] = [];
    const zebraGeoms: THREE.BufferGeometry[] = [];
    const stopGeoms: THREE.BufferGeometry[] = [];
    const laneOverlay: { id: string; geom: THREE.BufferGeometry }[] = [];
    const vehicleLanesByEdge = new Map<string, Lane[]>();
    for (const lane of geometry.lanes) {
      if (lane.function === 'normal') {
        if (lane.pedestrian) {
          walkGeoms.push(ribbonGeometry(lane.shape, lane.width + 0.6, 0.12));
          kerbGeoms.push(ribbonGeometry(lane.shape, lane.width + 1.0, 0.06));
        } else {
          roadGeoms.push(ribbonGeometry(lane.shape, lane.width + 0.05, 0));
          laneOverlay.push({ id: lane.id, geom: ribbonGeometry(lane.shape, lane.width - 0.4, 0.03) });
          const arr = vehicleLanesByEdge.get(lane.edge) ?? [];
          arr.push(lane);
          vehicleLanesByEdge.set(lane.edge, arr);
        }
      } else if (lane.function === 'internal') {
        roadGeoms.push(ribbonGeometry(lane.shape, lane.width + 0.3, 0));
      } else if (lane.function === 'walkingarea') {
        walkGeoms.push(polygonGeometry(lane.shape, 0.12));
        kerbGeoms.push(polygonGeometry(lane.shape, 0.06));
      } else if (lane.function === 'crossing') {
        roadGeoms.push(ribbonGeometry(lane.shape, lane.width, 0.005));
        zebraGeoms.push(...zebra(lane.shape, lane.width));
      }
    }
    for (const j of geometry.junctions) {
      if (j.shape.length >= 3) roadGeoms.push(polygonGeometry(j.shape, 0.01));
    }
    // lane markings: dashed between lanes of the same edge, solid yellow centre line between edge pairs
    for (const lanes of vehicleLanesByEdge.values()) {
      const sorted = [...lanes].sort((a, b) => a.id.localeCompare(b.id));
      for (let i = 0; i < sorted.length - 1; i++) markGeoms.push(...dashes(sorted[i]!, -1));
      const outer = sorted[sorted.length - 1]!; // highest index lane is next to oncoming traffic (lefthand mirrored)
      const L = polylineLength(outer.shape);
      const a = pointAt(outer.shape, 8);
      const b = pointAt(outer.shape, L - 10);
      const hw = outer.width / 2 + 0.1;
      const nx = -a.ty * -1;
      const ny = a.tx * -1;
      yellowGeoms.push(
        ribbonGeometry(
          [
            [a.x + nx * hw, a.y + ny * hw],
            [b.x + nx * hw, b.y + ny * hw],
          ],
          0.18,
          0.02,
        ),
      );
    }
    for (const sh of geometry.signals) {
      const pts = sh.stop_line;
      if (pts.length >= 2) {
        const a = pts[0]!;
        const b = pts[pts.length - 1]!;
        const dx = b[0] - a[0];
        const dy = b[1] - a[1];
        const len = Math.hypot(dx, dy) || 1;
        const ex = (dx / len) * 1.6;
        const ey = (dy / len) * 1.6;
        stopGeoms.push(
          ribbonGeometry(
            [
              [a[0] - ex - Math.cos((sh.heading_deg * Math.PI) / 180) * 1.0, a[1] - ey - Math.sin((sh.heading_deg * Math.PI) / 180) * 1.0],
              [b[0] + ex - Math.cos((sh.heading_deg * Math.PI) / 180) * 1.0, b[1] + ey - Math.sin((sh.heading_deg * Math.PI) / 180) * 1.0],
            ],
            0.5,
            0.03,
          ),
        );
      }
    }
    return {
      road: mergeGeoms(roadGeoms),
      walk: mergeGeoms(walkGeoms),
      kerb: mergeGeoms(kerbGeoms),
      marks: mergeGeoms(markGeoms),
      yellow: mergeGeoms(yellowGeoms),
      zebra: mergeGeoms(zebraGeoms),
      stop: mergeGeoms(stopGeoms),
      laneOverlay,
    };
  }, [geometry]);

  const [bx0, by0, bx1, by1] = geometry.bounds;
  return (
    <group>
      {/* ground */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[(bx0 + bx1) / 2, -0.05, -(by0 + by1) / 2]} receiveShadow>
        <planeGeometry args={[(bx1 - bx0) * 2.2, (by1 - by0) * 2.2]} />
        <meshStandardMaterial color={night ? '#1d1a22' : '#6f7a55'} />
      </mesh>
      {built.road && (
        <mesh geometry={built.road} receiveShadow>
          <meshStandardMaterial color={night ? ASPHALT_NIGHT : ASPHALT} roughness={0.95} />
        </mesh>
      )}
      {built.kerb && (
        <mesh geometry={built.kerb} receiveShadow>
          <meshStandardMaterial color={KERB} roughness={0.9} />
        </mesh>
      )}
      {built.walk && (
        <mesh geometry={built.walk} receiveShadow>
          <meshStandardMaterial color={night ? '#5a564f' : SIDEWALK} roughness={0.9} />
        </mesh>
      )}
      {built.marks && (
        <mesh geometry={built.marks}>
          <meshBasicMaterial color="#e8e6dc" />
        </mesh>
      )}
      {built.yellow && (
        <mesh geometry={built.yellow}>
          <meshBasicMaterial color="#e3b53a" />
        </mesh>
      )}
      {built.zebra && (
        <mesh geometry={built.zebra}>
          <meshBasicMaterial color="#f1efe6" />
        </mesh>
      )}
      {built.stop && (
        <mesh geometry={built.stop}>
          <meshBasicMaterial color="#f1efe6" />
        </mesh>
      )}
      {queueByLane &&
        built.laneOverlay.map(({ id, geom }) => {
          const q = queueByLane.get(id) ?? 0;
          const f = Math.min(q / 25, 1);
          const col = new THREE.Color().setHSL(0.55 - 0.55 * f, 0.9, 0.5);
          return (
            <mesh key={id} geometry={geom}>
              <meshBasicMaterial color={col} transparent opacity={0.15 + 0.55 * f} depthWrite={false} />
            </mesh>
          );
        })}
    </group>
  );
}
