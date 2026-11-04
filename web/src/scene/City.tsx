import { useMemo } from 'react';
import * as THREE from 'three';
import { Detailed } from '@react-three/drei';
import type { Geometry } from '../replay/geometry';
import { PASTELS, SIGN_TEXTS, facadeTexture, pick, rng, signTexture } from './util';

interface Building {
  x: number;
  z: number;
  w: number;
  d: number;
  h: number;
  colour: string;
  tex: THREE.CanvasTexture;
  texNight: THREE.CanvasTexture;
  tanks: [number, number][];
  balconies: boolean;
  shutter: string;
  sign: { text: string; bg: string; fg: string; side: number } | null;
}

interface Prop {
  kind: 'tree' | 'lamp' | 'pole' | 'stall' | 'rickshaw';
  x: number;
  z: number;
  rot: number;
  s: number;
}

const ROAD_HALF = 15; // half width of the road corridor (lanes + sidewalks)
const CLEARANCE = 9; // extra metres between the kerb and the first building

/** Is the point inside any road corridor (axis-aligned arms through the junction centres)? */
function nearRoad(geometry: Geometry, x: number, y: number, margin: number): boolean {
  for (const [cx, cy] of Object.values(geometry.junction_centres)) {
    if (Math.abs(x - cx) < margin || Math.abs(y - cy) < margin) return true;
  }
  return false;
}

export function buildCity(geometry: Geometry, seed = 7): { buildings: Building[]; props: Prop[]; wires: [THREE.Vector3, THREE.Vector3][] } {
  const r = rng(seed);
  const [bx0, by0, bx1, by1] = geometry.bounds;
  const buildings: Building[] = [];
  const props: Prop[] = [];
  const wires: [THREE.Vector3, THREE.Vector3][] = [];
  const step = 16;
  for (let x = bx0 - 40; x < bx1 + 40; x += step) {
    for (let y = by0 - 40; y < by1 + 40; y += step) {
      const jx = x + (r() - 0.5) * 4;
      const jy = y + (r() - 0.5) * 4;
      if (nearRoad(geometry, jx, jy, ROAD_HALF + CLEARANCE)) continue;
      // keep density: skip some cells for courtyards and alleys
      if (r() < 0.22) continue;
      const toCentre = Math.min(...Object.values(geometry.junction_centres).map(([cx, cy]) => Math.hypot(jx - cx, jy - cy)));
      const w = 9 + r() * 6;
      const d = 9 + r() * 6;
      const floors = toCentre < 70 ? 2 + Math.floor(r() * 3) : toCentre < 140 ? 3 + Math.floor(r() * 4) : 4 + Math.floor(r() * 6);
      const h = floors * 3.1;
      const colour = pick(r, PASTELS);
      const sub = rng(Math.floor(r() * 1e9));
      const tanks: [number, number][] = [];
      const nt = Math.floor(r() * 3);
      for (let i = 0; i < nt; i++) tanks.push([(r() - 0.5) * (w - 3), (r() - 0.5) * (d - 3)]);
      const facesRoad = nearRoad(geometry, jx, jy, ROAD_HALF + CLEARANCE + step);
      buildings.push({
        x: jx,
        z: -jy,
        w,
        d,
        h,
        colour,
        tex: facadeTexture(sub, Math.max(2, Math.round(w / 3)), floors, colour, false),
        texNight: facadeTexture(rng(Math.floor(sub() * 1e9)), Math.max(2, Math.round(w / 3)), floors, colour, true),
        tanks,
        balconies: r() < 0.5,
        shutter: pick(r, ['#5a7f8a', '#8a5a5a', '#7a8a5a', '#6a6a8a']),
        sign: facesRoad && r() < 0.8 ? { text: pick(r, SIGN_TEXTS), bg: pick(r, ['#c8322b', '#1d6fb5', '#1f8a4c', '#e2a62b', '#7b2d8b']), fg: '#fff8e6', side: 0 } : null,
      });
    }
  }
  // street furniture along each arm
  for (const [jid, arms] of Object.entries(geometry.arms)) {
    const [cx, cy] = geometry.junction_centres[jid]!;
    for (const [dir, arm] of Object.entries(arms)) {
      const vx = dir === 'E' ? 1 : dir === 'W' ? -1 : 0;
      const vy = dir === 'N' ? 1 : dir === 'S' ? -1 : 0;
      const len = arm.external ? 280 : 110;
      let prevL: THREE.Vector3 | null = null;
      let prevR: THREE.Vector3 | null = null;
      for (let s = 24; s < len; s += 22) {
        for (const side of [-1, 1]) {
          const px = cx + vx * s + -vy * side * (ROAD_HALF - 0.5);
          const py = cy + vy * s + vx * side * (ROAD_HALF - 0.5);
          const isPole = Math.round(s / 22) % 2 === 0;
          props.push({ kind: isPole ? 'pole' : r() < 0.6 ? 'tree' : 'lamp', x: px, z: -py, rot: r() * Math.PI, s: 0.8 + r() * 0.5 });
          if (isPole) {
            const top = new THREE.Vector3(px, 7.5, -py);
            const prev = side < 0 ? prevL : prevR;
            if (prev) wires.push([prev, top]);
            if (side < 0) prevL = top;
            else prevR = top;
          }
        }
        if (r() < 0.25) {
          const side = r() < 0.5 ? -1 : 1;
          const px = cx + vx * s + 8 + -vy * side * (ROAD_HALF + 2.5);
          const py = cy + vy * s + 8 + vx * side * (ROAD_HALF + 2.5);
          if (!nearRoad(geometry, px, py, ROAD_HALF - 1)) props.push({ kind: r() < 0.5 ? 'stall' : 'rickshaw', x: px, z: -py, rot: Math.atan2(vx, vy) + (side < 0 ? Math.PI : 0), s: 1 });
        }
      }
      if (prevL && prevR && r() < 0.7) wires.push([prevL, prevR]);
    }
  }
  return { buildings, props, wires };
}

function BuildingMesh({ b, night }: { b: Building; night: boolean }) {
  const mat = useMemo(() => {
    const m = new THREE.MeshStandardMaterial({ map: night ? b.texNight : b.tex, color: '#ffffff', roughness: 0.9 });
    if (night) {
      m.emissiveMap = b.texNight;
      m.emissive = new THREE.Color('#ffd9a0');
      m.emissiveIntensity = 0.9;
    }
    return m;
  }, [b, night]);
  const sign = useMemo(() => (b.sign ? signTexture(b.sign.text, b.sign.bg, b.sign.fg) : null), [b.sign]);
  return (
    <group position={[b.x, 0, b.z]}>
      <Detailed distances={[0, 220]}>
        <group>
          <mesh position={[0, b.h / 2, 0]} castShadow receiveShadow material={mat}>
            <boxGeometry args={[b.w, b.h, b.d]} />
          </mesh>
          {/* roof slab, water tanks, parapet */}
          <mesh position={[0, b.h + 0.15, 0]}>
            <boxGeometry args={[b.w + 0.4, 0.3, b.d + 0.4]} />
            <meshStandardMaterial color="#9a9188" />
          </mesh>
          {b.tanks.map(([tx, tz], i) => (
            <mesh key={i} position={[tx, b.h + 1.1, tz]} castShadow>
              <cylinderGeometry args={[0.8, 0.8, 1.6, 10]} />
              <meshStandardMaterial color="#2b2b2b" roughness={0.6} />
            </mesh>
          ))}
          {b.balconies &&
            Array.from({ length: Math.max(1, Math.floor(b.h / 3.1) - 1) }, (_, k) => (
              <mesh key={k} position={[0, 3.1 * (k + 1) + 0.1, b.d / 2 + 0.5]} castShadow>
                <boxGeometry args={[b.w * 0.8, 0.2, 1.0]} />
                <meshStandardMaterial color="#d8d0c4" />
              </mesh>
            ))}
          {/* shop shutter at street level */}
          <mesh position={[0, 1.4, b.d / 2 + 0.02]}>
            <boxGeometry args={[b.w * 0.6, 2.6, 0.08]} />
            <meshStandardMaterial color={b.shutter} roughness={0.7} metalness={0.3} />
          </mesh>
          {sign && (
            <mesh position={[0, 3.4, b.d / 2 + 0.15]}>
              <planeGeometry args={[Math.min(b.w * 0.8, 7), 2]} />
              <meshStandardMaterial map={sign} emissive={night ? '#ffffff' : '#000000'} emissiveMap={night ? sign : null} emissiveIntensity={night ? 0.8 : 0} />
            </mesh>
          )}
        </group>
        <mesh position={[0, b.h / 2, 0]}>
          <boxGeometry args={[b.w, b.h, b.d]} />
          <meshStandardMaterial color={b.colour} />
        </mesh>
      </Detailed>
    </group>
  );
}

function PropMesh({ p, night }: { p: Prop; night: boolean }) {
  switch (p.kind) {
    case 'tree':
      return (
        <group position={[p.x, 0, p.z]} rotation={[0, p.rot, 0]} scale={p.s}>
          <mesh position={[0, 1.5, 0]} castShadow>
            <cylinderGeometry args={[0.25, 0.35, 3, 6]} />
            <meshStandardMaterial color="#5a3d26" />
          </mesh>
          <mesh position={[0, 4.6, 0]} castShadow>
            <coneGeometry args={[2.6, 4.5, 7]} />
            <meshStandardMaterial color="#3f7a3a" flatShading />
          </mesh>
          <mesh position={[0.8, 3.6, 0.5]} castShadow>
            <coneGeometry args={[1.8, 3.2, 6]} />
            <meshStandardMaterial color="#4f8f44" flatShading />
          </mesh>
        </group>
      );
    case 'lamp':
      return (
        <group position={[p.x, 0, p.z]}>
          <mesh position={[0, 4, 0]} castShadow>
            <cylinderGeometry args={[0.08, 0.12, 8, 6]} />
            <meshStandardMaterial color="#444" />
          </mesh>
          <mesh position={[0, 8, 0]}>
            <sphereGeometry args={[0.35, 8, 8]} />
            <meshStandardMaterial color="#fff2c0" emissive="#ffd27a" emissiveIntensity={night ? 4 : 0.2} />
          </mesh>
          {night && <pointLight position={[0, 7.6, 0]} color="#ffc877" intensity={25} distance={28} decay={2} />}
        </group>
      );
    case 'pole':
      return (
        <group position={[p.x, 0, p.z]}>
          <mesh position={[0, 3.75, 0]} castShadow>
            <cylinderGeometry args={[0.1, 0.14, 7.5, 6]} />
            <meshStandardMaterial color="#6b6b6b" />
          </mesh>
          <mesh position={[0, 7.2, 0]}>
            <boxGeometry args={[1.2, 0.08, 0.08]} />
            <meshStandardMaterial color="#555" />
          </mesh>
        </group>
      );
    case 'stall':
      return (
        <group position={[p.x, 0, p.z]} rotation={[0, p.rot, 0]}>
          <mesh position={[0, 0.9, 0]} castShadow>
            <boxGeometry args={[2.6, 1.8, 1.6]} />
            <meshStandardMaterial color="#8c6b4a" />
          </mesh>
          <mesh position={[0, 2.2, 0.4]} rotation={[0.25, 0, 0]} castShadow>
            <boxGeometry args={[3.4, 0.08, 2.6]} />
            <meshStandardMaterial color="#c8463a" />
          </mesh>
          <mesh position={[0.9, 1.9, 0.6]}>
            <cylinderGeometry args={[0.2, 0.25, 0.4, 8]} />
            <meshStandardMaterial color="#d8d8d8" metalness={0.6} roughness={0.3} />
          </mesh>
        </group>
      );
    case 'rickshaw':
      return (
        <group position={[p.x, 0, p.z]} rotation={[0, p.rot, 0]}>
          {[0, 1, 2].map((i) => (
            <group key={i} position={[i * 1.6 - 1.6, 0, 0]}>
              <mesh position={[0, 0.9, 0]} castShadow>
                <boxGeometry args={[1.1, 0.5, 1.6]} />
                <meshStandardMaterial color={['#2b6fd6', '#d63b2b', '#2bb37a'][i]} />
              </mesh>
              <mesh position={[0, 1.6, -0.2]} castShadow>
                <cylinderGeometry args={[0.6, 0.6, 1.1, 8, 1, false, 0, Math.PI]} />
                <meshStandardMaterial color={['#f0c030', '#30a0f0', '#f05030'][i]} side={THREE.DoubleSide} />
              </mesh>
            </group>
          ))}
        </group>
      );
  }
}

function Wires({ wires }: { wires: [THREE.Vector3, THREE.Vector3][] }) {
  const geom = useMemo(() => {
    const pts: number[] = [];
    for (const [a, b] of wires) {
      const segs = 8;
      for (let k = 0; k < segs; k++) {
        const t0 = k / segs;
        const t1 = (k + 1) / segs;
        for (const t of [t0, t1]) {
          const sag = 0.9 * Math.sin(t * Math.PI);
          pts.push(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t - sag, a.z + (b.z - a.z) * t);
        }
      }
      // a second, tangled wire slightly offset
      for (let k = 0; k < segs; k++) {
        const t0 = k / segs;
        const t1 = (k + 1) / segs;
        for (const t of [t0, t1]) {
          const sag = 1.3 * Math.sin(t * Math.PI);
          pts.push(a.x + (b.x - a.x) * t + 0.3, a.y + (b.y - a.y) * t - sag - 0.3, a.z + (b.z - a.z) * t + 0.3);
        }
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    return g;
  }, [wires]);
  return (
    <lineSegments geometry={geom}>
      <lineBasicMaterial color="#111" />
    </lineSegments>
  );
}

export function City({ geometry, night }: { geometry: Geometry; night: boolean }) {
  const city = useMemo(() => buildCity(geometry), [geometry]);
  return (
    <group>
      {city.buildings.map((b, i) => (
        <BuildingMesh key={i} b={b} night={night} />
      ))}
      {city.props.map((p, i) => (
        <PropMesh key={i} p={p} night={night} />
      ))}
      <Wires wires={city.wires} />
    </group>
  );
}
