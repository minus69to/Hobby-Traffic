import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { Geometry, SignalHead } from '../replay/geometry';

const OFF = new THREE.Color('#201a1a');
const RED = new THREE.Color('#ff2a1a');
const AMBER = new THREE.Color('#ffb020');
const GREEN = new THREE.Color('#2cff6a');
const WALK = new THREE.Color('#4cff9a');

/** A pole with a three-lamp head facing the approaching traffic, plus a pedestrian signal. */
function Head({ head, getState, night }: { head: SignalHead; getState: () => string; night: boolean }) {
  const lamps = useRef<(THREE.MeshStandardMaterial | null)[]>([]);
  const ped = useRef<THREE.MeshStandardMaterial | null>(null);
  const linkIdx = head.vehicle_links['s'] ?? head.vehicle_links['l'] ?? head.vehicle_links['r'] ?? 0;
  const rightIdx = head.vehicle_links['r'];
  const rightLamp = useRef<THREE.MeshStandardMaterial | null>(null);
  // place the pole on the left kerb (lefthand traffic) at the stop line, facing the approach
  const h = (head.heading_deg * Math.PI) / 180;
  const left = head.stop_line[0]!;
  const px = left[0] - Math.sin(h) * 3.2 + Math.cos(h) * 0.5;
  const pz = -(left[1] + Math.cos(h) * 3.2 + Math.sin(h) * 0.5);
  const yaw = Math.atan2(Math.cos(h), -Math.sin(h)) + Math.PI;
  useFrame(() => {
    const s = getState();
    const c = s[linkIdx] ?? 'r';
    const r = c === 'r' || c === 'R';
    const y = c === 'y' || c === 'Y';
    const g = c === 'G' || c === 'g';
    const inten = night ? 3.5 : 2.2;
    const m0 = lamps.current[0];
    const m1 = lamps.current[1];
    const m2 = lamps.current[2];
    if (m0) {
      m0.emissive.copy(r ? RED : OFF);
      m0.emissiveIntensity = r ? inten : 0.2;
    }
    if (m1) {
      m1.emissive.copy(y ? AMBER : OFF);
      m1.emissiveIntensity = y ? inten : 0.2;
    }
    if (m2) {
      m2.emissive.copy(g ? GREEN : OFF);
      m2.emissiveIntensity = g ? inten : 0.2;
    }
    if (rightLamp.current && rightIdx !== undefined) {
      const rc = s[rightIdx] ?? 'r';
      const rg = rc === 'G' || rc === 'g';
      const ry = rc === 'y';
      rightLamp.current.emissive.copy(rg ? GREEN : ry ? AMBER : OFF);
      rightLamp.current.emissiveIntensity = rg || ry ? inten : 0.15;
    }
    if (ped.current && head.crossing_link !== null) {
      const pc = s[head.crossing_link] ?? 'r';
      const walk = pc === 'G' || pc === 'g';
      ped.current.emissive.copy(walk ? WALK : RED);
      ped.current.emissiveIntensity = inten * 0.8;
    }
  });
  return (
    <group position={[px, 0, pz]} rotation={[0, yaw, 0]}>
      <mesh position={[0, 2.6, 0]} castShadow>
        <cylinderGeometry args={[0.09, 0.12, 5.2, 8]} />
        <meshStandardMaterial color="#3a3f44" metalness={0.5} roughness={0.5} />
      </mesh>
      {/* mast arm over the road */}
      <mesh position={[0, 5.1, -2.2]} rotation={[Math.PI / 2, 0, 0]} castShadow>
        <cylinderGeometry args={[0.07, 0.07, 4.4, 6]} />
        <meshStandardMaterial color="#3a3f44" metalness={0.5} roughness={0.5} />
      </mesh>
      {/* lamp housing */}
      <mesh position={[0, 4.55, -4.2]} castShadow>
        <boxGeometry args={[0.42, 1.25, 0.4]} />
        <meshStandardMaterial color="#1b1b1b" />
      </mesh>
      {[0, 1, 2].map((i) => (
        <mesh key={i} position={[0, 4.95 - i * 0.4, -4.0]}>
          <sphereGeometry args={[0.14, 10, 10]} />
          <meshStandardMaterial
            ref={(m) => {
              lamps.current[i] = m;
            }}
            color="#111"
            emissive={OFF}
            toneMapped={false}
          />
        </mesh>
      ))}
      {rightIdx !== undefined && (
        <group position={[0.5, 4.55, -4.2]}>
          <mesh castShadow>
            <boxGeometry args={[0.36, 0.42, 0.36]} />
            <meshStandardMaterial color="#1b1b1b" />
          </mesh>
          <mesh position={[0, 0, 0.19]}>
            <boxGeometry args={[0.22, 0.08, 0.04]} />
            <meshStandardMaterial
              ref={(m) => {
                rightLamp.current = m;
              }}
              color="#111"
              emissive={OFF}
              toneMapped={false}
            />
          </mesh>
        </group>
      )}
      {/* pedestrian signal on the pole facing the crossing */}
      <mesh position={[0.3, 2.3, 0]} rotation={[0, Math.PI / 2, 0]}>
        <boxGeometry args={[0.3, 0.5, 0.2]} />
        <meshStandardMaterial
          ref={(m) => {
            ped.current = m;
          }}
          color="#111"
          emissive={RED}
          toneMapped={false}
        />
      </mesh>
    </group>
  );
}

export function Signals({ geometry, getState, night }: { geometry: Geometry; getState: (junction: string) => string; night: boolean }) {
  const getters = useMemo(() => {
    const m = new Map<string, () => string>();
    for (const j of Object.keys(geometry.phases)) m.set(j, () => getState(j));
    return m;
  }, [geometry, getState]);
  return (
    <group>
      {geometry.signals.map((h) => (
        <Head key={`${h.junction}-${h.arm}`} head={h} getState={getters.get(h.junction)!} night={night} />
      ))}
    </group>
  );
}
