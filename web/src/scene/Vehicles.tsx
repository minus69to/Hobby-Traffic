import { useEffect, useMemo, useRef } from 'react';
import { useFrame, type ThreeEvent } from '@react-three/fiber';
import * as THREE from 'three';
import { samplePose, type TrackPose } from '../replay/decode';
import { CLASS_NAMES, type ClassName, type Replay, type Track } from '../replay/types';
import { buildModels, type ClassModel } from './vehicleModels';
import { clock, useStore } from '../store/playback';

const CAPACITY: Record<ClassName, number> = { car: 500, bus: 200, bus_dd: 120, truck: 160, motorcycle: 500, cng: 450, rickshaw: 400, pedestrian: 900 };
const tmpM = new THREE.Matrix4();
const tmpQ = new THREE.Quaternion();
const tmpP = new THREE.Vector3();
const tmpS = new THREE.Vector3(1, 1, 1);
const tmpC = new THREE.Color();
const BLACK = new THREE.Color(0, 0, 0);
const pose: TrackPose = { x: 0, y: 0, angle: 0, speed: 0, wait: 0, accel: 0 };

function hashId(id: string): number {
  let h = 2166136261;
  for (let i = 0; i < id.length; i++) h = Math.imul(h ^ id.charCodeAt(i), 16777619);
  return (h >>> 0) / 4294967296;
}

interface ClassBuffers {
  body: THREE.InstancedMesh;
  detail: THREE.InstancedMesh;
  brake: THREE.InstancedMesh;
  head: THREE.InstancedMesh;
  ids: string[];
}

export interface VehiclesHandle {
  /** track id -> world position, updated every frame (used by the follow camera) */
  positions: Map<string, THREE.Vector3>;
}

export function Vehicles({
  replay,
  night,
  heat,
  onPick,
  handle,
  reducedMotion,
}: {
  replay: Replay;
  night: boolean;
  heat: boolean;
  onPick: (track: Track) => void;
  handle: VehiclesHandle;
  reducedMotion: boolean;
}) {
  const models = useMemo(() => buildModels(), []);
  const bodyMat = useMemo(() => new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0.15 }), []);
  const detailMat = useMemo(() => new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.8 }), []);
  const lightMat = useMemo(() => new THREE.MeshBasicMaterial({ vertexColors: true, toneMapped: false }), []);
  const buffers = useRef<Partial<Record<ClassName, ClassBuffers>>>({});
  const colourCache = useMemo(() => {
    const m = new Map<string, THREE.Color>();
    for (const tr of replay.tracks) {
      const pal = models[tr.cls].palette;
      m.set(tr.id, new THREE.Color(pal[Math.floor(hashId(tr.id) * pal.length)] ?? '#ffffff'));
    }
    return m;
  }, [replay, models]);
  const trackByClassInstance = useRef<Map<ClassName, Track[]>>(new Map());

  useEffect(() => {
    const b = buffers.current;
    for (const cls of CLASS_NAMES) {
      const cb = b[cls];
      if (!cb) continue;
      for (const im of [cb.body, cb.detail, cb.brake, cb.head]) {
        im.frustumCulled = false;
        im.count = 0;
        if (!im.instanceColor) {
          im.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(im.instanceMatrix.count * 3), 3);
        }
      }
    }
  }, [replay]);

  useFrame((state) => {
    const t = clock.time;
    const dt = replay.header.dt;
    const counts: Record<string, number> = {};
    const perClass = trackByClassInstance.current;
    for (const cls of CLASS_NAMES) {
      counts[cls] = 0;
      let arr = perClass.get(cls);
      if (!arr) {
        arr = [];
        perClass.set(cls, arr);
      }
      arr.length = 0;
    }
    const f = t / dt;
    const tracks = replay.tracks;
    const last = replay.lastFrame;
    const wobble = reducedMotion ? 0 : state.clock.elapsedTime;
    for (let k = 0; k < tracks.length; k++) {
      const tr = tracks[k]!;
      if (tr.first > f) break; // sorted by first frame
      if ((last[k] ?? 0) < f) continue;
      if (!samplePose(tr, dt, t, pose)) continue;
      const cls = tr.cls;
      const cb = buffers.current[cls];
      if (!cb) continue;
      const i = counts[cls]!;
      if (i >= CAPACITY[cls]) continue;
      counts[cls] = i + 1;
      perClass.get(cls)!.push(tr);
      const yaw = (-pose.angle * Math.PI) / 180;
      let y = 0;
      if (cls === 'pedestrian' && pose.speed > 0.1) y = 0.04 * Math.abs(Math.sin(wobble * 9 + hashId(tr.id) * 10));
      tmpP.set(pose.x, y, -pose.y);
      tmpQ.setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
      tmpM.compose(tmpP, tmpQ, tmpS);
      cb.body.setMatrixAt(i, tmpM);
      cb.detail.setMatrixAt(i, tmpM);
      let pos = handle.positions.get(tr.id);
      if (!pos) {
        pos = new THREE.Vector3();
        handle.positions.set(tr.id, pos);
      }
      pos.copy(tmpP);
      if (heat) {
        const w = Math.min(pose.wait / 120, 1);
        tmpC.setHSL(0.62 - 0.62 * w, 0.95, 0.5);
        cb.body.setColorAt(i, tmpC);
      } else {
        cb.body.setColorAt(i, colourCache.get(tr.id) ?? tmpC.set('#ffffff'));
      }
      if (cls !== 'pedestrian') {
        cb.brake.setMatrixAt(i, tmpM);
        cb.head.setMatrixAt(i, tmpM);
        const braking = pose.accel < -0.6 || (pose.speed < 0.1 && pose.wait < 2);
        cb.brake.setColorAt(i, braking ? tmpC.set(1, 1, 1) : night ? tmpC.set(0.35, 0.35, 0.35) : tmpC.set(0.25, 0.25, 0.25));
        cb.head.setColorAt(i, night ? tmpC.set(1, 1, 1) : BLACK);
      }
    }
    for (const cls of CLASS_NAMES) {
      const cb = buffers.current[cls];
      if (!cb) continue;
      const n = counts[cls] ?? 0;
      for (const im of [cb.body, cb.detail, cb.brake, cb.head]) {
        im.count = n;
        im.instanceMatrix.needsUpdate = true;
        if (im.instanceColor) im.instanceColor.needsUpdate = true;
      }
    }
  });

  const pick = (cls: ClassName) => (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    const i = e.instanceId;
    if (i === undefined) return;
    const tr = trackByClassInstance.current.get(cls)?.[i];
    if (tr) onPick(tr);
  };

  return (
    <group>
      {CLASS_NAMES.map((cls) => {
        const m: ClassModel = models[cls];
        const cap = CAPACITY[cls];
        return (
          <group key={cls}>
            <instancedMesh
              ref={(im) => {
                if (im) (buffers.current[cls] ??= { body: im, detail: im, brake: im, head: im, ids: [] }).body = im;
              }}
              args={[m.body, bodyMat, cap]}
              castShadow
              receiveShadow
              onClick={pick(cls)}
            />
            <instancedMesh
              ref={(im) => {
                if (im) (buffers.current[cls] ??= { body: im, detail: im, brake: im, head: im, ids: [] }).detail = im;
              }}
              args={[m.detail, detailMat, cap]}
              castShadow
              onClick={pick(cls)}
            />
            <instancedMesh
              ref={(im) => {
                if (im) (buffers.current[cls] ??= { body: im, detail: im, brake: im, head: im, ids: [] }).brake = im;
              }}
              args={[m.brake, lightMat, cap]}
            />
            <instancedMesh
              ref={(im) => {
                if (im) (buffers.current[cls] ??= { body: im, detail: im, brake: im, head: im, ids: [] }).head = im;
              }}
              args={[m.head, lightMat, cap]}
            />
          </group>
        );
      })}
    </group>
  );
}

/** Convenience for the HUD: current pose of a track (or null). */
export function trackPoseAt(replay: Replay, id: string, t: number): TrackPose | null {
  const tr = replay.tracks.find((x) => x.id === id);
  if (!tr) return null;
  const p: TrackPose = { x: 0, y: 0, angle: 0, speed: 0, wait: 0, accel: 0 };
  return samplePose(tr, replay.header.dt, t, p) ? p : null;
}

export function useReplayTime(): number {
  return useStore((s) => s.time);
}
