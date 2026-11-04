import { useCallback, useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { Geometry } from '../replay/geometry';
import type { LoadedReplay, Track } from '../replay/types';
import { CameraRig } from './CameraRig';
import { City } from './City';
import { Lighting, NightBloom, Rain } from './Environment';
import { Roads } from './Roads';
import { Signals } from './Signals';
import { Vehicles, type VehiclesHandle } from './Vehicles';
import { clock, useStore, type CameraPreset } from '../store/playback';

/** Which lane a stopped vehicle is on: inbound vehicle lanes are axis-aligned straight ribbons. */
function laneQueueCounter(geometry: Geometry) {
  const lanes = geometry.lanes.filter((l) => l.function === 'normal' && !l.pedestrian && geometry.phases[l.edge.split('_')[1] ?? '']?.incoming_lanes.includes(l.id));
  const boxes = lanes.map((l) => {
    const xs = l.shape.map((p) => p[0]);
    const ys = l.shape.map((p) => p[1]);
    const hw = l.width / 2 + 0.3;
    return { id: l.id, x0: Math.min(...xs) - hw, x1: Math.max(...xs) + hw, y0: Math.min(...ys) - hw, y1: Math.max(...ys) + hw };
  });
  return (replay: LoadedReplay, t: number): Map<string, number> => {
    const out = new Map<string, number>();
    const f = t / replay.replay.header.dt;
    const tracks = replay.replay.tracks;
    for (let k = 0; k < tracks.length; k++) {
      const tr = tracks[k]!;
      if (tr.first > f) break;
      const i = Math.floor(f) - tr.first;
      if (i < 0 || i >= tr.n || tr.cls === 'pedestrian') continue;
      if ((tr.speed[i] ?? 1) > 0.3) continue;
      const x = tr.x[i] ?? 0;
      const y = tr.y[i] ?? 0;
      for (const b of boxes) {
        if (x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1) {
          out.set(b.id, (out.get(b.id) ?? 0) + 1);
          break;
        }
      }
    }
    return out;
  };
}

export function Scene({
  geometry,
  loaded,
  primary,
  domElement,
}: {
  geometry: Geometry;
  loaded: LoadedReplay;
  primary: boolean;
  domElement: HTMLElement | null;
}) {
  const night = useStore((s) => s.night);
  const rain = useStore((s) => s.rain);
  const heat = useStore((s) => s.colourMode === 'heat');
  const heatmap = useStore((s) => s.heatmap);
  const camera = useStore((s) => s.camera);
  const follow = useStore((s) => s.follow);
  const reducedMotion = useStore((s) => s.reducedMotion);
  const viewMode = useStore((s) => s.viewMode);
  const set = useStore((s) => s.set);
  const handle = useRef<VehiclesHandle>({ positions: new Map() }).current;
  const centre = useMemo<[number, number]>(() => {
    const c = Object.values(geometry.junction_centres);
    const cx = c.reduce((a, p) => a + p[0], 0) / c.length;
    const cy = c.reduce((a, p) => a + p[1], 0) / c.length;
    return [cx, -cy];
  }, [geometry]);
  const replay = loaded.replay;
  const jIndex = useMemo(() => new Map(replay.header.junctions.map((j, i) => [j, i])), [replay]);
  const getState = useCallback(
    (junction: string) => {
      const f = Math.min(Math.max(Math.floor(clock.time / replay.header.dt), 0), replay.header.n_frames - 1);
      const nj = replay.header.junctions.length;
      const idx = replay.tls[f * nj + (jIndex.get(junction) ?? 0)] ?? 0;
      return replay.header.tls_states[idx] ?? '';
    },
    [replay, jIndex],
  );
  const counter = useMemo(() => laneQueueCounter(geometry), [geometry]);
  // the lane heatmap is React state updated at 2 Hz by the Viewer; here we only compute when asked
  const queueByLane = useStore((s) => s.time) && heatmap ? counter(loaded, clock.time) : null;
  const onPick = useCallback(
    (tr: Track) => {
      set({ follow: { id: tr.id, cls: tr.cls, speed: 0, wait: 0 }, camera: 'follow' as CameraPreset });
    },
    [set],
  );
  const followPos = useCallback(() => (follow ? (handle.positions.get(follow.id) ?? null) : null), [follow, handle]);
  const followHeading = useCallback(() => {
    if (!follow) return 0;
    const tr = replay.tracks.find((x) => x.id === follow.id);
    if (!tr) return 0;
    const i = Math.min(Math.max(Math.floor(clock.time / replay.header.dt) - tr.first, 0), tr.n - 1);
    return tr.angle[i] ?? 0;
  }, [follow, replay]);

  return (
    <>
      <CameraRig primary={primary} preset={camera} centre={centre} followPos={primary ? followPos : null} followHeading={primary ? followHeading : null} reducedMotion={reducedMotion} domElement={domElement} />
      <Lighting night={night} centre={centre} />
      <Roads geometry={geometry} night={night} queueByLane={queueByLane} />
      <City geometry={geometry} night={night} />
      <Signals geometry={geometry} getState={getState} night={night} />
      <Vehicles replay={replay} night={night} heat={heat} onPick={onPick} handle={handle} reducedMotion={reducedMotion} />
      {rain && <Rain centre={centre} reducedMotion={reducedMotion} />}
      {/* post-processing cannot be scissored per pane, so bloom is only used in single-pane mode */}
      <NightBloom enabled={night && viewMode === 'single'} />
      <group position={[centre[0], 0, centre[1]]}>
        <mesh visible={false}>
          <boxGeometry args={[1, 1, 1]} />
          <meshBasicMaterial color={new THREE.Color('#000')} />
        </mesh>
      </group>
    </>
  );
}
