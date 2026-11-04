import { fetchReplay } from './decode';
import { fetchGeometry, type Geometry } from './geometry';
import type { LoadedReplay, ReplayIndexEntry, ReplayManifest } from './types';

const BASE = `${import.meta.env.BASE_URL}data/`;
const cache = new Map<string, Promise<LoadedReplay>>();
const geometryCache = new Map<string, Promise<Geometry>>();
let indexCache: Promise<ReplayIndexEntry[]> | null = null;

export function loadGeometry(network = 'single'): Promise<Geometry> {
  let p = geometryCache.get(network);
  if (!p) {
    p = fetchGeometry(`${BASE}geometry_${network}.json`);
    geometryCache.set(network, p);
  }
  return p;
}

export function loadIndex(): Promise<ReplayIndexEntry[]> {
  indexCache ??= fetch(`${BASE}replays/index.json`)
    .then((r) => (r.ok ? r.json() : { replays: [] }))
    .then((j: { replays: ReplayIndexEntry[] }) => j.replays);
  return indexCache;
}

export function loadReplay(network: string, scenario: string, controller: string, onProgress?: (p: number) => void): Promise<LoadedReplay> {
  const name = `${network}_${scenario}_${controller}`;
  let p = cache.get(name);
  if (!p) {
    p = (async () => {
      const manifest = (await fetch(`${BASE}replays/${name}.json`).then((r) => {
        if (!r.ok) throw new Error(`replay ${name} not found`);
        return r.json();
      })) as ReplayManifest;
      const replay = await fetchReplay(`${BASE}replays/${manifest.file}`, (loaded, total) => onProgress?.(total ? loaded / total : 0));
      return { replay, manifest };
    })();
    cache.set(name, p);
    p.catch(() => cache.delete(name));
  }
  return p;
}
