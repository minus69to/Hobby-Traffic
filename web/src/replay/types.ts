/** Types shared by the replay decoder, scene and HUD. */

export const CLASS_NAMES = ['car', 'bus', 'bus_dd', 'truck', 'motorcycle', 'cng', 'rickshaw', 'pedestrian'] as const;
export type ClassName = (typeof CLASS_NAMES)[number];

export interface Track {
  id: string;
  cls: ClassName;
  first: number; // first frame index
  n: number; // number of samples
  x: Float32Array; // metres, SUMO coordinates
  y: Float32Array;
  angle: Float32Array; // degrees, 0 = north, clockwise
  speed: Float32Array; // m/s
  wait: Float32Array; // consecutive seconds with speed < 0.1 (derived at load)
}

export interface ReplayHeader {
  network: string;
  scenario: string;
  controller: string;
  seed: number;
  episode_seconds: number;
  warmup_seconds: number;
  step_length: number;
  dt: number;
  n_frames: number;
  quant: number;
  speed_quant: number;
  classes: string[];
  junctions: string[];
  tls_states: string[];
  n_pos: number;
}

export interface Replay {
  header: ReplayHeader;
  tracks: Track[];
  /** per frame, per junction: index into header.tls_states */
  tls: Uint8Array;
  /** tracks sorted by first frame, with lastFrame = first + n - 1 */
  lastFrame: Int32Array;
}

export interface PerSecondRow {
  t: number;
  queue: Record<string, number>;
  vehicles_waiting: number;
  avg_wait: number;
  vehicles: number;
  ped_waiting: number;
}

export interface Intervention {
  time: number;
  kind: string;
  requested: number | null;
  applied: number;
  reason: string;
}

export interface ReplayManifest {
  network: string;
  scenario: string;
  controller: string;
  seed: number;
  episode_seconds: number;
  warmup_seconds: number;
  step_length: number;
  file: string;
  size_bytes: number;
  n_frames: number;
  n_tracks: number;
  incident: { edge_suffix: string; lane: number; start: number; duration: number; end_pos: number } | null;
  arms_vph: Record<string, number> | null;
  metrics: Record<string, number>;
  shield_stats: Record<string, Record<string, number>>;
  interventions: Record<string, Intervention[]>;
  per_second: Record<string, PerSecondRow[]>;
  note: string;
}

export interface ReplayIndexEntry {
  network: string;
  scenario: string;
  controller: string;
  seed: number;
  file: string;
  size_bytes: number;
  n_frames: number;
  n_tracks: number;
}

export interface LoadedReplay {
  replay: Replay;
  manifest: ReplayManifest;
}
