/** Geometry exported from the SUMO network by scripts/export_geometry.py. */

export interface Lane {
  id: string;
  edge: string;
  function: 'normal' | 'internal' | 'crossing' | 'walkingarea';
  pedestrian: boolean;
  width: number;
  length: number;
  shape: [number, number][];
}

export interface JunctionShape {
  id: string;
  type: string;
  x: number;
  y: number;
  shape: [number, number][];
}

export interface SignalHead {
  junction: string;
  arm: 'N' | 'E' | 'S' | 'W';
  stop_line: [number, number][];
  centre: [number, number];
  heading_deg: number;
  vehicle_links: Record<string, number>;
  crossing_link: number | null;
  crossing_shape: [number, number][];
  inbound_lanes: string[];
}

export interface PhaseInfo {
  index: number;
  name: string;
  state: string;
  has_walk: boolean;
}

export interface PhaseTable {
  n_links: number;
  phases: PhaseInfo[];
  links: { index: number; from: string; to: string; arm: string; turn: string }[];
  incoming_lanes: string[];
  lane_phases: Record<string, number[]>;
}

export interface Geometry {
  network: string;
  net_offset: [number, number];
  bounds: [number, number, number, number];
  junction_centres: Record<string, [number, number]>;
  arms: Record<string, Record<string, { inbound: string; outbound: string; external: boolean }>>;
  directions: string[];
  lanes: Lane[];
  junctions: JunctionShape[];
  signals: SignalHead[];
  phases: Record<string, PhaseTable>;
  lefthand: boolean;
}

export async function fetchGeometry(url: string): Promise<Geometry> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to fetch ${url}`);
  return (await res.json()) as Geometry;
}

/** Map SUMO (x east, y north) to three.js (x east, z south). */
export function toWorld(x: number, y: number): [number, number] {
  return [x, -y];
}
