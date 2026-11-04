import { create } from 'zustand';
import type { Geometry } from '../replay/geometry';
import type { LoadedReplay } from '../replay/types';

export type CameraPreset = 'aerial' | 'top' | 'street' | 'follow';
export type ColourMode = 'realistic' | 'heat';
export type ViewMode = 'split' | 'single';

export interface FollowInfo {
  id: string;
  cls: string;
  speed: number;
  wait: number;
  side: 'left' | 'right';
}

export interface PlaybackState {
  // data
  geometry: Geometry | null;
  left: LoadedReplay | null;
  right: LoadedReplay | null;
  loading: { progress: number; label: string } | null;
  error: string | null;
  scenario: string;
  leftController: string;
  rightController: string;
  // playback (time is mutated through `tickTime` from the render loop; subscribe with care)
  time: number;
  playing: boolean;
  speed: number;
  // look
  camera: CameraPreset;
  night: boolean;
  rain: boolean;
  colourMode: ColourMode;
  heatmap: boolean;
  reducedMotion: boolean;
  viewMode: ViewMode;
  activeSide: 'left' | 'right';
  divider: number; // 0..1 fraction for the left pane
  follow: FollowInfo | null;
  explorer: boolean;
  // actions
  set: (p: Partial<PlaybackState>) => void;
  tickTime: (t: number) => void;
  togglePlay: () => void;
  seek: (t: number) => void;
}

export const useStore = create<PlaybackState>((set, get) => ({
  geometry: null,
  left: null,
  right: null,
  loading: null,
  error: null,
  scenario: 'morning_peak',
  leftController: 'fixed_time',
  rightController: 'ppo_s13',
  time: 300,
  playing: true,
  speed: 2,
  camera: 'aerial',
  night: false,
  rain: false,
  colourMode: 'realistic',
  heatmap: false,
  reducedMotion: typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  viewMode: typeof window !== 'undefined' && window.innerWidth < 760 ? 'single' : 'split',
  activeSide: 'right',
  divider: 0.5,
  follow: null,
  explorer: false,
  set: (p) => set(p),
  tickTime: (t) => set({ time: t }),
  togglePlay: () => set({ playing: !get().playing }),
  seek: (t) => set({ time: t }),
}));

/** Mutable clock shared with the render loop so the scene never re-renders React on every frame. */
export const clock = { time: 300, playing: true, speed: 2, duration: 3600 };
