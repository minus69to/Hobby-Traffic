import { useEffect, useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { View } from '@react-three/drei';
import { Scene } from './Scene';
import { clock, useStore } from '../store/playback';
import { loadGeometry, loadIndex, loadReplay } from '../replay/loader';

declare global {
  interface Window {
    __drs?: { clock: typeof clock; store: typeof useStore; frames: number };
  }
}

/** Advances the shared clock from the render loop and mirrors it to React at ~5 Hz. */
function Clock() {
  const acc = useRef(0);
  useFrame((_, delta) => {
    if (window.__drs) window.__drs.frames += 1;
    if (clock.playing) {
      clock.time = Math.min(clock.duration, clock.time + delta * clock.speed);
      if (clock.time >= clock.duration) clock.time = 0;
    }
    acc.current += delta;
    if (acc.current > 0.2) {
      acc.current = 0;
      useStore.getState().tickTime(clock.time);
    }
  });
  return null;
}

export function Viewer({ className }: { className?: string }) {
  const container = useRef<HTMLDivElement>(null);
  const leftRef = useRef<HTMLDivElement>(null!);
  const rightRef = useRef<HTMLDivElement>(null!);
  const geometry = useStore((s) => s.geometry);
  const left = useStore((s) => s.left);
  const right = useStore((s) => s.right);
  const scenario = useStore((s) => s.scenario);
  const leftController = useStore((s) => s.leftController);
  const rightController = useStore((s) => s.rightController);
  const viewMode = useStore((s) => s.viewMode);
  const activeSide = useStore((s) => s.activeSide);
  const divider = useStore((s) => s.divider);
  const loading = useStore((s) => s.loading);
  const error = useStore((s) => s.error);
  const set = useStore((s) => s.set);
  const [dragging, setDragging] = useState(false);
  const [el, setEl] = useState<HTMLDivElement | null>(null);

  useEffect(() => {
    setEl(container.current);
    // hooks for the headless media-capture script (scripts/capture_media.py)
    window.__drs = { clock, store: useStore, frames: 0 };
  }, []);

  useEffect(() => {
    let cancelled = false;
    set({ loading: { progress: 0, label: 'Loading network geometry' }, error: null });
    (async () => {
      try {
        const index = await loadIndex();
        const available = index.filter((r) => r.scenario === scenario).map((r) => r.controller);
        const allScenarios = Array.from(new Set(index.map((r) => r.scenario)));
        if (index.length === 0) throw new Error('no replays exported yet (run make replay)');
        if (!allScenarios.includes(scenario)) {
          set({ scenario: allScenarios[0] ?? scenario });
          return;
        }
        const lc = available.includes(leftController) ? leftController : (available[0] ?? leftController);
        const rc = available.includes(rightController) ? rightController : (available.find((c) => c.startsWith('ppo')) ?? available[available.length - 1] ?? rightController);
        if (lc !== leftController || rc !== rightController) {
          set({ leftController: lc, rightController: rc });
          return;
        }
        const geo = await loadGeometry('single');
        if (cancelled) return;
        set({ geometry: geo, loading: { progress: 0.05, label: `Loading ${leftController} replay` } });
        const l = await loadReplay('single', scenario, leftController, (p) => set({ loading: { progress: 0.05 + p * 0.45, label: `Loading ${leftController} replay` } }));
        if (cancelled) return;
        set({ left: l, loading: { progress: 0.5, label: `Loading ${rightController} replay` } });
        const r = await loadReplay('single', scenario, rightController, (p) => set({ loading: { progress: 0.5 + p * 0.5, label: `Loading ${rightController} replay` } }));
        if (cancelled) return;
        clock.duration = l.replay.header.n_frames * l.replay.header.dt;
        if (clock.time > clock.duration) clock.time = 0;
        set({ right: r, loading: null });
      } catch (e) {
        if (!cancelled) set({ error: (e as Error).message, loading: null });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [scenario, leftController, rightController, set]);

  useEffect(() => {
    if (!dragging) return;
    const move = (e: PointerEvent) => {
      const rect = container.current?.getBoundingClientRect();
      if (!rect) return;
      set({ divider: Math.min(0.85, Math.max(0.15, (e.clientX - rect.left) / rect.width)) });
    };
    const up = () => setDragging(false);
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    return () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
  }, [dragging, set]);

  const ready = geometry && left && right;
  const split = viewMode === 'split';
  const leftW = split ? `${divider * 100}%` : activeSide === 'left' ? '100%' : '0%';
  const rightW = split ? `${(1 - divider) * 100}%` : activeSide === 'right' ? '100%' : '0%';

  return (
    <div ref={container} className={`viewer ${className ?? ''}`} data-testid="viewer">
      <div ref={leftRef} className="pane" style={{ width: leftW }}>
        <div className="pane-label">{labelFor(leftController)}</div>
      </div>
      <div ref={rightRef} className="pane" style={{ width: rightW }}>
        <div className="pane-label right">{labelFor(rightController)}</div>
      </div>
      {split && (
        <div
          className={`divider ${dragging ? 'dragging' : ''}`}
          style={{ left: `calc(${divider * 100}% - 6px)` }}
          role="separator"
          aria-label="Drag to resize the split view"
          aria-valuenow={Math.round(divider * 100)}
          tabIndex={0}
          onPointerDown={() => setDragging(true)}
          onKeyDown={(e) => {
            if (e.key === 'ArrowLeft') set({ divider: Math.max(0.15, divider - 0.05) });
            if (e.key === 'ArrowRight') set({ divider: Math.min(0.85, divider + 0.05) });
          }}
        />
      )}
      {ready && el && (
        <Canvas
          className="canvas"
          eventSource={el}
          eventPrefix="client"
          shadows
          dpr={[1, 1.5]}
          gl={{ antialias: true, powerPreference: 'high-performance' }}
          frameloop="always"
          style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}
        >
          <Clock />
          <View track={leftRef} index={1}>
            <Scene geometry={geometry} loaded={left} primary={!split && activeSide === 'left' ? true : split} side="left" domElement={el} />
          </View>
          <View track={rightRef} index={2}>
            <Scene geometry={geometry} loaded={right} primary={!split && activeSide === 'right'} side="right" domElement={el} />
          </View>
        </Canvas>
      )}
      {(loading || error) && (
        <div className="loading" role="status">
          {error ? (
            <div className="error">Could not load the replay: {error}</div>
          ) : (
            <>
              <div className="loading-label">{loading?.label}</div>
              <progress value={loading?.progress ?? 0} max={1} />
            </>
          )}
        </div>
      )}
    </div>
  );
}

function labelFor(ctrl: string): string {
  if (ctrl === 'fixed_time') return 'Fixed-time (Webster)';
  if (ctrl === 'max_pressure') return 'Max-pressure';
  if (ctrl.startsWith('ppo')) return `Safe RL (PPO${ctrl.includes('_s') ? ', seed ' + ctrl.split('_s')[1] : ''})`;
  return ctrl;
}
