import { useEffect } from 'react';
import { clock, useStore, type CameraPreset } from '../store/playback';

const SPEEDS = [0.5, 1, 2, 4, 8];

export function Controls({ scenarios, controllers }: { scenarios: string[]; controllers: string[] }) {
  const s = useStore();
  const { set } = s;
  const duration = s.left ? s.left.replay.header.n_frames * s.left.replay.header.dt : 3600;
  const incident = s.left?.manifest.incident ?? null;

  useEffect(() => {
    clock.playing = s.playing;
    clock.speed = s.speed;
  }, [s.playing, s.speed]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT' || (e.target as HTMLElement)?.tagName === 'SELECT') return;
      switch (e.key) {
        case ' ':
          e.preventDefault();
          set({ playing: !useStore.getState().playing });
          break;
        case 'ArrowLeft':
          clock.time = Math.max(0, clock.time - 30);
          set({ time: clock.time });
          break;
        case 'ArrowRight':
          clock.time = Math.min(duration, clock.time + 30);
          set({ time: clock.time });
          break;
        case '1':
        case '2':
        case '3':
          set({ camera: (['aerial', 'top', 'street'] as CameraPreset[])[Number(e.key) - 1], follow: null });
          break;
        case 'n':
          set({ night: !useStore.getState().night });
          break;
        case 'h':
          set({ colourMode: useStore.getState().colourMode === 'heat' ? 'realistic' : 'heat' });
          break;
        case 'r':
          set({ rain: !useStore.getState().rain });
          break;
        case 'Escape':
          set({ follow: null, camera: 'aerial', explorer: false });
          break;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [set, duration]);

  const seek = (t: number) => {
    clock.time = t;
    set({ time: t });
  };

  return (
    <div className="controls" data-testid="controls">
      <div className="row">
        <button className="btn" onClick={() => set({ playing: !s.playing })} aria-label={s.playing ? 'Pause' : 'Play'}>
          {s.playing ? '❚❚' : '▶'}
        </button>
        <select className="sel" value={s.speed} onChange={(e) => set({ speed: Number(e.target.value) })} aria-label="Playback speed">
          {SPEEDS.map((v) => (
            <option key={v} value={v}>
              {v}×
            </option>
          ))}
        </select>
        <div className="timeline">
          <input
            type="range"
            min={0}
            max={duration}
            step={1}
            value={Math.floor(s.time)}
            onChange={(e) => seek(Number(e.target.value))}
            aria-label="Timeline"
            style={{ width: '100%' }}
          />
          {incident && (
            <div
              className="marker"
              style={{ left: `${(incident.start / duration) * 100}%`, width: `${(incident.duration / duration) * 100}%` }}
              title={`Incident: lane ${incident.lane} of the ${incident.edge_suffix} approach blocked for ${incident.duration} s`}
            />
          )}
          <div className="marker warm" style={{ left: 0, width: `${((s.left?.manifest.warmup_seconds ?? 300) / duration) * 100}%` }} title="Warm-up (excluded from metrics)" />
        </div>
        <span className="time">{formatTime(s.time)}</span>
      </div>
      <div className="row wrap">
        <label>
          scenario
          <select className="sel" value={s.scenario} onChange={(e) => set({ scenario: e.target.value })}>
            {scenarios.map((x) => (
              <option key={x} value={x}>
                {x.replace('_', ' ')}
              </option>
            ))}
          </select>
        </label>
        <label>
          left
          <select className="sel" value={s.leftController} onChange={(e) => set({ leftController: e.target.value })}>
            {controllers.map((x) => (
              <option key={x} value={x}>
                {x}
              </option>
            ))}
          </select>
        </label>
        <label>
          right
          <select className="sel" value={s.rightController} onChange={(e) => set({ rightController: e.target.value })}>
            {controllers.map((x) => (
              <option key={x} value={x}>
                {x}
              </option>
            ))}
          </select>
        </label>
        <div className="seg" role="group" aria-label="Camera preset">
          {(['aerial', 'top', 'street'] as CameraPreset[]).map((c) => (
            <button key={c} className={`btn ${s.camera === c ? 'on' : ''}`} onClick={() => set({ camera: c, follow: null })}>
              {c}
            </button>
          ))}
          <button className={`btn ${s.camera === 'follow' ? 'on' : ''}`} disabled={!s.follow} title="Click a vehicle to follow it" onClick={() => s.follow && set({ camera: 'follow' })}>
            follow
          </button>
        </div>
        <button className={`btn ${s.night ? 'on' : ''}`} onClick={() => set({ night: !s.night })}>
          {s.night ? 'night' : 'golden hour'}
        </button>
        <button className={`btn ${s.rain ? 'on' : ''}`} onClick={() => set({ rain: !s.rain })}>
          rain
        </button>
        <button className={`btn ${s.colourMode === 'heat' ? 'on' : ''}`} onClick={() => set({ colourMode: s.colourMode === 'heat' ? 'realistic' : 'heat' })} title="Colour vehicles by waiting time">
          heat colours
        </button>
        <button className={`btn ${s.heatmap ? 'on' : ''}`} onClick={() => set({ heatmap: !s.heatmap })} title="Lane overlay of queue density">
          lane heatmap
        </button>
        <button className={`btn ${s.viewMode === 'single' ? 'on' : ''}`} onClick={() => set({ viewMode: s.viewMode === 'split' ? 'single' : 'split' })}>
          {s.viewMode === 'split' ? 'split' : 'single'}
        </button>
        {s.viewMode === 'single' && (
          <button className="btn" onClick={() => set({ activeSide: s.activeSide === 'left' ? 'right' : 'left' })}>
            show {s.activeSide === 'left' ? 'right' : 'left'}
          </button>
        )}
        <button className={`btn ${s.reducedMotion ? 'on' : ''}`} onClick={() => set({ reducedMotion: !s.reducedMotion })} title="Disable camera animation, bobbing and rain motion">
          reduced motion
        </button>
      </div>
      <div className="hint">keys: space play/pause · ←/→ scrub 30 s · 1/2/3 cameras · n night · h heat · r rain · esc reset · click a vehicle to follow</div>
    </div>
  );
}

function formatTime(t: number): string {
  const m = Math.floor(t / 60);
  const sec = Math.floor(t % 60);
  return `${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
}
