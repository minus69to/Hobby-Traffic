import { useMemo } from 'react';
import { useStore } from '../store/playback';
import type { LoadedReplay, PerSecondRow } from '../replay/types';
import { Sparkline } from './Sparkline';
import { trackPoseAt } from '../scene/Vehicles';

function rowAt(l: LoadedReplay, t: number): PerSecondRow | null {
  const rows = l.manifest.per_second[l.replay.header.junctions[0] ?? 'C'];
  if (!rows) return null;
  return rows[Math.min(Math.max(Math.floor(t) - 1, 0), rows.length - 1)] ?? null;
}

function phaseInfo(l: LoadedReplay, t: number, phases: { name: string; state: string }[]): { name: string; kind: string; secs: number } {
  const h = l.replay.header;
  const f = Math.min(Math.max(Math.floor(t / h.dt), 0), h.n_frames - 1);
  const nj = h.junctions.length;
  const idx = l.replay.tls[f * nj] ?? 0;
  const state = h.tls_states[idx] ?? '';
  let k = f;
  while (k > 0 && (l.replay.tls[(k - 1) * nj] ?? -1) === idx) k--;
  const secs = (f - k) * h.dt;
  const p = phases.find((x) => x.state === state);
  const kind = state.includes('y') ? 'yellow' : !state.includes('G') && !state.includes('g') ? 'all-red' : 'green';
  return { name: p ? p.name.replace('_', ' ') : kind === 'green' ? 'custom' : kind, kind, secs };
}

function interventionsUpTo(l: LoadedReplay, t: number): number {
  let n = 0;
  for (const list of Object.values(l.manifest.interventions)) for (const iv of list) if (iv.time <= t) n++;
  return n;
}

function Panel({ l, t, side, phases }: { l: LoadedReplay; t: number; side: 'left' | 'right'; phases: { name: string; state: string }[] }) {
  const row = rowAt(l, t);
  const ph = phaseInfo(l, t, phases);
  const iv = interventionsUpTo(l, t);
  const arms = ['N', 'E', 'S', 'W'];
  return (
    <div className={`hud-panel ${side}`} aria-live="off">
      <div className="hud-title">{l.manifest.controller === 'fixed_time' ? 'Fixed-time' : l.manifest.controller === 'max_pressure' ? 'Max-pressure' : 'Safe RL (PPO)'}</div>
      <div className="hud-grid">
        <div>
          <span className="hud-k">waiting</span>
          <span className="hud-v">{row ? Math.round(row.vehicles_waiting) : '–'}</span>
        </div>
        <div>
          <span className="hud-k">avg wait</span>
          <span className="hud-v">{row ? `${row.avg_wait.toFixed(0)} s` : '–'}</span>
        </div>
        <div>
          <span className="hud-k">peds waiting</span>
          <span className="hud-v">{row ? row.ped_waiting : '–'}</span>
        </div>
        <div>
          <span className="hud-k">phase</span>
          <span className={`hud-v phase-${ph.kind}`}>
            {ph.name} <small>{ph.secs.toFixed(0)}s</small>
          </span>
        </div>
      </div>
      <div className="queues" aria-label="Queue per approach">
        {arms.map((a) => {
          const q = row?.queue[a] ?? 0;
          return (
            <div key={a} className="queue">
              <span className="queue-arm">{a}</span>
              <div className="queue-bar">
                <div className="queue-fill" style={{ width: `${Math.min(100, (q / 120) * 100)}%` }} />
              </div>
              <span className="queue-n">{Math.round(q)}</span>
            </div>
          );
        })}
      </div>
      <div className={`badge ${iv > 0 ? 'warn' : 'ok'}`} title="Times the safety shield rejected, ignored or forced a phase request so far">
        shield interventions: {iv}
      </div>
    </div>
  );
}

export function HUD() {
  const t = useStore((s) => s.time);
  const left = useStore((s) => s.left);
  const right = useStore((s) => s.right);
  const geometry = useStore((s) => s.geometry);
  const follow = useStore((s) => s.follow);
  const viewMode = useStore((s) => s.viewMode);
  const activeSide = useStore((s) => s.activeSide);
  const phases = useMemo(() => geometry?.phases[Object.keys(geometry.phases)[0] ?? 'C']?.phases ?? [], [geometry]);
  const series = useMemo(() => {
    if (!left || !right) return null;
    const j = left.replay.header.junctions[0] ?? 'C';
    const a = left.manifest.per_second[j] ?? [];
    const b = right.manifest.per_second[j] ?? [];
    return { waitA: a.map((r) => r.avg_wait), waitB: b.map((r) => r.avg_wait), qA: a.map((r) => r.vehicles_waiting), qB: b.map((r) => r.vehicles_waiting) };
  }, [left, right]);
  if (!left || !right) return null;
  const followPose = follow ? trackPoseAt((follow.side === 'right' ? right : left).replay, follow.id, t) : null;
  const showLeft = viewMode === 'split' || activeSide === 'left';
  const showRight = viewMode === 'split' || activeSide === 'right';
  return (
    <div className="hud" data-testid="hud">
      {showLeft && <Panel l={left} t={t} side="left" phases={phases} />}
      {showRight && <Panel l={right} t={t} side="right" phases={phases} />}
      {series && (
        <div className="hud-charts">
          <Sparkline a={series.waitA} b={series.waitB} upTo={Math.floor(t)} labelA="fixed" labelB="RL" unit="s" />
          <Sparkline a={series.qA} b={series.qB} upTo={Math.floor(t)} labelA="fixed" labelB="RL" unit=" veh" />
        </div>
      )}
      {follow && (
        <div className="hud-follow">
          <strong>{follow.cls.replace('_', ' ')}</strong> {follow.id}
          {followPose ? (
            <span>
              {' '}
              · {(followPose.speed * 3.6).toFixed(0)} km/h · waiting {followPose.wait.toFixed(0)} s
            </span>
          ) : (
            <span> · left the network</span>
          )}
        </div>
      )}
    </div>
  );
}
