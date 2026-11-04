/** Tiny SVG line chart comparing two series up to the current time. */
export function Sparkline({
  a,
  b,
  upTo,
  max,
  labelA,
  labelB,
  unit,
  width = 220,
  height = 56,
}: {
  a: number[];
  b: number[];
  upTo: number;
  max?: number;
  labelA: string;
  labelB: string;
  unit: string;
  width?: number;
  height?: number;
}) {
  const n = Math.max(a.length, b.length, 1);
  const lim = max ?? Math.max(1, ...a, ...b);
  const path = (s: number[]) => {
    const pts: string[] = [];
    const end = Math.min(upTo, s.length);
    const step = Math.max(1, Math.floor(end / width));
    for (let i = 0; i < end; i += step) {
      const x = (i / n) * width;
      const y = height - 4 - (Math.min(s[i] ?? 0, lim) / lim) * (height - 8);
      pts.push(`${pts.length ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`);
    }
    return pts.join(' ');
  };
  const cur = (s: number[]) => s[Math.min(upTo, s.length) - 1] ?? 0;
  return (
    <div className="spark">
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${labelA} vs ${labelB}`}>
        <path d={path(a)} fill="none" stroke="var(--c-fixed)" strokeWidth={1.6} />
        <path d={path(b)} fill="none" stroke="var(--c-rl)" strokeWidth={1.6} />
        <line x1={(upTo / n) * width} x2={(upTo / n) * width} y1={0} y2={height} stroke="rgba(255,255,255,0.3)" strokeDasharray="2 3" />
      </svg>
      <div className="spark-legend">
        <span style={{ color: 'var(--c-fixed)' }}>
          {labelA} {cur(a).toFixed(0)}
          {unit}
        </span>
        <span style={{ color: 'var(--c-rl)' }}>
          {labelB} {cur(b).toFixed(0)}
          {unit}
        </span>
      </div>
    </div>
  );
}
