import { useState } from 'react';
import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { CONTROLLER_LABELS, EVAL, METRIC_LABELS, SCENARIO_LABELS, stat } from '../data/results';

const COLOURS: Record<string, string> = { fixed_time: '#f59e0b', max_pressure: '#60a5fa', ppo: '#34d399' };
const MAIN_CONTROLLERS = ['fixed_time', 'max_pressure', 'ppo'];
const CLASSES = ['car', 'motorcycle', 'cng', 'rickshaw', 'bus', 'truck'];

export function ResultsCharts() {
  const [metric, setMetric] = useState('mean_waiting_time');
  const controllers = MAIN_CONTROLLERS.filter((c) => EVAL.summary[c]);
  const scenarios = EVAL.meta.scenarios;
  const data = scenarios.map((s) => {
    const row: Record<string, number | string> = { scenario: SCENARIO_LABELS[s] ?? s };
    for (const c of controllers) {
      const st = stat(c, s, metric);
      if (st) {
        row[c] = st.mean;
        row[`${c}_err`] = Math.max(st.mean - st.ci_low, st.ci_high - st.mean);
      }
    }
    return row;
  });
  const scen = scenarios.includes('morning_peak') ? 'morning_peak' : (scenarios[0] ?? '');
  const classData = CLASSES.map((cls) => {
    const row: Record<string, number | string> = { cls };
    for (const c of controllers) {
      const st = stat(c, scen, `wait_${cls}`);
      if (st) {
        row[c] = st.mean;
        row[`${c}_err`] = Math.max(st.mean - st.ci_low, st.ci_high - st.mean);
      }
    }
    return row;
  });
  return (
    <section className="section" id="results">
      <h2>Results across scenarios</h2>
      <p className="muted">
        {EVAL.meta.episodes_per_scenario} episodes per scenario and controller, error bars are 95 % bootstrap confidence intervals of the mean. Noisy sensors = Gaussian noise plus 5 % dropped
        detector readings on top of the morning peak.
      </p>
      <label className="metric-pick">
        metric{' '}
        <select className="sel" value={metric} onChange={(e) => setMetric(e.target.value)}>
          {EVAL.meta.metrics.map((m) => (
            <option key={m} value={m}>
              {METRIC_LABELS[m] ?? m}
            </option>
          ))}
        </select>
      </label>
      <div className="chart" role="img" aria-label={`${METRIC_LABELS[metric] ?? metric} by scenario and controller`}>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.25)" />
            <XAxis dataKey="scenario" tick={{ fill: 'var(--fg)', fontSize: 12 }} />
            <YAxis tick={{ fill: 'var(--fg)', fontSize: 12 }} />
            <Tooltip contentStyle={{ background: 'var(--bg-2)', border: '1px solid var(--line)', color: 'var(--fg)' }} formatter={(v: number) => v.toFixed(1)} />
            <Legend />
            {controllers.map((c) => (
              <Bar key={c} dataKey={c} name={CONTROLLER_LABELS[c] ?? c} fill={COLOURS[c]} radius={[4, 4, 0, 0]}>
                <ErrorBar dataKey={`${c}_err`} width={4} strokeWidth={1.5} stroke="var(--fg)" />
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <h3>Fairness across vehicle classes · {SCENARIO_LABELS[scen] ?? scen}</h3>
      <p className="muted">Mean waiting time per vehicle class. Slow, wide cycle-rickshaws and buses are the classes most easily starved by a controller that only counts vehicles.</p>
      <div className="chart" role="img" aria-label="Mean waiting time per vehicle class and controller">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={classData} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.25)" />
            <XAxis dataKey="cls" tick={{ fill: 'var(--fg)', fontSize: 12 }} />
            <YAxis tick={{ fill: 'var(--fg)', fontSize: 12 }} />
            <Tooltip contentStyle={{ background: 'var(--bg-2)', border: '1px solid var(--line)', color: 'var(--fg)' }} formatter={(v: number) => `${v.toFixed(1)} s`} />
            <Legend />
            {controllers.map((c) => (
              <Bar key={c} dataKey={c} name={CONTROLLER_LABELS[c] ?? c} fill={COLOURS[c]} radius={[4, 4, 0, 0]}>
                <ErrorBar dataKey={`${c}_err`} width={4} strokeWidth={1.5} stroke="var(--fg)" />
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <details className="table-wrap">
        <summary>Full table (mean, 95 % CI, change vs fixed-time)</summary>
        <table>
          <thead>
            <tr>
              <th>controller</th>
              <th>scenario</th>
              <th>wait (s)</th>
              <th>p95 travel (s)</th>
              <th>queue (veh)</th>
              <th>throughput</th>
              <th>ped wait (s)</th>
              <th>shield</th>
              <th>switches/h</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(EVAL.summary).map(([c, scens]) =>
              Object.entries(scens).map(([s, e]) => (
                <tr key={`${c}-${s}`}>
                  <td>{CONTROLLER_LABELS[c] ?? c}</td>
                  <td>{SCENARIO_LABELS[s] ?? s}</td>
                  {['mean_waiting_time', 'p95_travel_time', 'mean_queue_length', 'throughput', 'ped_mean_wait', 'shield_interventions', 'phase_switches_per_hour'].map((m) => {
                    const st = e[m];
                    if (typeof st !== 'object') return <td key={m}>–</td>;
                    return (
                      <td key={m}>
                        {st.mean.toFixed(1)} <small>[{st.ci_low.toFixed(1)}, {st.ci_high.toFixed(1)}]</small>
                        {st.pct_vs_fixed !== undefined && <small className={st.pct_vs_fixed < 0 ? 'neg' : 'pos'}> {st.pct_vs_fixed > 0 ? '+' : ''}{st.pct_vs_fixed.toFixed(0)}%</small>}
                      </td>
                    );
                  })}
                </tr>
              )),
            )}
          </tbody>
        </table>
      </details>
    </section>
  );
}
