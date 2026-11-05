import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { FORECAST, IMITATION, fmt } from '../data/results';

type Err = { mae: number; mae_ci: [number, number]; rmse: number; rmse_ci: [number, number] };

export function ForecastingCard() {
  const models = Object.entries(FORECAST.models);
  const ready = models.length > 0;
  return (
    <div className="card wide">
      <div className="card-title">Forecasting (supervised DL)</div>
      {ready ? (
        <>
          <p className="card-sub">
            Per-lane queue 30 s and 60 s ahead from the last {FORECAST.history_s} s of all lanes and the signal phase. MAE in vehicles with 95 % bootstrap CIs over held-out episodes
            ({FORECAST.n_test_windows?.toLocaleString()} windows).
          </p>
          <table>
            <thead>
              <tr>
                <th>model</th>
                <th>MAE 30 s</th>
                <th>MAE 60 s</th>
                <th>RMSE 60 s</th>
              </tr>
            </thead>
            <tbody>
              {models.map(([name, r]) => {
                const h30 = r['h30'] as Err | undefined;
                const h60 = r['h60'] as Err | undefined;
                return (
                  <tr key={name}>
                    <td>{name.replace('_', ' ')}</td>
                    <td>
                      {fmt(h30?.mae, 2)} <small>[{fmt(h30?.mae_ci[0], 2)}, {fmt(h30?.mae_ci[1], 2)}]</small>
                    </td>
                    <td>
                      {fmt(h60?.mae, 2)} <small>[{fmt(h60?.mae_ci[0], 2)}, {fmt(h60?.mae_ci[1], 2)}]</small>
                    </td>
                    <td>{fmt(h60?.rmse, 2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      ) : (
        <p className="card-sub">Results appear after <code>make forecast</code>.</p>
      )}
    </div>
  );
}

export function ImitationCard() {
  const runs = Object.entries(IMITATION.runs);
  const ready = runs.length > 0;
  const series = ready
    ? (() => {
        const steps = new Set<number>();
        runs.forEach(([, r]) => r.eval_rows.forEach((row) => steps.add(row.timesteps)));
        return Array.from(steps)
          .sort((a, b) => a - b)
          .map((t) => {
            const row: Record<string, number> = { timesteps: t };
            runs.forEach(([name, r]) => {
              const hit = r.eval_rows.find((x) => x.timesteps === t);
              if (hit) row[name] = hit.return;
            });
            return row;
          });
      })()
    : [];
  const bc = runs.find(([n]) => n.includes('bc'))?.[1];
  return (
    <div className="card wide">
      <div className="card-title">Imitation-learning warm start</div>
      {ready ? (
        <>
          <p className="card-sub">
            The policy network is first trained by behaviour cloning on {bc?.bc_samples?.toLocaleString()} max-pressure decisions (validation accuracy {fmt((bc?.bc_val_acc ?? 0) * 100, 1)}%), then
            fine-tuned with MaskablePPO. Validation return (morning peak, held-out seed) vs environment steps.
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={series} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.25)" />
              <XAxis dataKey="timesteps" tick={{ fill: 'var(--fg)', fontSize: 11 }} tickFormatter={(v: number) => `${Math.round(v / 1000)}k`} />
              <YAxis tick={{ fill: 'var(--fg)', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: 'var(--bg-2)', border: '1px solid var(--line)', color: 'var(--fg)' }} />
              <Legend />
              {runs.map(([name], i) => (
                <Line key={name} type="monotone" dataKey={name} stroke={['#f59e0b', '#34d399', '#60a5fa'][i % 3]} dot={false} strokeWidth={2} connectNulls />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </>
      ) : (
        <p className="card-sub">Results appear after the BC + PPO comparison runs.</p>
      )}
    </div>
  );
}

export function Extras() {
  return (
    <section className="section" id="extras">
      <h2>Additional ML components</h2>
      <div className="cards">
        <ForecastingCard />
        <ImitationCard />
      </div>
    </section>
  );
}
