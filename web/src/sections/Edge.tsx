import { EDGE, TRAINING, fmt } from '../data/results';

export function Edge() {
  const ok = EDGE.int8.n_calls > 0;
  return (
    <section className="section" id="edge">
      <h2>Edge deployment</h2>
      <p>
        The policy (with observation normalisation folded in) is exported to ONNX opset {EDGE.opset} and quantised to int8 with onnxruntime. An asyncio service subscribes to detector messages over
        MQTT, runs the ONNX policy, passes every decision through the same safety shield and publishes phase commands; a watchdog falls back to fixed-time when data goes stale or inference misses
        its deadline.
      </p>
      {ok ? (
        <div className="cards">
          <div className="card">
            <div className="card-title">int8 model size</div>
            <div className="card-value">{(EDGE.int8.size_bytes / 1024).toFixed(1)} kB</div>
            <div className="card-sub">float32: {(EDGE.float.size_bytes / 1024).toFixed(1)} kB</div>
          </div>
          <div className="card">
            <div className="card-title">Action agreement</div>
            <div className="card-value">{(EDGE.int8.agreement_vs_float * 100).toFixed(2)}%</div>
            <div className="card-sub">int8 vs float on {EDGE.n_random_obs.toLocaleString()} random observations · float vs PyTorch {(EDGE.float.agreement_vs_torch * 100).toFixed(2)}%</div>
          </div>
          <div className="card">
            <div className="card-title">Latency p50 / p95</div>
            <div className="card-value">
              {fmt(EDGE.int8.latency_p50_ms, 3)} / {fmt(EDGE.int8.latency_p95_ms, 3)} ms
            </div>
            <div className="card-sub">single thread, {EDGE.int8.n_calls.toLocaleString()} calls, approximation of a small edge CPU (not a Jetson measurement)</div>
          </div>
          {TRAINING.runs.length > 0 && (
            <div className="card">
              <div className="card-title">Training wall-clock</div>
              <div className="card-value">{(TRAINING.runs.reduce((a, r) => a + r.wall_clock_s, 0) / 3600).toFixed(1)} h</div>
              <div className="card-sub">
                {TRAINING.runs.length} seeds × {TRAINING.runs[0]?.total_timesteps.toLocaleString()} env steps on 4 CPU cores
              </div>
            </div>
          )}
        </div>
      ) : (
        <p className="muted">Edge benchmark numbers appear here after <code>make export</code>.</p>
      )}
    </section>
  );
}
