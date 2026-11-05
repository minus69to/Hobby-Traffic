export function Limitations() {
  return (
    <section className="section" id="limitations">
      <h2>Limitations and future work</h2>
      <ul className="bullets">
        <li>
          <strong>Synthetic demand.</strong> Vehicle mix, volumes, turning ratios and vehicle parameters are assumptions inspired by Dhaka traffic. Nothing was calibrated or validated against real
          data, and no real-world deployment is claimed.
        </li>
        <li>
          <strong>Simplified pedestrians.</strong> SUMO's striping model with signalised crossings; no jaywalking, no mid-block crossing, no street vendors.
        </li>
        <li>
          <strong>CPU training budget.</strong> Three seeds of a few hundred thousand environment steps each; longer training and larger networks would likely change the numbers.
        </li>
        <li>
          <strong>Single junction first.</strong> The 2×2 grid, the parameter-shared policy and coordination experiments are secondary and only included where time allowed.
        </li>
        <li>
          <strong>Next steps.</strong> Real detector data, sim-to-real transfer with domain randomisation, multi-agent coordination on corridors, and transfer to other cities with different mixes.
        </li>
      </ul>
    </section>
  );
}

export function Author() {
  return (
    <section className="section author" id="author">
      <h2>Author</h2>
      <p>
        <strong>Souvik Mandol</strong> · <a href="https://github.com/minus69to">GitHub</a> · <a href="https://github.com/minus69to/Hobby-Traffic">source repository</a>
      </p>
      <p className="muted">
        Built from scratch with public tools: SUMO, Gymnasium, Stable-Baselines3 / sb3-contrib, PyTorch, ONNX Runtime, FastAPI, MQTT, React, three.js. No proprietary data, code or protocols from any
        real traffic-signal deployment were used.
      </p>
      <footer className="footer">
        <span>Code: MIT licence.</span> <span>Simulation: Eclipse SUMO (EPL-2.0).</span> <span>Fonts: Noto Sans Bengali (OFL).</span> <span>3D: three.js (MIT).</span>
      </footer>
    </section>
  );
}
