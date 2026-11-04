import { useEffect, useMemo, useState } from 'react';
import { Viewer } from './scene/Viewer';
import { HUD } from './ui/HUD';
import { Controls } from './ui/Controls';
import { KeyResults } from './sections/KeyResults';
import { HowItWorks, SafetyFSM } from './sections/Diagrams';
import { ResultsCharts } from './sections/ResultsCharts';
import { Edge } from './sections/Edge';
import { Author, Limitations } from './sections/Outro';
import { Extras } from './sections/Extras';
import { loadIndex } from './replay/loader';
import { useStore } from './store/playback';
import type { ReplayIndexEntry } from './replay/types';

export default function App() {
  const explorer = useStore((s) => s.explorer);
  const set = useStore((s) => s.set);
  const [index, setIndex] = useState<ReplayIndexEntry[]>([]);
  useEffect(() => {
    loadIndex()
      .then((rows) => {
        setIndex(rows);
        const st = useStore.getState();
        const ctrls = Array.from(new Set(rows.map((r) => r.controller)));
        const scens = Array.from(new Set(rows.map((r) => r.scenario)));
        const patch: Record<string, string> = {};
        if (!scens.includes(st.scenario) && scens[0]) patch.scenario = scens[0];
        if (!ctrls.includes(st.rightController)) patch.rightController = ctrls.find((c) => c.startsWith('ppo')) ?? ctrls[ctrls.length - 1] ?? st.rightController;
        if (!ctrls.includes(st.leftController)) patch.leftController = ctrls[0] ?? st.leftController;
        if (Object.keys(patch).length) set(patch);
      })
      .catch(() => setIndex([]));
  }, [set]);
  const scenarios = useMemo(() => Array.from(new Set(index.map((r) => r.scenario))), [index]);
  const controllers = useMemo(() => Array.from(new Set(index.map((r) => r.controller))), [index]);
  useEffect(() => {
    document.body.classList.toggle('explorer', explorer);
  }, [explorer]);
  return (
    <div className={`app ${explorer ? 'explorer' : ''}`}>
      <header className="hero">
        <div className="hero-3d">
          <Viewer />
          <HUD />
          {explorer && <Controls scenarios={scenarios} controllers={controllers} />}
        </div>
        {!explorer && (
          <div className="hero-text">
            <h1>Safe Deep RL Traffic Signals for Dhaka-like Mixed Traffic</h1>
            <p className="sub">
              A synthetic SUMO study: buses, cars, motorcycles, CNG auto-rickshaws, cycle-rickshaws and pedestrians sharing lanes at a signalised junction. Left: a Webster fixed-time plan. Right: a
              MaskablePPO agent behind a runtime safety shield. Same demand, same clock.
            </p>
            <div className="hero-buttons">
              <button className="btn primary" onClick={() => set({ explorer: true })}>
                Explore the simulation
              </button>
              <a className="btn" href={`${import.meta.env.BASE_URL}report.html`}>
                Read the report
              </a>
            </div>
          </div>
        )}
        {explorer && (
          <button className="btn close" onClick={() => set({ explorer: false })} aria-label="Back to the page">
            ✕ back to the story
          </button>
        )}
      </header>
      {!explorer && (
        <main>
          <KeyResults />
          <HowItWorks />
          <SafetyFSM />
          <ResultsCharts />
          <Edge />
          <Extras />
          <Limitations />
          <Author />
        </main>
      )}
    </div>
  );
}
