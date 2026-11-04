/** Animated "how it works" pipeline and the shield state machine, as inline SVG (no assets). */

export function HowItWorks() {
  const boxes = [
    { x: 20, label: 'SUMO simulation', sub: 'sublane mixed traffic', c: '#3b82f6' },
    { x: 220, label: 'Observation', sub: 'queues · waits · classes · peds', c: '#8b5cf6' },
    { x: 420, label: 'PPO policy', sub: '2×128 tanh MLP', c: '#ec4899' },
    { x: 620, label: 'Action mask', sub: 'min/max green · clearance', c: '#f59e0b' },
    { x: 820, label: 'Safety shield', sub: 'FSM: yellow · all-red', c: '#10b981' },
  ];
  return (
    <section className="section" id="how">
      <h2>How it works</h2>
      <p>
        Every 5 s the agent sees normalised queue lengths and maximum waiting times per lane, vehicle counts per class per approach (so it can see slow rickshaws), the current phase and its age, and
        pedestrians waiting per crossing. It picks the next green phase. <strong>MaskablePPO</strong> can only choose phases the mask allows, and the <strong>safety shield</strong> turns the
        request into lamp states, inserting yellow and all-red and enforcing every timer, regardless of what the policy says.
      </p>
      <svg viewBox="0 0 1020 190" className="diagram" role="img" aria-label="Pipeline: simulation, observation, PPO policy, action mask, safety shield, back to the simulation">
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="currentColor" />
          </marker>
        </defs>
        {boxes.map((b, i) => (
          <g key={b.label}>
            <rect x={b.x} y={40} width={170} height={70} rx={12} fill={b.c} opacity={0.18} stroke={b.c} strokeWidth={2} />
            <text x={b.x + 85} y={70} textAnchor="middle" className="svg-label">
              {b.label}
            </text>
            <text x={b.x + 85} y={92} textAnchor="middle" className="svg-sub">
              {b.sub}
            </text>
            {i < boxes.length - 1 && <line x1={b.x + 172} y1={75} x2={b.x + 198} y2={75} stroke="currentColor" strokeWidth={2} markerEnd="url(#arrow)" />}
          </g>
        ))}
        <path d="M 905 112 C 905 170, 105 170, 105 112" fill="none" stroke="currentColor" strokeWidth={2} strokeDasharray="6 6" markerEnd="url(#arrow)">
          <animate attributeName="stroke-dashoffset" from="24" to="0" dur="1.2s" repeatCount="indefinite" />
        </path>
        <text x={505} y={168} textAnchor="middle" className="svg-sub">
          lamp states applied every second · reward = −Δwaiting − pressure − pedestrian wait − switch penalty
        </text>
        <circle r={5} fill="#fbbf24">
          <animateMotion dur="4s" repeatCount="indefinite" path="M 105 75 L 905 75" />
        </circle>
      </svg>
    </section>
  );
}

export function SafetyFSM() {
  return (
    <section className="section" id="safety">
      <h2>Safety by construction</h2>
      <div className="two-col">
        <svg viewBox="0 0 640 260" className="diagram" role="img" aria-label="Shield state machine: green, yellow, all-red, green of the next phase">
          <g>
            <circle cx={90} cy={130} r={52} fill="#10b981" opacity={0.25} stroke="#10b981" strokeWidth={3} />
            <text x={90} y={126} textAnchor="middle" className="svg-label">
              GREEN(p)
            </text>
            <text x={90} y={146} textAnchor="middle" className="svg-sub">
              10 s ≤ t ≤ 60 s
            </text>
            <circle cx={320} cy={60} r={44} fill="#f59e0b" opacity={0.25} stroke="#f59e0b" strokeWidth={3} />
            <text x={320} y={56} textAnchor="middle" className="svg-label">
              YELLOW
            </text>
            <text x={320} y={76} textAnchor="middle" className="svg-sub">
              3 s
            </text>
            <circle cx={320} cy={200} r={44} fill="#ef4444" opacity={0.25} stroke="#ef4444" strokeWidth={3} />
            <text x={320} y={196} textAnchor="middle" className="svg-label">
              ALL-RED
            </text>
            <text x={320} y={216} textAnchor="middle" className="svg-sub">
              2 s
            </text>
            <circle cx={550} cy={130} r={52} fill="#10b981" opacity={0.25} stroke="#10b981" strokeWidth={3} />
            <text x={550} y={126} textAnchor="middle" className="svg-label">
              GREEN(q)
            </text>
            <text x={550} y={146} textAnchor="middle" className="svg-sub">
              timer reset
            </text>
            <path d="M 135 105 Q 220 40 276 60" fill="none" stroke="currentColor" strokeWidth={2} markerEnd="url(#arrow)" />
            <text x={190} y={52} textAnchor="middle" className="svg-sub">
              request q accepted
            </text>
            <path d="M 320 104 L 320 156" fill="none" stroke="currentColor" strokeWidth={2} markerEnd="url(#arrow)" />
            <path d="M 364 200 Q 470 220 505 165" fill="none" stroke="currentColor" strokeWidth={2} markerEnd="url(#arrow)" />
            <path d="M 60 80 C 20 10, 150 10, 120 82" fill="none" stroke="currentColor" strokeWidth={2} strokeDasharray="4 4" markerEnd="url(#arrow)" />
            <text x={90} y={24} textAnchor="middle" className="svg-sub">
              rejected: t &lt; min green · walk min · conflict
            </text>
            <path d="M 135 150 Q 320 250 505 150" fill="none" stroke="currentColor" strokeWidth={2} strokeDasharray="4 4" markerEnd="url(#arrow)" />
            <text x={320} y={252} textAnchor="middle" className="svg-sub">
              forced after max green (no controller needed)
            </text>
          </g>
        </svg>
        <div>
          <ul className="bullets">
            <li>
              <strong>Action masking.</strong> A phase change is invalid before <em>min green</em> (10 s), forced after <em>max green</em> (60 s), and impossible during yellow, all-red or the
              pedestrian walk minimum (8 s). The policy never samples an invalid action.
            </li>
            <li>
              <strong>Runtime shield.</strong> An explicit finite-state machine that every controller (RL, both baselines and the deployed MQTT service) passes through. It inserts yellow and all-red,
              rejects conflicting greens using the network's conflict matrix, and logs every intervention. Shield interventions are reported in the results even when zero.
            </li>
            <li>
              <strong>Property-based tests.</strong> <code>hypothesis</code> generates hundreds of random action sequences and timing parameters and asserts that no conflicting greens appear, every green
              lasts between min and max, and every switch goes through the full clearance interval.
            </li>
          </ul>
        </div>
      </div>
    </section>
  );
}
