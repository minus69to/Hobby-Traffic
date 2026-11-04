/**
 * Build-time import of the numbers produced by the Python pipeline (results/ directory).
 * Nothing in the UI is hard-coded; if a file is missing the corresponding section says so.
 */
import evalSummary from '@results/eval_summary.json';
import edgeBenchmark from '@results/edge_benchmark.json';
import trainingSummary from '@results/training_summary.json';
import forecastResults from '@results/forecast.json';
import imitationResults from '@results/imitation.json';

export interface MetricStat {
  mean: number;
  std: number;
  ci_low: number;
  ci_high: number;
  pct_vs_fixed?: number;
  pct_ci_low?: number;
  pct_ci_high?: number;
}

export interface ScenarioEntry {
  n: number;
  [metric: string]: MetricStat | number;
}

export interface EvalSummary {
  meta: {
    network: string;
    scenarios: string[];
    controllers: string[];
    skipped_controllers: Record<string, string>;
    episodes_per_scenario: number;
    eval_seeds: number[];
    episode_seconds: number;
    warmup_seconds: number;
    metrics: string[];
    lower_is_better: Record<string, boolean>;
    wall_clock_s: number;
    n_episodes: number;
    note: string;
  };
  summary: Record<string, Record<string, ScenarioEntry>>;
}

export interface EdgeBenchmark {
  model: string;
  opset: number;
  obs_dim: number;
  n_actions: number;
  float: { size_bytes: number; agreement_vs_torch: number; latency_p50_ms: number; latency_p95_ms: number; n_calls: number };
  int8: { size_bytes: number; agreement_vs_float: number; agreement_vs_torch: number; latency_p50_ms: number; latency_p95_ms: number; n_calls: number };
  n_random_obs: number;
  threads: number;
  cpu: string;
  note: string;
}

export interface TrainingSummary {
  runs: { seed: number; network: string; total_timesteps: number; wall_clock_s: number; env_steps_per_s: number; best_validation_return: number | null }[];
  note: string;
}

export interface ForecastResults {
  n_train_windows?: number;
  n_test_windows?: number;
  history_s?: number;
  models: Record<string, Record<string, { mae: number; mae_ci: [number, number]; rmse: number; rmse_ci: [number, number] } | number | number[]>>;
  note: string;
}

export interface ImitationResults {
  runs: Record<string, { eval_rows: { timesteps: number; wall_s: number; return: number }[]; wall_clock_s: number; best_validation_return: number | null; bc_val_acc?: number; bc_samples?: number }>;
  note: string;
}

export const FORECAST = forecastResults as unknown as ForecastResults;
export const IMITATION = imitationResults as unknown as ImitationResults;
export const EVAL = evalSummary as unknown as EvalSummary;
export const EDGE = edgeBenchmark as unknown as EdgeBenchmark;
export const TRAINING = trainingSummary as unknown as TrainingSummary;

export const SCENARIO_LABELS: Record<string, string> = {
  offpeak: 'Off-peak',
  morning_peak: 'Morning peak',
  evening_peak: 'Evening peak',
  incident: 'Incident',
  surge: 'Surge (+20%)',
  noisy: 'Noisy sensors',
};

export const CONTROLLER_LABELS: Record<string, string> = {
  fixed_time: 'Fixed-time (Webster)',
  max_pressure: 'Max-pressure',
  ppo: 'Safe PPO (3 seeds pooled)',
  ppo_s13: 'Safe PPO seed 13',
  ppo_s42: 'Safe PPO seed 42',
  ppo_s2026: 'Safe PPO seed 2026',
};

export const METRIC_LABELS: Record<string, string> = {
  mean_travel_time: 'Mean travel time (s)',
  p95_travel_time: '95th pct travel time (s)',
  mean_waiting_time: 'Mean waiting time (s)',
  mean_time_loss: 'Mean time loss (s)',
  mean_queue_length: 'Mean queue (veh)',
  throughput: 'Throughput (veh/h)',
  ped_mean_wait: 'Pedestrian mean wait (s)',
  shield_interventions: 'Shield interventions',
  phase_switches_per_hour: 'Phase switches / h',
  wait_car: 'Car wait (s)',
  wait_bus: 'Bus wait (s)',
  wait_truck: 'Truck wait (s)',
  wait_motorcycle: 'Motorcycle wait (s)',
  wait_cng: 'CNG wait (s)',
  wait_rickshaw: 'Rickshaw wait (s)',
};

export function stat(ctrl: string, scen: string, metric: string): MetricStat | null {
  const e = EVAL.summary[ctrl]?.[scen];
  if (!e) return null;
  const v = e[metric];
  return typeof v === 'object' ? v : null;
}

export function fmt(x: number | undefined | null, digits = 1): string {
  if (x === undefined || x === null || Number.isNaN(x)) return 'n/a';
  return x.toFixed(digits);
}

export function fmtPct(x: number | undefined | null): string {
  if (x === undefined || x === null || Number.isNaN(x)) return 'n/a';
  return `${x > 0 ? '+' : ''}${x.toFixed(1)}%`;
}
