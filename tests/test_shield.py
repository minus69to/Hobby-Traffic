"""Unit tests for every rule of the safety shield plus property-based tests with hypothesis."""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from drs.config import SignalConfig
from drs.shield import Mode, SafetyShield
from drs.signal import PhaseTable


def make(table: PhaseTable, cfg: SignalConfig) -> SafetyShield:
    return SafetyShield(table, cfg)


def run_ticks(sh: SafetyShield, n: int) -> list[str]:
    return [sh.tick() for _ in range(n)]


def test_min_green_rejects_early_change(phase_table: PhaseTable, signal_cfg: SignalConfig) -> None:
    sh = make(phase_table, signal_cfg)
    run_ticks(sh, 5)
    assert not sh.request(2)
    assert sh.mode is Mode.GREEN and sh.phase == 0
    assert sh.stats.rejected_min_green == 1
    assert sh.log[0].reason.startswith("green for")
    run_ticks(sh, 5)
    assert sh.request(2)
    assert sh.mode is Mode.YELLOW


def test_clearance_sequence_yellow_then_all_red(phase_table: PhaseTable, signal_cfg: SignalConfig) -> None:
    sh = make(phase_table, signal_cfg)
    run_ticks(sh, 10)
    assert sh.request(2)
    states = run_ticks(sh, 5)
    assert all("y" in s for s in states[:3])
    assert all(set(s) == {"r"} for s in states[3:5])
    assert sh.mode is Mode.GREEN and sh.phase == 2
    assert sh.stats.switches == 1
    assert sh.tick() == phase_table.phases[2].state


def test_requests_during_clearance_are_ignored_and_logged(phase_table: PhaseTable, signal_cfg: SignalConfig) -> None:
    sh = make(phase_table, signal_cfg)
    run_ticks(sh, 10)
    sh.request(1)
    sh.tick()
    assert not sh.request(3)
    assert sh.stats.ignored_in_clearance == 1
    assert sh.target == 1
    assert sh.request(1)  # re-requesting the target is fine


def test_max_green_is_forced_without_controller(phase_table: PhaseTable, signal_cfg: SignalConfig) -> None:
    sh = make(phase_table, signal_cfg)
    states = run_ticks(sh, 200)
    greens = 0
    longest = 0
    for s in states:
        if s == phase_table.phases[sh.phase].state or "G" in s:
            greens += 1
            longest = max(longest, greens)
        else:
            greens = 0
    assert longest <= signal_cfg.max_green
    assert sh.stats.forced_max_green >= 2
    assert sh.phase != 0


def test_mask_matches_rules(phase_table: PhaseTable, signal_cfg: SignalConfig) -> None:
    sh = make(phase_table, signal_cfg)
    assert sh.valid_actions().tolist() == [True, False, False, False]
    run_ticks(sh, 10)
    assert sh.valid_actions().all()
    run_ticks(sh, 50)  # time in green = 60
    m = sh.valid_actions()
    assert not m[0] and m[1:].all()
    sh.request(3)
    assert sh.valid_actions().tolist() == [False, False, False, True]


def test_invalid_phase_rejected(phase_table: PhaseTable, signal_cfg: SignalConfig) -> None:
    sh = make(phase_table, signal_cfg)
    run_ticks(sh, 10)
    assert not sh.request(7)
    assert not sh.request(-1)
    assert sh.stats.rejected_invalid == 2
    assert sh.mode is Mode.GREEN


def test_ped_walk_minimum(phase_table: PhaseTable) -> None:
    cfg = SignalConfig(min_green=2, max_green=60, ped_walk_min=8, yellow=3, all_red=2)
    sh = SafetyShield(phase_table, cfg)
    run_ticks(sh, 4)
    assert not sh.request(1)
    assert sh.stats.rejected_ped_walk == 1
    run_ticks(sh, 4)
    assert sh.request(1)


# ----------------------------------------------------------------------------- property tests
def _check_trace(table: PhaseTable, cfg: SignalConfig, actions: list[int]) -> None:
    sh = SafetyShield(table, cfg)
    states: list[str] = []
    green_run = 0
    prev_green_links: set[int] = set()
    for a in actions:
        sh.request(a)
        for _ in range(5):
            s = sh.tick()
            states.append(s)
            assert table.is_conflict_free(s), "conflicting greens"
            green_links = {i for i, c in enumerate(s) if c in "Gg"}
            # links that lost green must have gone through yellow (vehicles) or red (crossings)
            if green_links != prev_green_links and prev_green_links and green_links:
                assert (
                    green_links.isdisjoint(prev_green_links)
                    or green_links <= prev_green_links
                    or prev_green_links <= green_links
                )
            prev_green_links = green_links
    # every green interval is between min_green and max_green seconds long,
    # and every switch is separated by yellow + all_red of clearance
    runs: list[tuple[str, int]] = []
    for s in states:
        kind = (
            "y"
            if "y" in s
            else ("r" if set(s) == {"r"} else "g" + str(sorted(i for i, c in enumerate(s) if c in "Gg")))
        )
        if runs and runs[-1][0] == kind:
            runs[-1] = (kind, runs[-1][1] + 1)
        else:
            runs.append((kind, 1))
    for k, (kind, n) in enumerate(runs):
        if kind.startswith("g"):
            green_run = n
            if 0 < k < len(runs) - 1:
                assert cfg.min_green <= green_run <= cfg.max_green, (kind, n)
            else:
                assert green_run <= cfg.max_green
        if kind == "y":
            last_run = k + 1 == len(runs) - 1
            assert n == cfg.yellow or (k == len(runs) - 1 and n <= cfg.yellow)  # trace may end mid-clearance
            if k + 1 < len(runs):
                assert runs[k + 1][0] == "r"
                assert runs[k + 1][1] == cfg.all_red or (last_run and runs[k + 1][1] <= cfg.all_red)
    assert sh.stats.switches == sum(1 for kind, _ in runs if kind == "y")


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(actions=st.lists(st.integers(min_value=-2, max_value=6), min_size=1, max_size=400))
def test_random_action_sequences_never_reach_unsafe_state(
    phase_table: PhaseTable, signal_cfg: SignalConfig, actions: list[int]
) -> None:
    _check_trace(phase_table, signal_cfg, actions)


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    actions=st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=300),
    min_green=st.integers(min_value=5, max_value=20),
    max_green=st.integers(min_value=25, max_value=90),
)
def test_random_timing_parameters(phase_table: PhaseTable, actions: list[int], min_green: int, max_green: int) -> None:
    cfg = SignalConfig(min_green=min_green, max_green=max_green, ped_walk_min=min(8, min_green), yellow=3, all_red=2)
    _check_trace(phase_table, cfg, actions)


def test_mask_never_allows_a_rejected_action(phase_table: PhaseTable, signal_cfg: SignalConfig) -> None:
    rng = np.random.default_rng(0)
    sh = SafetyShield(phase_table, signal_cfg)
    for _ in range(2000):
        mask = sh.valid_actions()
        a = int(rng.integers(0, 4))
        accepted = sh.request(a)
        if mask[a]:
            assert accepted, (a, mask, sh.mode, sh.time_in_mode)
        else:
            assert not accepted or a == sh.phase or a == sh.target
        sh.tick()
    assert sh.stats.rejected_min_green > 0 and sh.stats.ignored_in_clearance > 0
