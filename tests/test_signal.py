"""Phase table construction from the SUMO network."""

from __future__ import annotations

from drs.signal import PhaseTable


def test_four_phases_conflict_free(phase_table: PhaseTable) -> None:
    assert phase_table.n_phases == 4
    assert phase_table.n_links == 24
    for p in phase_table.phases:
        assert len(p.state) == phase_table.n_links
        assert phase_table.is_conflict_free(p.state)


def test_walk_phases_have_crossing_green(phase_table: PhaseTable) -> None:
    for p in phase_table.phases:
        crossing_green = any(p.state[i] == "G" for i in phase_table.crossings)
        assert crossing_green == p.has_walk


def test_every_vehicle_link_gets_green_somewhere(phase_table: PhaseTable) -> None:
    for lk in phase_table.links:
        assert any(p.state[lk.index] in "Gg" for p in phase_table.phases), lk


def test_yellow_state_only_yellows_links_losing_green(phase_table: PhaseTable) -> None:
    y = phase_table.yellow_state(0, 2)
    s0, s2 = phase_table.phases[0].state, phase_table.phases[2].state
    for i, c in enumerate(y):
        if s0[i] in "Gg" and s2[i] not in "Gg":
            assert c in ("y", "r")
            if phase_table.links[i].turn != "c":
                assert c == "y"
        else:
            assert c == "r"


def test_conflict_detection_rejects_all_green(phase_table: PhaseTable) -> None:
    assert not phase_table.is_conflict_free("G" * phase_table.n_links)
    assert phase_table.is_conflict_free("r" * phase_table.n_links)
    assert not phase_table.is_conflict_free("G")
