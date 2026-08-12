from __future__ import annotations

from pathlib import Path

from protocol_frontier.analysis import (
    _potential_edges,
    _ring_lattice_mask,
    run_bilateral_error_sensitivity,
    run_scaling,
    run_simulation,
    summarize,
    summarize_bilateral_error_sensitivity,
    summarize_coding,
    summarize_scaling,
    validate_case_coding,
)


ROOT = Path(__file__).resolve().parents[1]


def test_case_coding_is_complete_and_sourced() -> None:
    rows = validate_case_coding(
        ROOT / "data" / "cases" / "coding.csv",
        ROOT / "data" / "cases" / "sources.csv",
    )
    assert len(rows) == 36
    assert len({row["case_id"] for row in rows}) == 4


def test_case_summary_does_not_impute_unknowns() -> None:
    rows = validate_case_coding(
        ROOT / "data" / "cases" / "coding.csv",
        ROOT / "data" / "cases" / "sources.csv",
    )
    summary = summarize_coding(rows)
    for case in summary:
        assert (
            case["favorable"]
            + case["unfavorable"]
            + case["mixed"]
            + case["unknown"]
            == 9
        )


def test_scaling_edge_and_cost_identities() -> None:
    rows = run_scaling(ROOT / "configs" / "prespecified.toml")
    sample = [row for row in rows if row["replicate"] == 0 and row["agent_count"] == 32]
    by_family = {row["family"]: row for row in sample}
    assert by_family["complete"]["edge_count"] == 32 * 31 // 2
    assert by_family["bounded_degree"]["edge_count"] == 2 * 32
    ring_edges = [
        edge
        for edge, included in zip(_potential_edges(32), _ring_lattice_mask(32, 4), strict=True)
        if included
    ]
    degrees = {agent: 0 for agent in range(32)}
    for source, target in ring_edges:
        degrees[source] += 1
        degrees[target] += 1
    assert set(degrees.values()) == {4}
    for row in sample:
        assert row["setup_saving"] == (
            row["bilateral_setup_cost"] - row["protocol_setup_cost"]
        )


def test_scaling_is_deterministic_and_discriminates_graph_families() -> None:
    config = ROOT / "configs" / "prespecified.toml"
    first = run_scaling(config)
    second = run_scaling(config)
    assert first == second
    summary = summarize_scaling(first)
    endpoint = {
        row["family"]: row
        for row in summary
        if row["agent_count"] == 256
    }
    assert endpoint["complete"]["setup_saving_q025"] > 0
    assert endpoint["fixed_density"]["setup_saving_q025"] > 0
    assert endpoint["bounded_degree"]["setup_saving_q975"] < 0


def test_paired_intervals_and_screen_indeterminacy() -> None:
    config = ROOT / "configs" / "prespecified.toml"
    summary = summarize(run_simulation(config), config)
    lookup = {(row["scenario"], row["regime"]): row for row in summary}
    induced = lookup[("induced_unequal_access", "protocol_escalation")]
    low_volume = lookup[("low_volume_fixed_cost", "protocol_escalation")]
    assert induced["cost_difference_q025"] < 0 < induced["cost_difference_q975"]
    assert low_volume["cost_difference_q025"] < 0 < low_volume["cost_difference_q975"]
    assert induced["screen"] == "constraint-rejected"
    assert low_volume["screen"] == "direction-indeterminate"


def test_bilateral_error_sensitivity_is_monotone_and_preserves_zero_rate() -> None:
    config = ROOT / "configs" / "prespecified.toml"
    simulation = run_simulation(config)
    main_summary = summarize(simulation, config)
    sensitivity = summarize_bilateral_error_sensitivity(
        run_bilateral_error_sensitivity(simulation, config)
    )
    main_lookup = {
        (row["scenario"], row["regime"]): row
        for row in main_summary
        if row["regime"] != "bilateral"
    }
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in sensitivity:
        grouped.setdefault((row["scenario"], row["regime"]), []).append(row)
    for key, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: row["bilateral_error_rate"])
        assert ordered[0]["bilateral_error_rate"] == 0.0
        assert ordered[0]["welfare_difference_mean"] == main_lookup[key][
            "welfare_difference_vs_bilateral"
        ]
        assert all(
            current["welfare_difference_mean"] < following["welfare_difference_mean"]
            for current, following in zip(ordered, ordered[1:])
        )
