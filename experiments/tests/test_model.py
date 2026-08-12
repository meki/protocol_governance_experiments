from __future__ import annotations

from pathlib import Path

from protocol_frontier.analysis import read_config
from protocol_frontier.model import simulate_replicate


ROOT = Path(__file__).resolve().parents[1]


def test_replicate_is_deterministic() -> None:
    _, common, scenarios = read_config(ROOT / "configs" / "prespecified.toml")
    first = simulate_replicate(common, scenarios[0], "protocol_escalation", 1000, 17)
    second = simulate_replicate(common, scenarios[0], "protocol_escalation", 1000, 17)
    assert first == second


def test_rates_and_counts_are_bounded() -> None:
    _, common, scenarios = read_config(ROOT / "configs" / "prespecified.toml")
    for scenario in scenarios:
        for regime in ("bilateral", "protocol_escalation", "protocol_binary"):
            row = simulate_replicate(common, scenario, regime, 1000, 19)
            assert 0 <= row["completed"] <= row["attempted"]
            for key in (
                "legibility_gap",
                "legibility_gap_reference",
                "legibility_gap_disadvantaged",
            ):
                assert 0.0 <= row[key] <= 1.0
            assert -1.0 <= row["group_gap"] <= 1.0
            assert row["total_cost"] >= 0.0
            assert 0.0 <= row["represented_disagreement_rate"] <= 1.0


def test_binary_ablation_does_not_escalate() -> None:
    _, common, scenarios = read_config(ROOT / "configs" / "prespecified.toml")
    row = simulate_replicate(common, scenarios[2], "protocol_binary", 5000, 23)
    assert row["escalated"] == 0
    assert row["encoding_failures"] > 0


def test_cost_ledger_identities() -> None:
    _, common, scenarios = read_config(ROOT / "configs" / "prespecified.toml")
    scenario = scenarios[1]
    bilateral = simulate_replicate(common, scenario, "bilateral", 2000, 29)
    assert bilateral["total_cost"] == (
        common.bilateral_fixed_cost
        + bilateral["attempted"] * common.bilateral_unit_cost
    )

    escalated = simulate_replicate(common, scenario, "protocol_escalation", 2000, 29)
    assert escalated["total_cost"] == (
        common.protocol_design_cost
        + common.protocol_adoption_cost
        + scenario.maintenance_cost
        + escalated["attempted"] * common.protocol_unit_cost
        + escalated["escalated"] * common.escalation_unit_cost
    )

    binary = simulate_replicate(common, scenario, "protocol_binary", 2000, 29)
    assert binary["total_cost"] == (
        common.protocol_design_cost
        + common.protocol_adoption_cost
        + scenario.maintenance_cost
        + binary["attempted"] * common.protocol_unit_cost
    )


def test_welfare_proxy_identity() -> None:
    _, common, scenarios = read_config(ROOT / "configs" / "prespecified.toml")
    for regime in ("bilateral", "protocol_escalation", "protocol_binary"):
        row = simulate_replicate(common, scenarios[2], regime, 2000, 31)
        assert row["welfare_proxy"] == (
            row["access_benefit"]
            + row["correct_completion_value"]
            - row["total_cost"]
            - row["harm_cost"]
        )
