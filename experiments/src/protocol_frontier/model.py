from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CommonParameters:
    bilateral_fixed_cost: float
    protocol_design_cost: float
    protocol_adoption_cost: float
    bilateral_unit_cost: float
    protocol_unit_cost: float
    escalation_unit_cost: float
    false_decision_cost: float
    access_value: float
    correct_completion_value: float
    base_positive_rate: float
    disadvantaged_share: float


@dataclass(frozen=True)
class Scenario:
    id: str
    label: str
    bilateral_volume_multiplier: float
    protocol_volume_multiplier: float
    encoding_failure_reference: float
    encoding_failure_disadvantaged: float
    decision_uncertainty: float
    represented_error: float
    additional_error: float
    maintenance_cost: float


def _safe_rate(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def simulate_replicate(
    common: CommonParameters,
    scenario: Scenario,
    regime: str,
    cases_per_replicate: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Simulate one replicate under a specified coordination regime.

    The data-generating process is a designed mechanism check. It is not a
    calibrated estimate of any deployed institution.
    """

    if regime not in {"bilateral", "protocol_escalation", "protocol_binary"}:
        raise ValueError(f"Unknown regime: {regime}")

    rng = np.random.default_rng(seed)
    group_disadvantaged = rng.random(cases_per_replicate) < common.disadvantaged_share
    eligible = rng.random(cases_per_replicate) < common.base_positive_rate
    propensity = rng.random(cases_per_replicate)

    volume_multiplier = (
        scenario.bilateral_volume_multiplier
        if regime == "bilateral"
        else scenario.protocol_volume_multiplier
    )
    included = propensity < volume_multiplier
    attempted = int(included.sum())

    if regime == "bilateral":
        completed = attempted
        correct = attempted
        fixed_cost = common.bilateral_fixed_cost
        variable_cost = attempted * common.bilateral_unit_cost
        total_cost = fixed_cost + variable_cost
        return {
            "scenario": scenario.id,
            "scenario_label": scenario.label,
            "regime": regime,
            "seed": seed,
            "attempted": attempted,
            "completed": completed,
            "escalated": 0,
            "unresolved": 0,
            "encoding_failures": 0,
            "decision_failures": 0,
            "false_denials": 0,
            "false_approvals": 0,
            "represented_disagreements": 0,
            "represented_disagreement_rate": 0.0,
            "legibility_gap": 0.0,
            "legibility_gap_reference": 0.0,
            "legibility_gap_disadvantaged": 0.0,
            "group_gap": 0.0,
            "total_cost": float(total_cost),
            "cost_per_completed": float(total_cost / completed) if completed else float("nan"),
            "access_benefit": float(attempted * common.access_value),
            "correct_completion_value": float(correct * common.correct_completion_value),
            "harm_cost": 0.0,
            "welfare_proxy": float(
                attempted * common.access_value
                + correct * common.correct_completion_value
                - total_cost
            ),
        }

    included_group = group_disadvantaged[included]
    included_eligible = eligible[included]
    encoding_prob = np.where(
        included_group,
        scenario.encoding_failure_disadvantaged,
        scenario.encoding_failure_reference,
    )
    encoding_failure = rng.random(attempted) < encoding_prob
    represented = ~encoding_failure

    decision_failure = np.zeros(attempted, dtype=bool)
    decision_failure[represented] = (
        rng.random(int(represented.sum())) < scenario.decision_uncertainty
    )
    decidable = represented & ~decision_failure

    error_prob = min(1.0, scenario.represented_error + scenario.additional_error)
    represented_error = np.zeros(attempted, dtype=bool)
    represented_error[decidable] = rng.random(int(decidable.sum())) < error_prob

    if regime == "protocol_escalation":
        escalated = encoding_failure | decision_failure
        unresolved = np.zeros(attempted, dtype=bool)
        incorrect = represented_error
        represented_disagreement = represented_error
        completed_mask = np.ones(attempted, dtype=bool)
        variable_cost = (
            attempted * common.protocol_unit_cost
            + int(escalated.sum()) * common.escalation_unit_cost
        )
    else:
        escalated = np.zeros(attempted, dtype=bool)
        unresolved = decision_failure
        forced_denial = encoding_failure
        incorrect = represented_error | forced_denial
        represented_disagreement = represented_error
        completed_mask = ~unresolved
        variable_cost = attempted * common.protocol_unit_cost

    predicted_positive = included_eligible.copy()
    predicted_positive[incorrect] = ~included_eligible[incorrect]
    if regime == "protocol_binary":
        predicted_positive[encoding_failure] = False

    evaluated = completed_mask
    disagreement = evaluated & (predicted_positive != included_eligible)
    gap = encoding_failure | decision_failure | disagreement
    false_denial = evaluated & included_eligible & ~predicted_positive
    false_approval = evaluated & ~included_eligible & predicted_positive

    completed = int(completed_mask.sum())
    correct = int((evaluated & ~disagreement).sum())
    fixed_cost = common.protocol_design_cost + common.protocol_adoption_cost
    total_cost = fixed_cost + scenario.maintenance_cost + variable_cost
    harm_cost = int(disagreement.sum()) * common.false_decision_cost
    access_benefit = attempted * common.access_value
    correct_value = correct * common.correct_completion_value
    welfare_proxy = access_benefit + correct_value - total_cost - harm_cost

    reference_mask = ~included_group
    disadvantaged_mask = included_group
    gap_reference = _safe_rate(int((gap & reference_mask).sum()), int(reference_mask.sum()))
    gap_disadvantaged = _safe_rate(
        int((gap & disadvantaged_mask).sum()), int(disadvantaged_mask.sum())
    )

    return {
        "scenario": scenario.id,
        "scenario_label": scenario.label,
        "regime": regime,
        "seed": seed,
        "attempted": attempted,
        "completed": completed,
        "escalated": int(escalated.sum()),
        "unresolved": int(unresolved.sum()),
        "encoding_failures": int(encoding_failure.sum()),
        "decision_failures": int(decision_failure.sum()),
        "false_denials": int(false_denial.sum()),
        "false_approvals": int(false_approval.sum()),
        "represented_disagreements": int(represented_disagreement.sum()),
        "represented_disagreement_rate": _safe_rate(
            int(represented_disagreement.sum()), attempted
        ),
        "legibility_gap": _safe_rate(int(gap.sum()), attempted),
        "legibility_gap_reference": gap_reference,
        "legibility_gap_disadvantaged": gap_disadvantaged,
        "group_gap": gap_disadvantaged - gap_reference,
        "total_cost": float(total_cost),
        "cost_per_completed": float(total_cost / completed) if completed else float("nan"),
        "access_benefit": float(access_benefit),
        "correct_completion_value": float(correct_value),
        "harm_cost": float(harm_cost),
        "welfare_proxy": float(welfare_proxy),
    }
