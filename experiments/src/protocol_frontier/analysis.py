from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tomllib
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import CommonParameters, Scenario, simulate_replicate


REGIMES = ("bilateral", "protocol_escalation", "protocol_binary")
RATE_METRICS = (
    "legibility_gap",
    "legibility_gap_reference",
    "legibility_gap_disadvantaged",
    "group_gap",
)
CONTINUOUS_METRICS = (
    "attempted",
    "completed",
    "escalated",
    "unresolved",
    "encoding_failures",
    "decision_failures",
    "false_denials",
    "false_approvals",
    "represented_disagreements",
    "represented_disagreement_rate",
    "legibility_gap",
    "legibility_gap_reference",
    "legibility_gap_disadvantaged",
    "group_gap",
    "total_cost",
    "cost_per_completed",
    "access_benefit",
    "correct_completion_value",
    "harm_cost",
    "welfare_proxy",
)
SCALING_FAMILIES = ("complete", "fixed_density", "bounded_degree")


def read_config(path: Path) -> tuple[dict, CommonParameters, list[Scenario]]:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    common = CommonParameters(**raw["common"])
    scenarios = [Scenario(**item) for item in raw["scenarios"]]
    return raw, common, scenarios


def run_simulation(config_path: Path) -> list[dict]:
    raw, common, scenarios = read_config(config_path)
    master_seed = int(raw["master_seed"])
    replicates = int(raw["replicates"])
    cases_per_replicate = int(raw["cases_per_replicate"])
    rows: list[dict] = []
    for scenario_index, scenario in enumerate(scenarios):
        for replicate in range(replicates):
            shared_seed = master_seed + scenario_index * 100_000 + replicate
            for regime in REGIMES:
                result = simulate_replicate(
                    common=common,
                    scenario=scenario,
                    regime=regime,
                    cases_per_replicate=cases_per_replicate,
                    seed=shared_seed,
                )
                result["replicate"] = replicate
                rows.append(result)
    return rows


def _lognormal_parameters(mean: float, coefficient_of_variation: float) -> tuple[float, float]:
    sigma2 = math.log1p(coefficient_of_variation**2)
    return math.log(mean) - sigma2 / 2.0, math.sqrt(sigma2)


def _potential_edges(agent_count: int) -> list[tuple[int, int]]:
    return [
        (source, target)
        for source in range(agent_count)
        for target in range(source + 1, agent_count)
    ]


def _ring_lattice_mask(agent_count: int, degree: int) -> np.ndarray:
    if degree <= 0 or degree % 2 or degree >= agent_count:
        raise ValueError("Ring-lattice degree must be positive, even, and below agent count")
    edge_index = {edge: index for index, edge in enumerate(_potential_edges(agent_count))}
    mask = np.zeros(len(edge_index), dtype=bool)
    for source in range(agent_count):
        for distance in range(1, degree // 2 + 1):
            target = (source + distance) % agent_count
            edge = tuple(sorted((source, target)))
            mask[edge_index[edge]] = True
    return mask


def run_scaling(config_path: Path) -> list[dict]:
    raw, _, _ = read_config(config_path)
    scaling = raw["scaling"]
    master_seed = int(raw["master_seed"]) + 9_000_000
    mapping_mu, mapping_sigma = _lognormal_parameters(
        float(scaling["bilateral_mapping_mean"]),
        float(scaling["cost_coefficient_of_variation"]),
    )
    adapter_mu, adapter_sigma = _lognormal_parameters(
        float(scaling["adapter_cost_mean"]),
        float(scaling["cost_coefficient_of_variation"]),
    )
    rows: list[dict] = []
    for n_index, agent_count in enumerate(scaling["agent_counts"]):
        possible_edges = agent_count * (agent_count - 1) // 2
        ring_mask = _ring_lattice_mask(agent_count, int(scaling["ring_degree"]))
        for replicate in range(int(scaling["replicates"])):
            seed = master_seed + n_index * 100_000 + replicate
            rng = np.random.default_rng(seed)
            mapping_costs = rng.lognormal(mapping_mu, mapping_sigma, possible_edges)
            adapter_costs = rng.lognormal(adapter_mu, adapter_sigma, agent_count)
            fixed_density_mask = rng.random(possible_edges) < float(scaling["fixed_density"])
            masks = {
                "complete": np.ones(possible_edges, dtype=bool),
                "fixed_density": fixed_density_mask,
                "bounded_degree": ring_mask,
            }
            protocol_setup_cost = float(scaling["design_cost"] + adapter_costs.sum())
            for family, mask in masks.items():
                edge_count = int(mask.sum())
                bilateral_setup_cost = float(mapping_costs[mask].sum())
                rows.append(
                    {
                        "family": family,
                        "agent_count": agent_count,
                        "replicate": replicate,
                        "seed": seed,
                        "possible_edges": possible_edges,
                        "edge_count": edge_count,
                        "density": edge_count / possible_edges,
                        "bilateral_setup_cost": bilateral_setup_cost,
                        "protocol_setup_cost": protocol_setup_cost,
                        "setup_saving": bilateral_setup_cost - protocol_setup_cost,
                    }
                )
    return rows


def _quantile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.quantile(array, q))


def summarize(rows: list[dict], config_path: Path) -> list[dict]:
    raw, _, scenarios = read_config(config_path)
    decision = raw["decision"]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["scenario"], row["regime"]), []).append(row)

    summaries: list[dict] = []
    for scenario in scenarios:
        baseline = grouped[(scenario.id, "bilateral")]
        baseline_by_replicate = {int(row["replicate"]): row for row in baseline}
        for regime in REGIMES:
            records = grouped[(scenario.id, regime)]
            item: dict[str, float | str] = {
                "scenario": scenario.id,
                "scenario_label": scenario.label,
                "regime": regime,
            }
            for metric in CONTINUOUS_METRICS:
                values = [float(row[metric]) for row in records]
                item[f"{metric}_mean"] = statistics.mean(values)
                item[f"{metric}_q025"] = _quantile(values, 0.025)
                item[f"{metric}_q975"] = _quantile(values, 0.975)
            paired_cost_differences = [
                float(row["total_cost"])
                - float(baseline_by_replicate[int(row["replicate"])]["total_cost"])
                for row in records
            ]
            paired_welfare_differences = [
                float(row["welfare_proxy"])
                - float(baseline_by_replicate[int(row["replicate"])]["welfare_proxy"])
                for row in records
            ]
            item["cost_difference_vs_bilateral"] = statistics.mean(paired_cost_differences)
            item["cost_difference_q025"] = _quantile(paired_cost_differences, 0.025)
            item["cost_difference_q975"] = _quantile(paired_cost_differences, 0.975)
            item["probability_cost_difference_positive"] = statistics.mean(
                value > 0 for value in paired_cost_differences
            )
            item["welfare_difference_vs_bilateral"] = statistics.mean(
                paired_welfare_differences
            )
            item["welfare_difference_q025"] = _quantile(paired_welfare_differences, 0.025)
            item["welfare_difference_q975"] = _quantile(paired_welfare_differences, 0.975)
            item["probability_welfare_difference_positive"] = statistics.mean(
                value > 0 for value in paired_welfare_differences
            )
            if regime == "bilateral":
                screen = "counterfactual"
            else:
                false_denial_rate = float(item["false_denials_mean"]) / max(
                    float(item["attempted_mean"]), 1.0
                )
                rights_pass = (
                    false_denial_rate <= decision["rights_false_denial_ceiling"]
                    and abs(float(item["group_gap_mean"]))
                    <= decision["rights_group_gap_ceiling"]
                )
                cost_indeterminate = (
                    float(item["cost_difference_q025"]) <= decision["cost_tolerance"]
                    <= float(item["cost_difference_q975"])
                )
                welfare_indeterminate = (
                    float(item["welfare_difference_q025"])
                    <= decision["welfare_tolerance"]
                    <= float(item["welfare_difference_q975"])
                )
                cost_pass = float(item["cost_difference_q975"]) < decision["cost_tolerance"]
                welfare_pass = (
                    float(item["welfare_difference_q025"])
                    > decision["welfare_tolerance"]
                )
                if not rights_pass:
                    screen = "constraint-rejected"
                elif cost_indeterminate or welfare_indeterminate:
                    screen = "direction-indeterminate"
                elif cost_pass and welfare_pass:
                    screen = "cost-and-welfare-favorable"
                elif cost_pass and not welfare_pass:
                    screen = "cost-favorable-welfare-adverse"
                elif not cost_pass and welfare_pass:
                    screen = "cost-adverse-welfare-favorable"
                else:
                    screen = "cost-and-welfare-adverse"
            item["screen"] = screen
            item["false_denial_rate"] = (
                float(item["false_denials_mean"]) / max(float(item["attempted_mean"]), 1.0)
            )
            item["escalation_rate"] = (
                float(item["escalated_mean"]) / max(float(item["attempted_mean"]), 1.0)
            )
            item["unresolved_rate"] = (
                float(item["unresolved_mean"]) / max(float(item["attempted_mean"]), 1.0)
            )
            summaries.append(item)
    return summaries


def run_bilateral_error_sensitivity(rows: list[dict], config_path: Path) -> list[dict]:
    raw, common, scenarios = read_config(config_path)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((str(row["scenario"]), str(row["regime"])), []).append(row)
    error_value = common.correct_completion_value + common.false_decision_cost
    sensitivity_rows: list[dict] = []
    for scenario in scenarios:
        bilateral = {
            int(row["replicate"]): row
            for row in grouped[(scenario.id, "bilateral")]
        }
        for regime in ("protocol_escalation", "protocol_binary"):
            for row in grouped[(scenario.id, regime)]:
                baseline = bilateral[int(row["replicate"])]
                for error_rate in raw["sensitivity"]["bilateral_error_rates"]:
                    adjusted_bilateral_welfare = float(baseline["welfare_proxy"]) - (
                        float(baseline["attempted"]) * float(error_rate) * error_value
                    )
                    sensitivity_rows.append(
                        {
                            "scenario": scenario.id,
                            "scenario_label": scenario.label,
                            "regime": regime,
                            "replicate": int(row["replicate"]),
                            "bilateral_error_rate": float(error_rate),
                            "bilateral_attempted": int(baseline["attempted"]),
                            "bilateral_welfare_zero_error": float(
                                baseline["welfare_proxy"]
                            ),
                            "bilateral_welfare_adjusted": adjusted_bilateral_welfare,
                            "protocol_welfare": float(row["welfare_proxy"]),
                            "welfare_difference_adjusted": float(row["welfare_proxy"])
                            - adjusted_bilateral_welfare,
                        }
                    )
    return sensitivity_rows


def summarize_bilateral_error_sensitivity(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row["scenario"]),
                str(row["regime"]),
                float(row["bilateral_error_rate"]),
            ),
            [],
        ).append(row)
    summaries: list[dict] = []
    for key in sorted(grouped):
        records = grouped[key]
        differences = [float(row["welfare_difference_adjusted"]) for row in records]
        summaries.append(
            {
                "scenario": key[0],
                "scenario_label": records[0]["scenario_label"],
                "regime": key[1],
                "bilateral_error_rate": key[2],
                "welfare_difference_mean": statistics.mean(differences),
                "welfare_difference_q025": _quantile(differences, 0.025),
                "welfare_difference_q975": _quantile(differences, 0.975),
                "probability_welfare_difference_positive": statistics.mean(
                    difference > 0 for difference in differences
                ),
            }
        )
    return summaries


def summarize_scaling(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((str(row["family"]), int(row["agent_count"])), []).append(row)
    summaries: list[dict] = []
    for family in SCALING_FAMILIES:
        agent_counts = sorted(n for candidate_family, n in grouped if candidate_family == family)
        for agent_count in agent_counts:
            records = grouped[(family, agent_count)]
            savings = [float(row["setup_saving"]) for row in records]
            summaries.append(
                {
                    "family": family,
                    "agent_count": agent_count,
                    "edge_count_mean": statistics.mean(float(row["edge_count"]) for row in records),
                    "density_mean": statistics.mean(float(row["density"]) for row in records),
                    "bilateral_setup_cost_mean": statistics.mean(
                        float(row["bilateral_setup_cost"]) for row in records
                    ),
                    "protocol_setup_cost_mean": statistics.mean(
                        float(row["protocol_setup_cost"]) for row in records
                    ),
                    "setup_saving_mean": statistics.mean(savings),
                    "setup_saving_q025": _quantile(savings, 0.025),
                    "setup_saving_q975": _quantile(savings, 0.975),
                    "probability_setup_saving_positive": statistics.mean(
                        value > 0 for value in savings
                    ),
                }
            )
    return summaries


def validate_case_coding(coding_path: Path, sources_path: Path) -> list[dict[str, str]]:
    with sources_path.open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    source_ids = {row["source_id"] for row in source_rows}
    with coding_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    expected_variables = {
        "repetition",
        "formalizability",
        "cross_setting_regularity",
        "exception_frequency",
        "externalities",
        "exit_switching",
        "reversibility",
        "observability",
        "power_symmetry",
    }
    valid_directions = {"favorable", "unfavorable", "mixed", "unknown"}
    valid_confidence = {"high", "medium", "low"}
    seen: set[tuple[str, str]] = set()
    case_variables: dict[str, set[str]] = {}
    for row in rows:
        key = (row["case_id"], row["variable"])
        if key in seen:
            raise ValueError(f"Duplicate case-variable row: {key}")
        seen.add(key)
        case_variables.setdefault(row["case_id"], set()).add(row["variable"])
        if row["variable"] not in expected_variables:
            raise ValueError(f"Unknown variable: {row['variable']}")
        if row["direction"] not in valid_directions:
            raise ValueError(f"Unknown direction: {row['direction']}")
        if row["confidence"] not in valid_confidence:
            raise ValueError(f"Unknown confidence: {row['confidence']}")
        if row["source_id"] not in source_ids:
            raise ValueError(f"Unknown source ID: {row['source_id']}")
        if not row["locator"].strip() or not row["evidence_note"].strip():
            raise ValueError(f"Missing evidence detail: {key}")
    for case_id, variables in case_variables.items():
        if variables != expected_variables:
            raise ValueError(f"Incomplete variable set for {case_id}: {variables}")
    return rows


def summarize_coding(rows: list[dict[str, str]]) -> list[dict[str, str | int | float]]:
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    result: list[dict[str, str | int | float]] = []
    for case_id, records in by_case.items():
        counts = Counter(row["direction"] for row in records)
        known = len(records) - counts["unknown"]
        result.append(
            {
                "case_id": case_id,
                "case_label": records[0]["case_label"],
                "favorable": counts["favorable"],
                "unfavorable": counts["unfavorable"],
                "mixed": counts["mixed"],
                "unknown": counts["unknown"],
                "known_coverage": known,
                "coverage_share": known / len(records),
                "confidence_high": sum(row["confidence"] == "high" for row in records),
                "confidence_medium": sum(row["confidence"] == "medium" for row in records),
                "confidence_low": sum(row["confidence"] == "low" for row in records),
            }
        )
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_summary(path: Path, summaries: list[dict]) -> None:
    rows = [row for row in summaries if row["regime"] != "bilateral"]
    labels = {
        "stable_high_volume": "Stable high volume",
        "high_exception": "High exception",
        "induced_unequal_access": "Induced unequal access",
        "low_volume_fixed_cost": "Low volume",
        "post_drift_high_error": "Post-drift high error",
    }
    regimes = {
        "protocol_escalation": "Escalation",
        "protocol_binary": "Binary ablation",
    }
    screens = {
        "cost-and-welfare-favorable": "Both favorable",
        "cost-favorable-welfare-adverse": "Cost only",
        "cost-adverse-welfare-favorable": "Welfare only",
        "cost-and-welfare-adverse": "Neither",
        "constraint-rejected": "Constraint rejected",
        "direction-indeterminate": "Indeterminate",
    }
    lines = [
        "\\begin{tabular}{@{}llrrrl@{}}",
        "\\toprule",
        "Scenario & Regime & $\\Delta C$ [95\\%] & $\\Delta W$ [95\\%] & FD & Screen \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{labels[row['scenario']]} & {regimes[row['regime']]} & "
            f"{row['cost_difference_vs_bilateral']:.0f} "
            f"[{row['cost_difference_q025']:.0f}, {row['cost_difference_q975']:.0f}] & "
            f"{row['welfare_difference_vs_bilateral']:.0f} "
            f"[{row['welfare_difference_q025']:.0f}, {row['welfare_difference_q975']:.0f}] & "
            f"{row['false_denial_rate']:.3f} & {screens[row['screen']]} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")


def write_latex_case_summary(path: Path, rows: list[dict]) -> None:
    lines = [
        "\\begin{tabular}{@{}lrrrrr@{}}",
        "\\toprule",
        "Case & Favorable & Unfavorable & Mixed & Unknown & Coverage \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['case_label']} & {row['favorable']} & {row['unfavorable']} & "
            f"{row['mixed']} & {row['unknown']} & {row['known_coverage']}/9 \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")


def write_latex_scaling_summary(path: Path, rows: list[dict]) -> None:
    labels = {
        "complete": "Complete",
        "fixed_density": "Fixed density (0.10)",
        "bounded_degree": "Degree-four ring",
    }
    growth = {
        "complete": "$N(N-1)/2$",
        "fixed_density": "$0.10N(N-1)/2$ (expected)",
        "bounded_degree": "$2N$",
    }
    lines = [
        "\\begin{tabular}{@{}llllr@{}}",
        "\\toprule",
        "Graph family & Edge growth & Robust crossover & $S_0$ at $N=256$ [95\\%] & $\\Pr(S_0>0)$ \\\\",
        "\\midrule",
    ]
    for family in SCALING_FAMILIES:
        family_rows = sorted(
            (row for row in rows if row["family"] == family),
            key=lambda row: int(row["agent_count"]),
        )
        robust = next(
            (
                int(row["agent_count"])
                for row in family_rows
                if float(row["setup_saving_q025"]) > 0
            ),
            None,
        )
        endpoint = family_rows[-1]
        crossover = str(robust) if robust is not None else "None tested"
        lines.append(
            f"{labels[family]} & {growth[family]} & {crossover} & "
            f"{endpoint['setup_saving_mean']:.0f} "
            f"[{endpoint['setup_saving_q025']:.0f}, {endpoint['setup_saving_q975']:.0f}] & "
            f"{endpoint['probability_setup_saving_positive']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")


def plot_quadrants(path: Path, summaries: list[dict]) -> None:
    rows = [row for row in summaries if row["regime"] != "bilateral"]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    markers = {"protocol_escalation": "o", "protocol_binary": "s"}
    colors = {
        "stable_high_volume": "#0072B2",
        "high_exception": "#D55E00",
        "induced_unequal_access": "#009E73",
        "low_volume_fixed_cost": "#E69F00",
        "post_drift_high_error": "#CC79A7",
    }
    scenario_labels = {
        "stable_high_volume": "stable high volume",
        "high_exception": "high exception",
        "induced_unequal_access": "induced unequal access",
        "low_volume_fixed_cost": "low volume fixed cost",
        "post_drift_high_error": "post-drift high error",
    }
    label_offsets = {
        ("stable_high_volume", "protocol_escalation"): (6, -14),
        ("high_exception", "protocol_escalation"): (6, 6),
        ("induced_unequal_access", "protocol_escalation"): (-142, 8),
        ("low_volume_fixed_cost", "protocol_escalation"): (-132, 10),
        ("post_drift_high_error", "protocol_escalation"): (6, 6),
    }
    for row in rows:
        ax.errorbar(
            row["cost_difference_vs_bilateral"],
            row["welfare_difference_vs_bilateral"],
            xerr=[
                [row["cost_difference_vs_bilateral"] - row["cost_difference_q025"]],
                [row["cost_difference_q975"] - row["cost_difference_vs_bilateral"]],
            ],
            yerr=[
                [row["welfare_difference_vs_bilateral"] - row["welfare_difference_q025"]],
                [row["welfare_difference_q975"] - row["welfare_difference_vs_bilateral"]],
            ],
            fmt=markers[row["regime"]],
            color=colors[row["scenario"]],
            ecolor=colors[row["scenario"]],
            elinewidth=0.9,
            capsize=2.5,
            markersize=6.8,
            markeredgecolor="black",
            markeredgewidth=0.5,
            alpha=0.88,
        )
        if row["regime"] == "protocol_escalation":
            ax.annotate(
                scenario_labels[row["scenario"]],
                (
                    row["cost_difference_vs_bilateral"],
                    row["welfare_difference_vs_bilateral"],
                ),
                xytext=label_offsets[(row["scenario"], row["regime"])],
                textcoords="offset points",
                fontsize=7,
            )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlim(-10_800, 1_050)
    ax.set_ylim(-10_200, 10_800)
    ax.set_xlabel(r"Paired accounted-cost difference, $\Delta C$ (negative is lower)")
    ax.set_ylabel(r"Paired welfare difference, $\Delta W$ (positive is higher)")
    ax.set_title("Designed cost and welfare differences need not coincide")
    ax.grid(alpha=0.2)
    ax.scatter([], [], marker="o", color="white", edgecolor="black", label="Escalation protocol")
    ax.scatter([], [], marker="s", color="white", edgecolor="black", label="Binary ablation")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_case_audit(path: Path, case_rows: list[dict]) -> None:
    labels = [str(row["case_label"]) for row in case_rows]
    categories = ["favorable", "mixed", "unfavorable", "unknown"]
    colors = ["#2a9d8f", "#e9c46a", "#e76f51", "#adb5bd"]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    left = np.zeros(len(case_rows))
    for category, color in zip(categories, colors, strict=True):
        values = np.array([int(row[category]) for row in case_rows])
        ax.barh(labels, values, left=left, label=category.title(), color=color)
        left += values
    ax.set_xlim(0, 9)
    ax.set_xlabel("Frontier variables (count; no aggregate score)")
    ax.set_title("Retrospective evidence-availability audit")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.10), frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_scaling(path: Path, rows: list[dict]) -> None:
    labels = {
        "complete": "Complete",
        "fixed_density": "Fixed density 0.10",
        "bounded_degree": "Degree-four ring",
    }
    colors = {
        "complete": "#0072B2",
        "fixed_density": "#E69F00",
        "bounded_degree": "#CC79A7",
    }
    styles = {
        "complete": ("o", "-"),
        "fixed_density": ("s", "--"),
        "bounded_degree": ("^", "-."),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6), sharex=True)
    for family in SCALING_FAMILIES:
        family_rows = sorted(
            (row for row in rows if row["family"] == family),
            key=lambda row: int(row["agent_count"]),
        )
        x = np.array([int(row["agent_count"]) for row in family_rows])
        mean = np.array([float(row["setup_saving_mean"]) for row in family_rows])
        low = np.array([float(row["setup_saving_q025"]) for row in family_rows])
        high = np.array([float(row["setup_saving_q975"]) for row in family_rows])
        for ax in axes:
            ax.plot(
                x,
                mean,
                marker=styles[family][0],
                linestyle=styles[family][1],
                linewidth=1.6,
                label=labels[family],
                color=colors[family],
            )
            ax.fill_between(x, low, high, alpha=0.16, color=colors[family])
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64, 128, 256])
        ax.set_xticklabels(["8", "16", "32", "64", "128", "256"])
        ax.set_xlabel("Agents, $N$")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Fixed setup saving $S_0$")
    axes[0].set_title("Full range")
    axes[1].set_ylim(-1_600, 1_600)
    axes[1].set_title("Crossover detail")
    fig.suptitle("Agreement-compression scaling depends on graph structure", y=0.99)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root: Path) -> dict[str, str | bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def write_manifest(
    path: Path,
    project_root: Path,
    input_paths: list[Path],
    output_paths: list[Path],
    command: str,
    master_seed: int,
) -> None:
    manifest = {
        "schema_version": 1,
        "command": command,
        "master_seed": master_seed,
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            "numpy": version("numpy"),
            "matplotlib": version("matplotlib"),
        },
        "git": _git_state(project_root),
        "inputs": {str(path.relative_to(project_root)): sha256(path) for path in input_paths},
        "outputs": {str(path.relative_to(project_root)): sha256(path) for path in output_paths},
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_all(experiment_root: Path) -> dict[str, Path]:
    project_root = experiment_root.parent
    config_path = experiment_root / "configs" / "prespecified.toml"
    coding_path = experiment_root / "data" / "cases" / "coding.csv"
    sources_path = experiment_root / "data" / "cases" / "sources.csv"
    tables = experiment_root / "results" / "tables"
    figures = experiment_root / "results" / "figures"
    manifests = experiment_root / "results" / "manifests"
    for directory in (tables, figures, manifests):
        directory.mkdir(parents=True, exist_ok=True)

    raw_rows = run_simulation(config_path)
    summaries = summarize(raw_rows, config_path)
    sensitivity_rows = run_bilateral_error_sensitivity(raw_rows, config_path)
    sensitivity_summary = summarize_bilateral_error_sensitivity(sensitivity_rows)
    scaling_rows = run_scaling(config_path)
    scaling_summary = summarize_scaling(scaling_rows)
    coding_rows = validate_case_coding(coding_path, sources_path)
    case_summary = summarize_coding(coding_rows)

    outputs = {
        "simulation_raw": tables / "simulation_replicates.csv",
        "simulation_summary": tables / "simulation_summary.csv",
        "sensitivity_raw": tables / "bilateral_error_sensitivity_replicates.csv",
        "sensitivity_summary": tables / "bilateral_error_sensitivity_summary.csv",
        "scaling_raw": tables / "scaling_replicates.csv",
        "scaling_summary": tables / "scaling_summary.csv",
        "case_summary": tables / "case_summary.csv",
        "simulation_tex": tables / "simulation_summary.tex",
        "scaling_tex": tables / "scaling_summary.tex",
        "case_tex": tables / "case_summary.tex",
        "quadrant_figure": figures / "cost_welfare_quadrants.png",
        "scaling_figure": figures / "agreement_compression_scaling.png",
        "case_figure": figures / "case_audit.png",
        "manifest": manifests / "run_manifest.json",
    }
    write_csv(outputs["simulation_raw"], raw_rows)
    write_csv(outputs["simulation_summary"], summaries)
    write_csv(outputs["sensitivity_raw"], sensitivity_rows)
    write_csv(outputs["sensitivity_summary"], sensitivity_summary)
    write_csv(outputs["scaling_raw"], scaling_rows)
    write_csv(outputs["scaling_summary"], scaling_summary)
    write_csv(outputs["case_summary"], case_summary)
    write_latex_summary(outputs["simulation_tex"], summaries)
    write_latex_scaling_summary(outputs["scaling_tex"], scaling_summary)
    write_latex_case_summary(outputs["case_tex"], case_summary)
    plot_quadrants(outputs["quadrant_figure"], summaries)
    plot_scaling(outputs["scaling_figure"], scaling_summary)
    plot_case_audit(outputs["case_figure"], case_summary)

    raw, _, _ = read_config(config_path)
    generated = [path for key, path in outputs.items() if key != "manifest"]
    source_inputs = sorted((experiment_root / "src" / "protocol_frontier").glob("*.py"))
    write_manifest(
        outputs["manifest"],
        project_root=project_root,
        input_paths=[
            experiment_root / "pyproject.toml",
            experiment_root / "uv.lock",
            config_path,
            coding_path,
            sources_path,
            *source_inputs,
        ],
        output_paths=generated,
        command="uv run protocol-frontier run-all",
        master_seed=int(raw["master_seed"]),
    )
    return outputs
