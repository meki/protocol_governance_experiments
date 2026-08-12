# Protocolization Frontier evidence package

This directory contains three bounded forms of evidence for the manuscript:

1. a fixed-cost scaling sweep that directly instantiates agents, compatibility edges, bespoke mappings, and local adapters;
2. a designed case-flow simulation comparing bilateral coordination, a protocol with explicit escalation, and a binary protocol ablation; and
3. a retrospective evidence-availability audit of X-tee, eIDAS 1.0, SyRI, and Robodebt across the nine frontier variables.

None is a population-level validation. The simulations instantiate the paper's own concepts and therefore test scaling and accounting behavior. The case audit uses a purposive four-case sample, does not infer missing values, and does not construct an aggregate success score.

## Reproduce

Requires Python 3.12 and `uv`.

```sh
cd experiments
uv sync --frozen
uv run pytest
uv run protocol-frontier run-all
```

The last command regenerates every file under `results/`. The original case-flow design and the later scaling and bilateral-error sensitivity designs are preserved in `configs/prespecified.toml`; none was registered in an external registry. The two later designs were frozen in dated internal protocols before their respective first executions. Retrospective coding and source identities are under `data/cases/`. Initial `uv sync` may require network access to obtain pinned Python and package artifacts.

## Output interpretation

- `simulation_replicates.csv`: all stochastic replicates; no scenario is selectively omitted.
- `simulation_summary.csv`: means and 95% simulation intervals plus the separate cost/welfare screen.
- `bilateral_error_sensitivity_replicates.csv`: paired welfare differences after deterministic expected adjustments for bilateral error rates of 0%, 1%, 3%, and 5%.
- `bilateral_error_sensitivity_summary.csv`: mean and 95% simulation intervals for that assumption sensitivity.
- `scaling_replicates.csv`: all fixed-cost agreement-compression sweeps across agent counts and graph families.
- `scaling_summary.csv`: mean fixed setup savings, simulation intervals, and crossover probabilities.
- `case_summary.csv`: direction counts and evidence coverage, not an aggregate score.
- `run_manifest.json`: input/output hashes, dependency versions, seed, command, and Git state.

The cost ledger and welfare proxy are intentionally separate. `constraint-rejected` is triggered by pre-specified false-denial or between-group legibility-gap ceilings and overrides a favorable cost result.

The bilateral regime is a strong processing baseline: every attempted case completes correctly. The later sensitivity weakens only that assumption by calculating `W_B(q) = W_B(0) - attempted_B * q * (correct_completion_value + false_decision_cost)` at each fixed error rate; it holds volume, resource cost, protocol outcomes, and constraints unchanged. This expected adjustment isolates baseline dependence without sampling additional error events and is not a calibrated estimate. In `protocol_escalation`, encoding failures and decisional uncertainty are resolved correctly at a fixed additional resource cost; in `protocol_binary`, encoding failures become denials and uncertain represented cases remain unresolved. These are designed mechanism assumptions, not estimates of real bilateral or human-review performance. The group-gap constraint uses the absolute difference between disadvantaged and reference-group protocol-legibility gaps.

## Audit coding rule

The retrospective audit designates one authoritative primary document per case and asks what that document makes measurable. `unknown` therefore means "not established in the designated document," not "absent from all public evidence." A direction is `favorable` or `unfavorable` when the document directly reports a condition or finding that moves the named frontier variable. `mixed` records material evidence in both directions or an explicit boundary on inference. Recommendations are used directionally only when the issuing body identifies the missing safeguard or process as a response to a documented deficiency; the evidence note must make that inference visible. Confidence is `high` for direct findings or operational statements, `medium` for bounded interpretation, and `low` when the designated document does not establish direction. No missing direction is imputed.

The generated `case_audit.png` is retained as an artifact-level diagnostic; the manuscript uses the compact table instead. The manifest's Git commit identifies the parent state at generation time when the worktree is dirty; input hashes identify the executed configuration, code, lock file, and coding data. Output hashes establish byte identity in the recorded environment; floating-point reductions and Matplotlib rendering can differ at insignificant final digits or pixels across NumPy, BLAS, operating-system, or font-stack versions, so cross-environment reproduction should compare reported quantities within stated precision rather than require identical output hashes.
