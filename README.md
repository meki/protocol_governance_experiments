# Protocol Governance Experiments

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21905375.svg)](https://doi.org/10.5281/zenodo.21905375)

This repository contains the complete computational evidence package for the article **Agreement Compression: From Counterculture to Protocol Governance** by Satoshi Sashida (Independent Researcher).

The package contains three bounded forms of evidence:

1. a fixed-cost graph sweep that directly instantiates the paper's agreement-compression mechanism;
2. a designed case-flow simulation of resource, welfare, and constraint screens, including a bilateral-error sensitivity analysis; and
3. a retrospective evidence-availability audit of X-tee, eIDAS 1.0, SyRI, and Robodebt.

These studies are mechanism checks and a documentary audit. They are not population-level validation, causal estimates, or evidence that the four proposed public interfaces have been implemented.

## Reproduce

The locked environment requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
cd experiments
uv sync --frozen
uv run pytest
uv run protocol-frontier run-all
```

The final command regenerates every declared table, figure, replicate file, and the run manifest under `experiments/results/`. Detailed assumptions, output interpretation, audit-coding rules, and portability limits are documented in [`experiments/README.md`](experiments/README.md).

## Repository contents

- `experiments/configs/`: fixed simulation, scaling, and sensitivity configuration
- `experiments/data/cases/`: 36-cell coding matrix and authoritative-source register
- `experiments/src/`: Python implementation
- `experiments/tests/`: regression and invariant tests
- `experiments/results/`: all reported replicates, summaries, LaTeX tables, figures, and run manifest
- `CITATION.cff` and `.zenodo.json`: archival citation metadata

## Version

The current archival release is version 0.1.1. The release tag, rather than the mutable default branch, identifies the evidence package used for citation and preservation. Version 0.1.1 changes only the package metadata and the development-only pytest pin; the evidence inputs, analysis implementation, and reported outputs are unchanged from version 0.1.0.

Current archived release: [Sashida (2026), *Protocol Governance Experiments: Evidence Package for Agreement Compression*, version 0.1.1](https://doi.org/10.5281/zenodo.21905375). The concept DOI for all archived versions is `10.5281/zenodo.21905317`. Version 0.1.0 remains available at [10.5281/zenodo.21905318](https://doi.org/10.5281/zenodo.21905318).

## License

The software and repository materials are released under the MIT License. The source documents listed in the retrospective audit remain subject to their respective publishers' terms and are linked rather than redistributed.
