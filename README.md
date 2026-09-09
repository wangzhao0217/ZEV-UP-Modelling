# ZEV-UP Modelling

## Paper 1: Round 2 calculation subset

This repository provides a small, reviewable implementation of the prepared-attribute calculations for:

**What Share of Daily Car Trips Is Potentially Suitable for Low-Cost, Short-Range Electric Vehicles? A Scenario-Based Screening Framework Integrating Demographics, Infrastructure, and Trip Purpose**

**Current scope:** the four R2 calculation modules, their original regression tests, a command-line example, fictional test inputs, rounded aggregate reference tables and an independent compact-archive verifier. This is **not yet the complete national reproduction archive**. The seven-region compact attributes, matching OA scores, station ledgers and upstream demand-generation inputs are not included. The example does not reproduce the Scottish headline.

The pre-existing `functions/` and `regional_analysis.qmd` are retained as legacy development material. They are not the entry point for the new R2 subset and should not be assumed to reproduce the corrected national release.

### Run the available example and tests

Use Python 3.11 or later. From this repository's root:

```bash
python -m venv .venv
# Activate the environment using the command appropriate to your operating system.
python -m pip install -r requirements-r2.txt
python -m pytest -q tests/test_paper1_feasibility.py tests/test_paper1_charging.py tests/test_paper1_scenarios.py tests/test_paper1_chaining.py
python reproduce.py --flows examples/synthetic_flows.csv --stations examples/synthetic_stations.csv --output build/synthetic_results.csv
```

The last command recalculates four cases: raw and purpose-weighted demand, each with and without the demographic screen. Station loads and available capacity are recomputed for every case. The input fixture contains **fictional** trips designed to exercise inclusive thresholds, charging-access failures and shared-station capacity. Its outputs are not estimates for Scotland.

`requirements-r2.txt` records the environment used to test this small package; it is not a claim about the historical national-run environment. Install `pyarrow` separately to read a supplied Parquet archive. CSV examples do not require it.

### What the calculation does

For each retained flow, the numerator is person-trip volume times a continuous purpose weight, a binary demographic scenario gate and a range indicator. Charging access and destination capacity split that numerator into immediate and conditional portions. The denominator retains all supplied demand, including flows outside the screen or range. AP and purpose weights are not purchase probabilities.

Modules in `paper1_r2/` are copied from the development implementation. Only package import paths were adjusted to avoid importing the unrelated legacy analysis stack. The independent verifier's package-manifest root was adapted to its new location. `transfer_manifest.json` records original and transferred SHA-256 fingerprints; arithmetic and original test assertions are unchanged.

### Inspect the national reference values

`reference/` contains four tables transcribed from the R2 manuscript: the charging comparison, regional results, purpose results and AP-band sensitivity. These are **rounded reference values**, not independently regenerated outputs or substitute input data. Table identifiers, captions and hashes are recorded in `transfer_manifest.json`.

### Reproduce the national results when the compact archive is available

See [the input contract](docs/INPUTS.md). The independent check is:

```bash
python verify_packaged_paper1_r2.py --run-dir /path/to/the/supplied/R2/run --output build/national_check.json
```

This needs `final/national_metrics.json`, `final/adoption_attributes.parquet` and every regional `final/attributes/<RTP>.parquet`. It independently checks origin-score joins, numerator identities, stored charging/tier consistency and national totals. It checks stored capacity flags; it does **not** independently regenerate those flags or the upstream OD population. `reproduce.py` recomputes capacity from supplied station point counts for a prepared regional input. Keep station identifiers unique when combining regional inputs.

Full origin-destination generation, geospatial processing, empirical registration checks and all sensitivity families require the additional source inputs and executable stages in `docs/INPUTS.md`; those stages have not been represented as complete here.

### Publication and licensing boundaries

No manuscript drafts, reviewer correspondence, internal assessments, repository history, credentials or raw MOT/POI records were transferred. A public repository is not a licence to redistribute third-party inputs. See [licensing status](docs/LICENSING.md) before reuse or adding data. The authors must complete the compact-data release and choose explicit code/data licences before describing this repository as a complete openly licensed national reproduction package.
