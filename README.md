# Scotland frugal-EV screening: reproduction package

Data and code for **What Share of Daily Local Car Trips Is Potentially Suitable
for Low-Cost, Short-Range Electric Vehicles? A Scenario-Based Screening Framework
Integrating Demographics, Infrastructure, and Trip Purpose**.

This package contains the final analysis, a clean executable manuscript, and the
inputs needed for downstream reproduction. It contains no reviewer correspondence
or revision comparisons. Exact regeneration of the original synthetic trips and
frozen adoption scores is outside the verified scope; see [DATA.md](DATA.md).

## Run

Use Python 3.13. Download `paper1-data.tar.gz` from the matching
[GitHub release](https://github.com/wangzhao0217/ZEV-UP-Modelling/releases).
Extract it into this directory so that `data/flows/` and `manuscript/figures/`
exist. From the package directory:

```sh
tar -xzf paper1-data.tar.gz
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py check
python reproduce.py
```

The run recomputes the seven-region screening and shared charging capacity,
126 sensitivity rows, 28 alternative-screen rows, 56 chaining rows, geographic
BEV comparisons, postcode regressions, threshold diagnostics, 280 vehicle
assignment settings, trip-purpose robustness experiments, AP-band membership
stability and matched-route agreement statistics. It compares
results with the supplied reference tables and fails if they disagree.
Actual checks are recorded in `results/verification.json`.

The reference calculation retains 1,454,960 routed observations representing
8,016,915.314 car-person trips/day. The theoretical volume is 1,955,936.271;
1,177,137.393 pass the shared capacity check and 778,798.877 are capacity constrained.
These are scenario outputs, not forecasts of adoption.

## Render

Install R and Quarto, then the R packages listed in `environment/R-packages.csv`:

```sh
Rscript scripts/install_r.R
python reproduce.py render
python reproduce.py render --format pdf
```

HTML is the default; PDF also needs a working LaTeX installation. The renderer
uses the same Python executable as `reproduce.py`. Maps, the framework and the
SPT biplot are retained manuscript assets; their source code and spatial inputs
are included separately. Statistical figures and executable manuscript tables
are calculated during reproduction/rendering. To regenerate a static scientific
figure separately, run `python scripts/figures.py adoption`, `charging`, or
`biplot`; these write to `results/figures/`. The framework source is
`scripts/figure_sources/framework.mmd`.

## Contents

| Location | Contents |
|---|---|
| `reproduce.py` | Check, calculate, and render commands |
| `functions/`, `scripts/` | Model functions and analysis entry points |
| `data/` | Model inputs, statistical inputs, and reference outputs |
| `manuscript/` | Current scientific content, bibliography and figures |
| `tests/` | Focused scientific checks (`python -m pytest -q tests`) |
| `manifest.json`, `provenance.json`, `coverage.csv` | SHA-256 inventory, original sources and table/figure coverage |
| `DATA.md` | Data dictionary, source access, scope and redistribution status |

The code repository stays small: data and figure assets are in the accompanying
archive, and generated results are ignored by Git. Public
repository: [ZEV-UP-Modelling](https://github.com/wangzhao0217/ZEV-UP-Modelling).
Original code is available under the [MIT licence](LICENSE). Data and third-party
materials retain the terms described in [DATA.md](DATA.md) and [NOTICE](NOTICE).
