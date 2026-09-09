# Paper 1 R2: reproducibility scope

This is a **calculation-rule release**, not a complete national data release.
It separates two tasks that must not be confused:

1. Exercise the corrected screening/weighting/charging equations on explicit
   inputs, including the four AP/purpose comparisons and chaining stresses.
   This is supported by the included modules, fixtures and tests.
2. Reproduce Scotland-wide results from source data and the original demand
   generator. This still needs the frozen national archive listed below.
   Passing the included tests does not reproduce or empirically validate 36.5%.

## Quick start

Run from the repository root in a Python 3.11+ environment. The isolated package
was tested under Python 3.13.5; `requirements-paper1-r2.txt` pins the package
versions exercised in that environment, not the missing historical run environment.

```bash
python -m pip install -r requirements-paper1-r2.txt
python -m pytest -q tests/paper1_r2
python -m paper1_r2 --flows data/paper1_r2/illustrative_flows.csv --stations data/paper1_r2/illustrative_stations.csv --input-kind illustrative --output results/illustrative-comparison.json
```

The command recomputes station loads in all four cases. Purpose-weighted cases
use W in both reported volume and demand; raw cases use W=1 in both. All shares
within a case use that case's supplied retained-demand denominator. The fixture
is only eight artificial flows. Its baseline volume is 88, not 6.70 million.
The expected fixture outputs are in `data/paper1_r2/illustrative_expected.json`.
They are regression targets for the fixture, not a national results table.

## Scope and modules

- `paper1_r2/feasibility.py`: range, continuous suitability weights and the
  inclusive demographic gate; complete-tier partition and fixed-support tilts.
- `charging.py` and `scenarios.py`: charger access, vehicle-equivalent loads,
  shared station capacity and AP/raw-versus-weighted comparisons.
- `purpose.py` and `purpose_types.py`: explicit purpose-label mapping and the
  configured base/temporal suitability blend.
- `chaining.py`: declared exposure/failure losses on the home-short subset and
  idealised two-stop geometries. These are not observed Scottish tour bounds.

The calculation bodies were transferred unchanged. Imports were retargeted to
this small package so running it does not import the repository's legacy GIS,
plotting and optional machine-learning stack. The purpose enum was extracted
verbatim. Source fingerprints and adaptations are recorded in
`source_manifest.json`. Tests of the legacy charging-flexibility assessor are
not claimed: five label tests here check the canonical mapping only.

The existing top-level `regional_analysis.qmd` and `functions/` tree predate this
scoped release. They are left intact and should not be assumed to be the pinned
national R2 workflow. Use the explicit commands above for this release.

## Inputs still needed for full national reproduction

The private repository's tracked tree and releases did not contain the complete
national archive. No placeholder has been substituted for it. Required items:

- The actual OD-generation scripts under `EV_od/`, exact negative-beta parameter
  file, production controls, purpose mappings, seeds and generation manifests.
- The frozen national run's `final/national_metrics.json`,
  `final/adoption_attributes.parquet` and `final/attributes/<RTP>.parquet`.
  Each flow table must carry the original `vol`, `w`, `ap`, `d`, origin OA,
  charging/access/capacity fields and saved tier attributes.
- Input checksums, filtering/routing ledgers, pinned routing/map build and the
  original environment/package record. Current dependency pins alone cannot
  recover a historical routing service or stochastic demand generation.
- Lawful access to source census/geography, charging inventory and relevant
  POIs. Restricted MOT/POI inputs must be obtained on their own applicable terms.

When a vetted frozen archive is supplied separately, the transferred verifier
can reconcile saved flow-level attributes to the national metrics:

```bash
python -m pip install pyarrow
python tools/verify_packaged_paper1_r2.py --run-dir /path/to/frozen-run --output results/national-identities.json
```

That verifier checks saved numeric identities and OA joins. It does **not**
regenerate demand, reroute trips or independently validate the model. PyArrow
is optional and the archive verification command has not been exercised here
because the national Parquet inputs are absent. The optional `--package-manifest`
uses repository-relative paths, as in the original script; do not pass the
module source manifest as if it were the missing national archive manifest.

## Disclosure and rights

Only selected calculation code, regression tests, small aggregate assumptions,
artificial fixtures and documentation were transferred. No private repository
history, reviewer correspondence, manuscripts, drafts, credentials or personal
records were copied. No repository visibility was changed. A software licence
has not been selected in this pass; see `SOURCE_RIGHTS.md` before redistribution.
A complete national archive and an explicit software licence remain release
requirements, not completed deliverables.
