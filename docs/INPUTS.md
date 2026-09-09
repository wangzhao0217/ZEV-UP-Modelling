# R2 input contract and remaining national-release requirements

## Prepared-flow scenario runner

`reproduce.py` accepts explicit CSV or Parquet files. It never downloads private inputs or fills missing scientific data with reference values.

| Flow column | Meaning and units |
|---|---|
| `vol` | Retained daily car-person trip volume, finite and nonnegative. |
| `w` | Continuous purpose-suitability weight in [0, 1]. Raw cases explicitly use 1 instead. |
| `ap` | Origin OA adoption-propensity score in [0, 1], not a purchase probability. |
| `d` | One-way routed distance in kilometres, not metres or straight-line distance. |
| `home_score` | Origin home-charging score in [0, 1]; available at score >= 0.5. |
| `P_i`, `P_j` | Binary origin/destination public-charging access flags. |
| `occupancy` | Assumed persons per vehicle, finite and >= 1. |
| `charging_station_position` | Zero-based position in the unique physical-station table; -1 when destination access is absent. |

The station table contains `station_position` and integer nonnegative `N_points`; positions must be ordered uniquely from 0. Do not duplicate the same physical station for different destinations, and do not reuse local station positions across regions without remapping. The runner preserves supplied input order.

The default scenario uses an inclusive AP band [0.30, 0.80], 80 km effective range, a 40 km short-trip threshold, and the existing implementation's capacity settings. A short trip requires home **or** origin public access. A medium trip requires home **and** destination public access, plus sufficient pooled destination capacity. Trips over 80 km remain in the denominator but not in the potential numerator. Capacity uses weighted vehicle equivalents and the specified average-day point-hour model; it is not a queue simulation.

`examples/synthetic_flows.csv` and `synthetic_stations.csv` are fictional unit-test inputs. No actual person, household, address or observed trip is represented.

## Independent archived-result reconciliation

The verifier expects this layout under `--run-dir`:

```text
final/
  national_metrics.json
  adoption_attributes.parquet
  attributes/
    HITRANS.parquet
    Nestrans.parquet
    SESTRAN.parquet
    SPT.parquet
    SWESTRANS.parquet
    Tactran.parquet
    ZetTrans.parquet
```

OA attributes require unique `geo_code` and `adoption_propensity`. Each regional flow additionally requires `origin_geo_code`, `N_ij`, `Cap_j` and `cat`, along with `vol`, `w`, `ap`, `d`, `home_score`, `P_i` and `P_j`. In the archived category convention, immediate feasibility is `cat == "fully_feasible"`. `national_metrics.json` supplies `regions`, `oa_count`, `oa_in_band` and the eight full-precision totals checked by the verifier. Do not reconstruct those totals from rounded manuscript tables to make a check pass.

These files were documented for the authors' R2 run but **were absent from the tracked source snapshot used for this transfer**. No external download location has been invented. Original release manifests/checksums and the recorded runtime environment must accompany an eventual compact-data release. Additional station point counts and flow-to-station assignments are required to recompute capacity, rather than merely reconcile stored flags.

## Upstream inputs and access

A full rebuild additionally requires the exact source vintages, processing scripts and dependency environment for census scoring, twelve-purpose demand generation, OSRM routing and output collection. In particular, the documented `EV_od` national generation source was not tracked in the available snapshot. These missing stages must be supplied before claiming an end-to-end rebuild.

| Input family | Provider/access requirement |
|---|---|
| Census demographics and OA/DZ lookup | National Records of Scotland / Scotland's Census. Obtain the 2022 tables and relevant earlier production controls; retain definitions, release dates and checksums. |
| Journey-purpose shares | Scottish Household Survey. Retain the exact imported table and the purpose-exclusion/redistribution rules. |
| Public charging | OpenChargeMap. Obtain the recorded snapshot, site deduplication and point-count definitions. Provider and contributed-data terms must be checked. |
| Road network/routing | OpenStreetMap / Geofabrik and OSRM. Record the exact extract, routing profile, engine version and snapping policy; a new latest extract is a different scenario. |
| POIs | Project-licensed Ordnance Survey/source POI data; obtain through an authorised account. Raw files are not redistributed. |
| Occupancy and trip chaining | Department for Transport National Travel Survey tables. Record the workbook, selected historical year, purpose mapping and imported values. |
| Registration corroboration | DVSA/DfT MOT and registration sources with the applicable access conditions. No raw vehicle or postcode records are included here. |

Official provider entry points: https://www.scotlandscensus.gov.uk/ ; https://www.transport.gov.scot/our-approach/statistics/ ; https://openchargemap.org/ ; https://download.geofabrik.de/ ; https://www.gov.uk/government/statistical-data-sets/nts09-vehicle-mileage-and-occupancy . These are acquisition entry points, not checksum-matched replacements for the historical run.

## Release gate for additional data

Before adding any real derived data, the authors should confirm redistribution rights and disclosure risk; retain only fields needed by the documented calculations; remove credentials and original machine paths; validate schemas and hashes; and reconcile the regenerated results against the full-precision release. Compact synthetic flows can still inherit source-data restrictions. Do not copy a whole internal computational archive: it may contain private drafts and review material irrelevant to reproduction.
