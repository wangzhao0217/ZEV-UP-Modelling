# Data included in this scoped release

`illustrative_flows.csv` and `illustrative_stations.csv` are newly constructed,
eight-flow/two-site test fixtures. They contain no real people, observed Scottish
trips, coordinates or charging-station identities. Volumes are arbitrary test
weights. They deliberately exercise range boundaries, the inclusive AP band,
short-trip access and competition for destination capacity. **Their outputs are
not national estimates and must never be cited as the paper's Scottish results.**

`charging_occupancy_R2.json` is an exact copy of the small parameter record used
in the R2 analysis. It identifies the 2019 England NTS0905 observations in the
2024 workbook edition, the workbook SHA-256, units and purpose mapping. These
are transferred scenario assumptions, not measured Scottish occupancy. Contains
public sector information licensed under the Open Government Licence v3.0.
Source: https://www.gov.uk/government/statistical-data-sets/nts09-vehicle-mileage-and-occupancy
Licence: https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
The original workbook is not bundled. Its aggregate values were copied from the
existing checked-in parameter record; the original workbook was not re-extracted
in this packaging pass. The illustrative fixture supplies its own occupancy
column and does not purport to apply the national purpose mix.

No raw census tables, MOT microdata, POI source files, modelled national flow
attributes, provider credentials or manuscript/reviewer files are included.
