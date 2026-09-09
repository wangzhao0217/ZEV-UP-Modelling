# ZEV-UP-Modelling

Code supporting frugal-EV scenario modelling.

## Corrected Paper 1 R2 calculation rules

Start with [the scoped reproduction guide](reproducibility/paper1_r2/README.md).
It includes isolated screening, charging, purpose and chaining calculations,
small illustrative inputs, the published occupancy-assumption record and tests.

[Recorded aggregate tables](data/paper1_aggregate/README.md) provide nine
machine-readable transcriptions of the corrected manuscript results. They retain
reported rounding and must not be treated as independently regenerated model
outputs or substituted for missing flow-level inputs.

**The complete frozen Scottish national data archive is not included yet.**
The example and tests reproduce calculation rules, not the paper's national
estimates. The guide lists exactly what remains necessary for full national
reproduction and describes the separate saved-attribute verification script.

The legacy `regional_analysis.qmd` and `functions/` tree remain available but
are not represented as a complete pinned R2 national workflow.

See [source/data rights](SOURCE_RIGHTS.md). No new software licence has been
selected in this packaging pass.
