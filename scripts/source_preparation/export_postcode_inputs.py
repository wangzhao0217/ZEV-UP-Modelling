"""Export aggregate postcode inputs by executing the designated manuscript cell.

Requires the author-side MOT and spatial inputs in the original workspace.
Writes only to a new export directory; the manuscript and its outputs are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Execute the actual analysis and retain aggregates without vehicle records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        parser.error("--output-dir must be new")
    manuscript = ROOT / "paper/Revised_manuscript--with-changes-marked_R2_v4.qmd"
    source = manuscript.read_text()
    matches = [m.group(1) for m in re.finditer(r"^```\{python\}\n([\s\S]*?)^```", source, re.M)
               if "#| label: mot-fy-validation-py" in m.group(1)]
    if len(matches) != 1:
        raise ValueError("Expected exactly one postcode calculation cell")
    original = matches[0]
    substitutions = {
        'repo = Path("..").resolve()': f"repo = Path({str(ROOT)!r})",
        'repo / "output/paper1_r2_v4/2026-09-20/band_edge_fits.csv"': f"Path({str(out / 'band_edge_fits.csv')!r})",
        'figs_dir = repo / "paper" / "figs"': f"figs_dir = Path({str(out)!r})",
    }
    code = original
    for before, after in substitutions.items():
        if code.count(before) != 1:
            raise ValueError(f"Manuscript output contract changed: {before}")
        code = code.replace(before, after)
    out.mkdir(parents=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    namespace: dict = {}
    exec(compile(code, str(manuscript), "exec"), namespace)
    for name in ("table_fy_2011", "table_fy_1920_v_2022", "delta_table_fy", "val_hpi_2022",
                 "postcode_segment_counts", "segment_analysis", "top20_by_pc", "band3080_by_pc"):
        namespace[name].drop(columns=["geometry"], errors="ignore").to_csv(out / f"{name}.csv", index=False)
    namespace["edge_scores"][[namespace["SCORE_COL"], "pc_area"]].to_parquet(out / "edge_scores.parquet", index=False)
    record = {"manuscript_sha256": hashlib.sha256(manuscript.read_bytes()).hexdigest(),
              "cell_sha256": hashlib.sha256(original.encode()).hexdigest(),
              "role": "Aggregate exports of the executed manuscript cell; no vehicle-level records",
              "r_squared": {name: namespace[name] for name in
                  ("r2_primary", "r2_segment_top20_premium", "r2_segment_band3080_lower")}}
    (out / "export.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
