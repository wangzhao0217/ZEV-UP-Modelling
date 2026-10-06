"""Regenerate static scientific figures into results/figures without replacing references."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    """Run a selected figure producer with only the released local inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=("adoption", "charging", "biplot"))
    args = parser.parse_args()
    from reproduce import verify_files

    verify_files(ROOT)
    out = ROOT / "results/figures"
    out.mkdir(parents=True, exist_ok=True)
    if args.figure == "adoption":
        subprocess.run(["Rscript", str(ROOT / "scripts/figure_sources/adoption_map.R")], cwd=ROOT, check=True)
    elif args.figure == "charging":
        subprocess.run([sys.executable, str(ROOT / "scripts/figure_sources/charging_map.py"),
            "--area-root", str(ROOT / "data/charging"), "--area-manifest", str(ROOT / "data/maps/area_manifest.json"),
            "--basemap-dir", str(ROOT / "data/maps"), "--chargers", str(ROOT / "data/maps/charging_locations.geojson"),
            "--region-boundaries", str(ROOT / "data/maps/region_boundaries.geojson"),
            "--expected-oa-count", "46359", "--expected-charger-count", "1977", "--output-dir", str(out)], cwd=ROOT, check=True)
    else:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from functions.propensity_group_analysis import PropensityGroupAnalyzer

        frame = gpd.read_file(ROOT / "data/areas/SPT.gpkg")
        analyzer = PropensityGroupAnalyzer(tau_AP_min=.3, tau_AP_max=.8)
        categorized = analyzer.categorize_areas(frame)
        figure = analyzer.create_biplot(categorized, output_path=str(out / "demographic_biplot_SPT.png"), region_name="SPT")
        plt.close(figure)


if __name__ == "__main__":
    main()
