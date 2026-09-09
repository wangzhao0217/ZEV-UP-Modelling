"""Check transferred reference tables, not independent national model reproduction."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "paper1_aggregate"


def test_aggregate_reference_fingerprints():
    manifest = json.loads((DATA / "aggregate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["raw_record_transfer"] is False
    assert manifest["scientific_table_values_changed"] is False
    for relative, expected in manifest["files"].items():
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(DATA.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative


def test_aggregate_reference_shapes_and_disclosure():
    inventory = json.loads((DATA / "table_inventory.json").read_text(encoding="utf-8"))
    assert len(inventory) == 9
    assert {entry["file"] for entry in inventory} == {path.name for path in DATA.glob("*.csv")}
    for entry in inventory:
        assert entry["precision"] == "as printed; rounded; not per-flow computational inputs"
        assert entry["caption"]
        with (DATA / entry["file"]).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        assert len(rows) == entry["rows"] + 1
        assert len(rows[0]) >= 3
        assert all(len(row) == len(rows[0]) for row in rows)
