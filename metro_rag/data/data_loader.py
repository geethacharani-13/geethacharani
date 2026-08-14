"""
Loader for the Delhi Metro dataset.

Source: Delhi Metro Dataset (Kaggle, arunjangir245/delhi-metro-dataset),
compiled from DMRC/Wikipedia station data.

Handles:
1. Loading the raw CSV.
2. Fixing 3 known corrupted coordinates (verified against real-world
   station locations -- see README for details).
3. Resolving interchange station names: the raw data tags interchange
   stations like "Rajouri Garden [Conn: Blue]" -- we strip this tag to a
   canonical name so the same physical station merges into one graph node
   across lines.
4. Grouping stations by line, sorted by real distance-from-start (this
   recovers the true physical station order along each line).
"""

import csv
import re
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "Delhi metro.csv")

_ANNOTATION_RE = re.compile(
    r"\s*\[Conn:[^\]]*\]|\s*\(First Station\)|\s*\(last station\)",
    re.IGNORECASE,
)

# Known data errors in the source CSV, corrected against real-world
# station locations.
_COORDINATE_FIXES = {
    "Shyam park": {"Longitude": 77.4235},
    "Lal Quila": {"Latitude": 28.6562, "Longitude": 77.2410},
    "Hindon River": {"Latitude": 28.6862, "Longitude": 77.4074},
}


def canonical_name(raw_name):
    return _ANNOTATION_RE.sub("", raw_name).strip()


def load_records(csv_path=CSV_PATH):
    """Returns list of dicts: name, canonical_name, line, distance_km, lat, lon."""
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Station Names"]
            lat = float(row["Latitude"])
            lon = float(row["Longitude"])

            if name in _COORDINATE_FIXES:
                fix = _COORDINATE_FIXES[name]
                lat = fix.get("Latitude", lat)
                lon = fix.get("Longitude", lon)

            records.append({
                "name": name,
                "canonical_name": canonical_name(name),
                "line": row["Metro Line"],
                "distance_km": float(row["Dist. From First Station(km)"]),
                "lat": lat,
                "lon": lon,
            })
    return records


def group_by_line(records):
    """Group records by line, sorted by distance-from-start (= physical order)."""
    lines = {}
    for r in records:
        lines.setdefault(r["line"], []).append(r)
    for line in lines:
        lines[line].sort(key=lambda r: r["distance_km"])
    return lines


def station_coordinates(records):
    """canonical_name -> (lat, lon), for use as an A* heuristic."""
    coords = {}
    for r in records:
        coords.setdefault(r["canonical_name"], (r["lat"], r["lon"]))
    return coords


if __name__ == "__main__":
    records = load_records()
    lines = group_by_line(records)
    print(f"Total rows: {len(records)}")
    print(f"Lines: {list(lines.keys())}")
    print(f"Unique canonical stations: {len(station_coordinates(records))}")
