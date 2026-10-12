"""
Convert WebPlotDigitizer (WPD) exported data to meropenem_timekill_filled.csv format.

WPD exports a simple text file with tab-separated x,y values.
This script maps those to the full CSV schema.

Usage:
    python wpd_to_csv.py <wpd_export.txt> <regimen> <concentration>

Example:
    python wpd_to_csv.py fig2A_4xMIC.txt "40%" "4xMIC"
    python wpd_to_csv.py fig2A_16xMIC.txt "40%" "16xMIC"
"""
import sys
import os
import csv

def convert(wpd_file, regimen, concentration, strain="ATCC-27853"):
    """Read WPD export and return list of (regimen, concentration, time_h, value, strain) rows."""
    rows = []
    with open(wpd_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    time_h = round(float(parts[0]), 1)
                    value = round(float(parts[1]), 3)
                    rows.append((regimen, concentration, time_h, value, strain))
                except ValueError:
                    continue
    return rows


def merge_into_csv(new_rows, csv_path):
    """Merge new digitized rows into the filled CSV, replacing NAs where possible."""
    # Read existing CSV
    existing = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            existing.append(row)

    # Build lookup: (regimen, concentration, time_h) -> index
    row_map = {}
    for i, row in enumerate(existing):
        key = (row[0], row[1], row[2])
        row_map[key] = i

    updated = 0
    added = 0
    for reg, conc, t, val, strain in new_rows:
        key = (reg, conc, str(t))
        if key in row_map:
            idx = row_map[key]
            old_val = existing[idx][3]
            if old_val.strip().upper() == "NA":
                existing[idx][3] = str(val)
                updated += 1
                print(f"  Updated: {reg} {conc} t={t}: NA -> {val}")
            else:
                print(f"  Skipped: {reg} {conc} t={t}: already has value {old_val}")
        else:
            existing.append([reg, conc, str(t), str(val), strain])
            added += 1
            print(f"  Added: {reg} {conc} t={t} = {val}")

    # Sort by regimen, concentration, time
    existing.sort(key=lambda r: (
        {"40%": 0, "2x20%": 1, "3x13.3%": 2}.get(r[0], 9),
        {"4xMIC": 0, "16xMIC": 1}.get(r[1], 9),
        float(r[2])
    ))

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(existing)

    print(f"\n  {updated} values updated, {added} rows added")
    return updated + added


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    wpd_file = sys.argv[1]
    regimen = sys.argv[2]
    concentration = sys.argv[3]
    strain = sys.argv[4] if len(sys.argv) > 4 else "ATCC-27853"

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "meropenem_timekill_filled.csv")

    print(f"Reading WPD export: {wpd_file}")
    rows = convert(wpd_file, regimen, concentration, strain)
    print(f"Found {len(rows)} data points\n")

    if rows:
        print(f"Merging into: {csv_path}")
        merge_into_csv(rows, csv_path)
    else:
        print("No data points found!")
