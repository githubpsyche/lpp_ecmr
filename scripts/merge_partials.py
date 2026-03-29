"""Merge per-subject partial fit JSONs into single result files.

Run this after all per-subject cluster fitting jobs have completed.
Groups partial files in fits/ by their prefix (everything before _sub{N}),
merges each group, and writes the combined result.

Usage:
    python scripts/merge_partials.py [--fits-dir fits] [--dry-run]
"""

import argparse
import glob
import json
import re
from pathlib import Path

from jaxcmr.helpers import find_project_root


def discover_groups(fits_dir: Path) -> dict[str, list[Path]]:
    partials = sorted(fits_dir.glob("*_sub*.json"))
    groups: dict[str, list[Path]] = {}
    for p in partials:
        prefix = re.sub(r"_sub\d+(?:_\d+)*\.json$", "", p.name)
        groups.setdefault(prefix, []).append(p)
    for paths in groups.values():
        paths.sort(
            key=lambda p: [int(s) for s in re.findall(r"sub(\d+)", p.name)] or [0]
        )
    return groups


def merge_group(paths: list[Path]) -> dict:
    partials = []
    for p in paths:
        with p.open() as f:
            partials.append(json.load(f))

    merged: dict = {
        "fixed": partials[0]["fixed"],
        "free": partials[0]["free"],
        "hyperparameters": partials[0]["hyperparameters"],
        "fitness": [],
        "fits": {k: [] for k in partials[0]["fits"]},
        "fit_time": sum(p["fit_time"] for p in partials),
    }
    for p in partials:
        merged["fitness"] += p["fitness"]
        for k in merged["fits"]:
            merged["fits"][k] += p["fits"][k]

    for key in partials[0]:
        if key not in merged:
            merged[key] = partials[0][key]

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fits-dir", default="fits", help="Directory containing partial JSONs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be merged without writing"
    )
    args = parser.parse_args()

    project_root = Path(find_project_root())
    fits_dir = project_root / args.fits_dir

    groups = discover_groups(fits_dir)
    if not groups:
        print(f"No partial files (*_sub*.json) found in {fits_dir}")
        return

    for prefix, paths in sorted(groups.items()):
        output = fits_dir / f"{prefix}.json"
        print(f"{prefix}: {len(paths)} partials -> {output.relative_to(project_root)}")
        if args.dry_run:
            continue
        merged = merge_group(paths)
        n_subjects = len(merged["fits"].get("subject", []))
        with output.open("w") as f:
            json.dump(merged, f, indent=4)
        print(f"  wrote {n_subjects} subjects, fit_time={merged['fit_time']:.0f}s")


if __name__ == "__main__":
    main()
