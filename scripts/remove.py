"""
thin_las.py – Remove points from LAS/LAZ files.

Usage examples:
  python thin_las.py input.las output.las --keep-percent 19
  python thin_las.py input.las output.las --keep-count 100000
  python thin_las.py input.las output.las --remove-count 200000

Requirements:
  pip install "laspy[lazrs]"
"""

import argparse
import sys
import numpy as np
import laspy


def main():
    parser = argparse.ArgumentParser(description="Thin (subsample) points in a LAS/LAZ file.")
    parser.add_argument("input",  help="Input LAS/LAZ file path")
    parser.add_argument("output", help="Output LAS/LAZ file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--keep-percent", type=float, metavar="PCT",
                       help="Percentage of points to KEEP (0-100)")
    group.add_argument("--keep-count", type=int, metavar="N",
                       help="Exact number of points to KEEP")
    group.add_argument("--remove-count", type=int, metavar="N",
                       help="Exact number of points to REMOVE")

    args = parser.parse_args()

    print(f"Reading: {args.input}")
    las = laspy.read(args.input)
    total = len(las.points)
    print(f"  Total points: {total:,}")

    # Resolve keep count
    if args.keep_percent is not None:
        if not (0 < args.keep_percent <= 100):
            sys.exit("Error: --keep-percent must be between 0 and 100.")
        keep = max(1, int(round(total * args.keep_percent / 100)))
    elif args.remove_count is not None:
        keep = max(0, total - args.remove_count)
    else:
        keep = args.keep_count

    if keep >= total:
        print("  Nothing to remove - writing file as-is.")
        las.write(args.output)
        return

    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(total, size=keep, replace=False))

    out_las = laspy.LasData(header=las.header)
    out_las.points = las.points[indices]

    print(f"  Keeping: {keep:,} points ({keep / total * 100:.1f}%)")
    print(f"  Size: {keep * 4 * 8}")
    print(f"  Removed: {total - keep:,} points")
    out_las.write(args.output)
    print(f"  Written: {args.output}")


if __name__ == "__main__":
    main()