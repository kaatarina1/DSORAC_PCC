import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from pipeline import W, H

# Re-renders a few fixed cameras of an existing dataset, to test renderer changes without regenerating it.
# The cameras are picked once from data/<scene> and saved to test_views/<scene>.json (edit it to change them).
#   python test_views.py data/pc/ljubljana_1_normals.las [--count 6] [--ratio 0.25] [--label name]


def read_images_txt(path):
    # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (every second line holds the 2D points)
    poses = {}
    for line in Path(path).read_text().splitlines():
        e = line.split()
        if len(e) != 10 or e[0].startswith("#"):
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, e[1:8])
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        poses[e[9]] = (R, np.array([tx, ty, tz]))
    return poses


def pick_views(dataset, count, ratio):
    # The eye is in the file name (index_x_y_z_mode.png), the view direction comes from images.txt.
    # Near/far in the renderer depend only on the direction, so eye + forward works as the target.
    poses = read_images_txt(Path(dataset, "sparse/0/images.txt"))
    views = []
    for name in sorted(poses):
        R, t = poses[name]
        eye = np.array(list(map(float, name.split("_")[1:4])))
        assert np.allclose(-R.T @ t, eye, atol=1e-2), f"{name}: images.txt does not match the file name"
        views.append({"name": name, "eye": eye, "forward": R[2]})

    # generateDataset puts overview cameras (radiusRatio 1) at distance h and quadrant cameras
    # (radiusRatio 0.25) at h / 4, the image index is not reliable because noCams changed between datasets
    dist = np.array([np.linalg.norm(v["eye"]) for v in views])
    for v, d in zip(views, dist):
        v["radiusRatio"] = 1.0 if d > dist.max() / 2 else 0.25

    candidates = [v for v in views if v["radiusRatio"] == ratio]
    if not candidates:
        raise RuntimeError(f"No images with radiusRatio {ratio} in {dataset}")
    picked = [candidates[i] for i in np.linspace(0, len(candidates) - 1, count).round().astype(int)]
    return [{"name": v["name"], "eye": v["eye"].tolist(), "target": (v["eye"] + v["forward"]).tolist(),
             "radiusRatio": v["radiusRatio"]} for v in picked]


def render_views(las_file, views, run_dir, dataset):
    # capture.js takes one radiusRatio per run, so render each ratio separately
    for ratio in sorted({v["radiusRatio"] for v in views}):
        group = [(i, v) for i, v in enumerate(views) if v["radiusRatio"] == ratio]
        tmp = Path(run_dir, f"_capture_{ratio}")
        shutil.rmtree(tmp, ignore_errors=True)   # capture.js appends to an existing images.txt
        tmp.mkdir(parents=True)
        poses_path = Path(tmp, "poses.json")
        poses_path.write_text(json.dumps([{"eye": v["eye"], "target": v["target"]} for _, v in group]))
        subprocess.run([
            "deno", "task", "capture", str(tmp),
            "--type=lookAt",
            f"--lasFile={las_file}",
            f"--width={W}", f"--height={H}",
            f"--poses={poses_path}",
            f"--radiusRatio={ratio}",
        ], check=True)

        # capture.js names images in render order, so they map back to the group by index
        rendered = sorted(Path(tmp, "images").glob("*.png"))
        for (i, v), image in zip(group, rendered):
            shutil.move(image, Path(run_dir, f"{i:02d}_new.png"))
            shutil.copy(Path(dataset, "images", v["name"]), Path(run_dir, f"{i:02d}_old.png"))
        shutil.rmtree(tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("las_file")
    parser.add_argument("--count", type=int, default=6, help="How many views to pick (only when picking)")
    parser.add_argument("--ratio", type=float, default=0.25, help="radiusRatio of the picked views")
    parser.add_argument("--label", default="run")
    args = parser.parse_args()

    scene = Path(args.las_file).stem
    dataset = Path("data", scene)
    views_path = Path("test_views", f"{scene}.json")
    if not views_path.exists():
        views_path.parent.mkdir(exist_ok=True)
        views_path.write_text(json.dumps(pick_views(dataset, args.count, args.ratio), indent=2))
        print(f"Picked views saved to {views_path}")
    views = json.loads(views_path.read_text())

    run_dir = Path("output", "test_views", scene, args.label)
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    render_views(args.las_file, views, run_dir, dataset)
    print(f"{len(views)} views in {run_dir} (NN_new.png = now, NN_old.png = dataset)")


if __name__ == "__main__":
    main()
