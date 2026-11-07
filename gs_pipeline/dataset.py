import json
import os
import shutil
import subprocess
from pathlib import Path

import laspy
import numpy as np

from gs_pipeline.config import FOV, W, H, REPO_ROOT


def run_capture(*args):
    # "deno task capture" is defined in deno.json, so it has to run from the repo root
    subprocess.run(["deno", "task", "capture", *map(str, args)], check=True, cwd=REPO_ROOT)


def radius_ratio(data_dir, ratios_for_new):
    ratios_path = Path(data_dir, "radius_ratios.json")
    ratios = json.loads(ratios_path.read_text()) if ratios_path.exists() else {}
    new = [p.name for p in sorted(Path(data_dir, "images").glob("*.png")) if p.name not in ratios]
    for name, ratio in zip(new, ratios_for_new(len(new))):
        ratios[name] = ratio
    ratios_path.write_text(json.dumps(ratios, indent=2))


def read_point_cloud(path):
    las = laspy.read(path)

    print(las.header.point_format.id, las.header.point_count)
    print(list(las.point_format.dimension_names))

    # Coordinates
    x, y, z = np.asarray(las.x), np.asarray(las.z), -np.asarray(las.y)
    # Colors
    rgb = np.vstack((las.red, las.green, las.blue)).T / 65535.0

    # Center on all points like js/LasLoader.js, so targets are in the frame capture.js renders in
    x = x - (x.min() + x.max()) / 2
    z = z - (z.min() + z.max()) / 2
    y = y - y.min()
    xyz = np.column_stack((x, y, z)).astype(np.float32)
    bb_min, bb_max = xyz.min(axis=0), xyz.max(axis=0)

    normals = None
    if {"nx", "ny", "nz"} <= set(las.point_format.dimension_names):
        n = np.column_stack((las.nx, las.nz, -las.ny)).astype(np.float32)
        normals = n / np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)

    return xyz, rgb, bb_min, bb_max, normals


def write_points3d(path, xyz, rgb, sampling_rate):
    idx = np.arange(0, len(xyz), sampling_rate).astype(int)
    colors = np.round(rgb[idx] * 255)
    rows = np.column_stack([np.arange(len(idx)), xyz[idx], colors, np.zeros(len(idx))])
    with open(path, "w") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        np.savetxt(f, rows, fmt="%d %.7g %.7g %.7g %d %d %d %d")


def floor_height(xyz, cx, cz, radius, q=2.0):
    d = np.hypot(xyz[:, 0] - cx, xyz[:, 2] - cz)
    ys = xyz[d < radius, 1]
    if len(ys) < 100:
        ys = xyz[:, 1]
    return float(np.percentile(ys, q))


def capture_views(las_file, out, poses, mode="BILLBOARDS"):
    # All poses in one deno run, so the renderer and LAS are loaded only once
    Path(out).mkdir(parents=True, exist_ok=True)
    poses_path = Path(out, "poses.json")
    poses_path.write_text(json.dumps([{"eye": list(map(float, eye)), "target": list(map(float, target)), "radiusRatio": float(ratio)}
                                      for eye, target, ratio in poses]))
    run_capture(
        Path(out).resolve(),
        "--type=lookAt",
        f"--lasFile={Path(las_file).resolve()}",
        f"--width={W}", f"--height={H}",
        f"--poses={poses_path.resolve()}",
        f"--mode={mode}",
        "--pointSize=0.2",
    )
    radius_ratio(out, lambda n: [p[2] for p in poses])


def voxel_centroids(xyz, voxel):
    keys = np.floor((xyz - xyz.min(0)) / voxel).astype(np.int64)
    k = keys - keys.min(0)
    dims = k.max(0) + 1
    flat = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    _, inv = np.unique(flat, return_inverse=True)
    inv = inv.ravel()                      # numpy 2.x may return it 2-D
    sums = np.stack([np.bincount(inv, weights=xyz[:, k]) for k in range(3)], axis=1)
    return sums / np.bincount(inv)[:, None]   # one representative point per voxel


def half_fov_horizontal():
    # Horizontal half field of view of the W x H capture with vertical field of view FOV
    return np.arctan(np.tan(np.deg2rad(FOV / 2)) * W / H)


def generate_dataset(path, out, alpha, positions, bb_min, bb_max, sampling_rate):
    # capture.js appends to an existing dataset, so start from an empty folder
    if Path(out).exists():
        shutil.rmtree(out, ignore_errors=True)
    out = Path(out).resolve()
    path = Path(path).resolve()

    V = voxel_centroids(positions, 0.01 * (bb_max - bb_min).max())

    mean = V.mean(axis=0)
    cov = np.cov((V - mean).T)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]

    s = np.sqrt(eigvals)
    planarity = (s[1] - s[2]) / s[0]
    sphericity = s[2] / s[0]

    centroid = V.mean(axis=0)
    cx, cz = centroid[0], centroid[2]
    target = [cx, centroid[1], cz]

    R = np.linalg.norm(V - target, axis=1).max()
    h = R / np.sin(alpha)
    span = min(bb_max[0] - bb_min[0], bb_max[2] - bb_min[2])

    print(planarity, sphericity)
    if planarity < sphericity:
        run_capture(
            out,
            f"--lasFile={path}",
            f"--width={W}",
            f"--height={H}",
            f"--targetX={target[0]}",
            f"--targetY={target[1]}",
            f"--targetZ={target[2]}",
            "--noCams=250",
            f"--radius={h}",
            "--fullSphera=1",
            "--radiusRatio=1",
            f"--samplingRate={sampling_rate}",
        )
        radius_ratio(out, lambda n: [1.0] * n)
    else:
        floor = floor_height(V, cx, cz, span / 4)
        floor += 0.02 * (bb_max[1] - bb_min[1])
        target = [cx, floor, cz]
        run_capture(
            out,
            f"--lasFile={path}",
            f"--width={W}",
            f"--height={H}",
            f"--targetX={target[0]}",
            f"--targetY={target[1]}",
            f"--targetZ={target[2]}",
            "--noCams=200",
            f"--radius={h}",
            "--radiusRatio=1",
            f"--samplingRate={sampling_rate}",
        )
        radius_ratio(out, lambda n: [1.0] * n)
        # Quadrants around the centroid, each target is the centroid of the points in that quadrant
        for i in [1, -1]:
            for j in [1, -1]:
                in_quad = (i * (V[:, 0] - cx) >= 0) & (j * (V[:, 2] - cz) >= 0)
                if not in_quad.any():
                    continue
                Q = V[in_quad]
                tx, tz = Q[:, [0, 2]].mean(axis=0)
                ty = floor_height(V, tx, tz, span / 8)
                ty += 0.02 * (bb_max[1] - bb_min[1])
                tq = np.array([tx, ty, tz])
                Rq = np.linalg.norm(Q - tq, axis=1).max()
                hq = max(Rq / np.sin(alpha), 0.25 * h)
                ratio = hq / h
                run_capture(
                    out,
                    f"--lasFile={path}",
                    f"--width={W}",
                    f"--height={H}",
                    f"--targetX={tx}",
                    f"--targetY={ty}",
                    f"--targetZ={tz}",
                    "--noCams=200",
                    f"--radius={hq}",
                    f"--radiusRatio={ratio}",
                    f"--samplingRate={sampling_rate}",
                )
                radius_ratio(out, lambda n: [ratio] * n)


def prepare_dataset(las_file, base_dir, work_dir, alpha, xyz, rgb, bb_min, bb_max, sampling_rate, regenerate=False):
    # Base dataset is generated once and never modified; each run works on a copy.
    # The "lasFile" key must stay as it is, otherwise existing markers no longer match and every dataset is regenerated.
    params = {"lasFile": str(Path(las_file).resolve()), "W": W, "H": H, "FOV": FOV}
    marker = Path(base_dir, "dataset_params.json")
    if regenerate or not marker.exists() or json.loads(marker.read_text()) != params:
        generate_dataset(las_file, base_dir, alpha, xyz, bb_min, bb_max, max(1, int(sampling_rate)))
        marker.write_text(json.dumps(params, indent=2))
    else:
        print(f"Reusing dataset {base_dir}")

    src, dst = Path(base_dir), Path(work_dir)
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    (dst / "images").mkdir(parents=True)
    shutil.copytree(src / "sparse", dst / "sparse")
    shutil.copy(src / "radius_ratios.json", dst / "radius_ratios.json")
    write_points3d(dst / "sparse/0/points3D.txt", xyz, rgb, sampling_rate)
    for img in (src / "images").iterdir():
        os.link(img, dst / "images" / img.name)
