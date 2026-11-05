import laspy
import numpy as np
import argparse
import os
import subprocess
from scipy.spatial import cKDTree
from pathlib import Path
import json
import shutil

def lasOrigin(path):
    # Scene origin as in js/LasLoader.js (all points, no outlier removal): x/z centre, lowest y
    las = laspy.read(path)
    x, y, z = np.asarray(las.x), np.asarray(las.z), -np.asarray(las.y)
    return np.array([(x.min() + x.max()) / 2, y.min(), (z.min() + z.max()) / 2])

def evaluate(model_path, las_file, eval_dir):
    # eval_dir holds eval.json and the GT images / masks it names, any mix of
    # PERSPECTIVE / ORTHOGRAPHIC / SPHERICAL cameras. Results go to <model_path>/eval/iteration_<n>.
    config_path = Path(eval_dir, "eval.json")
    config = json.loads(config_path.read_text())
    offset = np.zeros(3)
    if "lasOrigin" in config:
        offset = np.array(config["lasOrigin"]) - lasOrigin(las_file)
    out_dir = Path(model_path, "eval", latestGaussianPly(model_path).parent.name)
    conda = os.environ.get("CONDA_EXE", "conda")
    subprocess.run([
        conda, "run", "-n", "gaussian_splatting", "--no-capture-output",
        "python", "./gaussian_splatting_utils/render_views.py",
        "-m", model_path, "--config", str(config_path), "--out_dir", str(out_dir), "--offset", *map(str, offset)
    ], check=True)
    subprocess.run(["python", "./gaussian_splatting_utils/eval_views.py", str(out_dir), str(config_path)], check=True)
    return json.loads((out_dir / "metrics_views.json").read_text())

def latestGaussianPly(model_path):
    iterDir = max(Path(model_path, "point_cloud").glob("iteration_*"), key=lambda p: int(p.name.split("_")[1]))
    return iterDir / "point_cloud.ply"

def readGaussianPly(path):
    with open(path, "rb") as f:
        names = []
        while (line := f.readline().strip()) != b"end_header":
            t = line.split()
            if t[0] == b"property":
                names.append(t[2].decode())   # 3DGS writes only float properties
        gauss = np.fromfile(f, dtype=[(n, "<f4") for n in names])
        return gauss

def compressionRatio(las_file, model_path):
    # size per point = 8, size per gaussian = 58
    points = pointCount(las_file)
    gaussians = len(readGaussianPly(latestGaussianPly(model_path)))
    ratio = (points * 8) / (gaussians * 58)
    print(f"Compression ratio: {points} points * 8 / {gaussians} gaussians * 58 = {ratio:.3f}")
    return ratio

def gaussianCount(model_path):
    with open(latestGaussianPly(model_path), "rb") as f:
        for line in f:
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
    raise ValueError("no vertex count in ply header")

def pointCount(las_file):
    with laspy.open(las_file) as f:
        points = f.header.point_count
    return points