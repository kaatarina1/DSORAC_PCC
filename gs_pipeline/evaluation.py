import json
import subprocess
import sys
from pathlib import Path

import laspy
import numpy as np

from gs_pipeline.config import GS_UTILS
from gs_pipeline.gs_runner import run_in_gs_env


def gaussian_bytes(sh_degree):
    return 13 + 3 * ((sh_degree + 1) ** 2 - 1)


def las_origin(path):
    # Scene origin as in js/LasLoader.js (all points, no outlier removal): x/z centre, lowest y
    las = laspy.read(path)
    x, y, z = np.asarray(las.x), np.asarray(las.z), -np.asarray(las.y)
    return np.array([(x.min() + x.max()) / 2, y.min(), (z.min() + z.max()) / 2])


def evaluate(model_path, las_file, eval_dir, sh_degree=3):
    # eval_dir holds eval.json and the GT images / masks it names, any mix of
    # PERSPECTIVE / ORTHOGRAPHIC / SPHERICAL cameras. Results go to <model_path>/eval/iteration_<n>.
    config_path = Path(eval_dir, "eval.json")
    config = json.loads(config_path.read_text())
    offset = np.zeros(3)
    if "lasOrigin" in config:
        offset = np.array(config["lasOrigin"]) - las_origin(las_file)
    out_dir = Path(model_path, "eval", latest_gaussian_ply(model_path).parent.name)
    run_in_gs_env(GS_UTILS / "render_views.py",
                  "-m", model_path, "--config", config_path, "--out_dir", out_dir, "--offset", *offset,
                  "--sh_degree", sh_degree)
    # Runs in the pipeline environment, not via run_in_gs_env: eval_views.py needs lpips and matplotlib,
    # which were not verified to be installed in the gaussian_splatting env (see evaluate_gaussians in gs_runner.py).
    subprocess.run([sys.executable, str(GS_UTILS / "eval_views.py"), str(out_dir), str(config_path)], check=True)
    return json.loads((out_dir / "metrics_views.json").read_text())


def latest_gaussian_ply(model_path):
    iter_dir = max(Path(model_path, "point_cloud").glob("iteration_*"), key=lambda p: int(p.name.split("_")[1]))
    return iter_dir / "point_cloud.ply"


def read_gaussian_ply(path):
    with open(path, "rb") as f:
        names = []
        while (line := f.readline().strip()) != b"end_header":
            t = line.split()
            if t[0] == b"property":
                names.append(t[2].decode())   # 3DGS writes only float properties
        gauss = np.fromfile(f, dtype=[(n, "<f4") for n in names])
        return gauss


def compression_ratio(las_file, model_path, sh_degree=3):
    points = point_count(las_file)
    gaussians = len(read_gaussian_ply(latest_gaussian_ply(model_path)))
    byts = gaussian_bytes(sh_degree)
    ratio = (points * 8) / (gaussians * byts)
    print(f"Compression ratio: {points} points * 8 / {gaussians} gaussians * {byts} = {ratio:.3f}")
    return ratio


def gaussian_count(model_path):
    with open(latest_gaussian_ply(model_path), "rb") as f:
        for line in f:
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
    raise ValueError("no vertex count in ply header")


def point_count(las_file):
    with laspy.open(las_file) as f:
        points = f.header.point_count
    return points
