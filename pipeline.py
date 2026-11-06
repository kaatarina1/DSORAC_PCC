import laspy
import numpy as np
import argparse
import os
import subprocess
from pathlib import Path
import json
import shutil
import time
from contextlib import contextmanager
from render_eval import compressionRatio, evaluate, gaussianCount, pointCount, gaussianBytes
import optimization_util as opt_util

FOV = 45
W = 1024
H = 512

class StageTimer:
    # Wall-clock time per pipeline stage, printed to the console and rewritten to a JSON file
    # after every stage, so the timings survive a crash or Ctrl+C.
    def __init__(self, path):
        self.path = Path(path)
        self.start = time.perf_counter()
        self.records = []

    @contextmanager
    def stage(self, name, **info):
        print(f"[time] {name} started", flush=True)
        t0 = time.perf_counter()
        status = "failed"
        try:
            yield
            status = "ok"
        finally:
            seconds = time.perf_counter() - t0
            total = time.perf_counter() - self.start
            self.records.append({"stage": name, **info, "seconds": round(seconds, 2), "status": status})
            print(f"[time] {name}: {seconds:.1f}s ({status}), total {total:.1f}s", flush=True)
            self.save()

    def save(self):
        totals = {}
        for r in self.records:
            totals[r["stage"]] = round(totals.get(r["stage"], 0) + r["seconds"], 2)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "total_seconds": round(time.perf_counter() - self.start, 2),
            "totals_per_stage": totals,
            "stages": self.records,
        }, indent=2))

    def print_summary(self):
        totals = json.loads(self.path.read_text())["totals_per_stage"] if self.path.exists() else {}
        total = time.perf_counter() - self.start
        print(f"\n{'stage':<20}{'seconds':>10}{'share':>8}")
        for name, seconds in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"{name:<20}{seconds:>10.1f}{100 * seconds / max(total, 1e-9):>7.1f}%")
        print(f"{'total':<20}{total:>10.1f}")

def latestMethodDir(model_path):
    return max(Path(model_path, "train").glob("ours_*"), key=lambda p: int(p.name.split("_")[1]))


def radiusRatio(data_dir, ratios_for_new): 
    ratios_path = Path(data_dir, "radius_ratios.json")
    ratios = json.loads(ratios_path.read_text()) if ratios_path.exists() else {}
    new = [p.name for p in sorted(Path(data_dir, "images").glob("*.png")) if p.name not in ratios]
    for name, ratio in zip(new, ratios_for_new(len(new))):
        ratios[name] = ratio
    ratios_path.write_text(json.dumps(ratios, indent=2))

def subsetDataset(data_dir, subset_dir, keep_names):
    src, dst = Path(data_dir), Path(subset_dir)
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    (dst / "images").mkdir(parents=True)
    (dst / "sparse/0").mkdir(parents=True)
    for f in ("cameras.txt", "points3D.txt"):
        shutil.copy(src / "sparse/0" / f, dst / "sparse/0" / f)

    entries = [line for line in (src / "sparse/0/images.txt").read_text().splitlines()
                if len(line.split()) == 10 and not line.startswith("#")
                and line.split()[9] in keep_names]
    (dst / "sparse/0/images.txt").write_text(
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "".join(e + "\n\n" for e in entries))
    for name in keep_names:
        os.link(src / "images" / name, dst / "images" / name)

def rankWorstViews(model_path):
    eval_dir =  latestMethodDir(model_path)
    scores = json.loads((eval_dir / "scores.json").read_text())
    names = sorted(scores)

    m = {k: np.array([scores[n][k] for n in names])
            for k in ("psnr", "ssim", "lpips")}
    m["psnr"] = 10 * np.log10(m["psnr"] + 1e-10)

    z = lambda x: (x - x.mean()) / (x.std() + 1e-8)
    score = z(m["psnr"]) + z(m["ssim"]) + z(m["lpips"])

    order = np.argsort(-score)
    return [int(names[i]) for i in order]

def readPointCloud(path):
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
    bbMin, bbMax = xyz.min(axis=0), xyz.max(axis=0)

    normals = None
    if {"nx", "ny", "nz"} <= set(las.point_format.dimension_names):
        n = np.column_stack((las.nx, las.nz, -las.ny)).astype(np.float32)
        normals = n / np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)

    return xyz, rgb, bbMin, bbMax, normals

def writePoints3D(path, xyz, rgb, sampling_rate):
    idx = np.arange(0, len(xyz), sampling_rate).astype(int)
    colors = np.round(rgb[idx] * 255)
    rows = np.column_stack([np.arange(len(idx)), xyz[idx], colors, np.zeros(len(idx))])
    with open(path, "w") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        np.savetxt(f, rows, fmt="%d %.7g %.7g %.7g %d %d %d %d")

def floorHeight(xyz, cx, cz, radius, q=2.0):
    d = np.hypot(xyz[:, 0] - cx, xyz[:, 2] - cz)
    ys = xyz[d < radius, 1]
    if len(ys) < 100:
        ys = xyz[:, 1]
    return float(np.percentile(ys, q))

# TODO: Make it from new locations
def cameraPoses(model_path, worst, rest, scene_size, ratios, seed= 0, step_range=(0.03, 0.08), max_angle=10, floor_y=None):
    cams = json.loads(Path(model_path, "cameras.json").read_text())
    if max(worst) >= len(cams):
        raise RuntimeError(f"{model_path}/cameras.json has {len(cams)} cameras but renders go up to index {max(worst)}: "
                           "cameras.json is not from the run that produced the renders, re-run training for this model")
    poses = []
    for i in worst:
        c = cams[i]
        assert c["id"] == i
        eye = np.array(c["position"])
        forward = np.array(c["rotation"])[:, 2]
        poses.append((eye, eye + forward, ratios[c["img_name"]]))

    rng = np.random.default_rng(seed)

    for i in rest:
        c = cams[i]
        
        eye = np.array(c["position"])
        forward = np.array(c["rotation"])[:, 2]

        offset = rng.normal(size=3)
        offset /= np.linalg.norm(offset)

        eye_new = eye + offset * rng.uniform(*step_range) * scene_size

        if floor_y is not None:
            eye_new[1] = max(eye_new[1], floor_y)

        forward_new = randomRotate(forward, max_angle, rng)
        poses.append((eye_new, eye_new + forward_new, ratios[c["img_name"]]))

    return poses

def captureViews(lasFile, out, poses, mode = "BILLBOARDS"):
    # All poses in one deno run, so the renderer and LAS are loaded only once
    Path(out).mkdir(parents=True, exist_ok=True)
    posesPath = Path(out, "poses.json")
    posesPath.write_text(json.dumps([{"eye": list(map(float, eye)), "target": list(map(float, target)), "radiusRatio": float(ratio)}
                                 for eye, target, ratio in poses]))
    cmd = [
        "deno", "task", "capture", out,
        f"--type=lookAt",
        f"--lasFile={lasFile}",
        f"--width={W}", f"--height={H}",
        f"--poses={posesPath}",
        f"--mode={mode}",
        f"--pointSize=0.2",
    ]
    subprocess.run(cmd, check=True)
    radiusRatio(out, lambda n: [p[2] for p in poses])

def voxelCentroids(xyz, voxel):
    keys = np.floor((xyz - xyz.min(0)) / voxel).astype(np.int64)
    k = keys - keys.min(0)
    dims = k.max(0) + 1
    flat = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    _, inv = np.unique(flat, return_inverse=True)
    inv = inv.ravel()                      # numpy 2.x may return it 2-D
    sums = np.stack([np.bincount(inv, weights=xyz[:, k]) for k in range(3)], axis=1)
    return sums / np.bincount(inv)[:, None]   # one representative point per voxel

def generateDataset(path, out, alpha, positions, colors, bbMin, bbMax, samplingRate):
    # capture.js appends to an existing dataset, so start from an empty folder
    if Path(out).exists():
        shutil.rmtree(out, ignore_errors=True)

    V = voxelCentroids(positions,  0.01 * (bbMax - bbMin).max())

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
    span = min(bbMax[0] - bbMin[0], bbMax[2] - bbMin[2])

    print(planarity, sphericity)
    if (planarity < sphericity):
        cmd = [
            "deno", "task", "capture", out,
            f"--lasFile={path}",
            f"--width={W}",
            f"--height={H}",
            f"--targetX={target[0]}",
            f"--targetY={target[1]}",
            f"--targetZ={target[2]}",
            f"--noCams=250",
            f"--radius={h}",
            f"--fullSphera=1",
            f"--radiusRatio=1",
            f"--samplingRate={samplingRate}"
        ]
        subprocess.run(cmd, check=True)
        radiusRatio(out, lambda n: [1.0] * n)
    else:
        floor = floorHeight(V, cx, cz, span / 4) 
        floor +=  0.02 * (bbMax[1] - bbMin[1])
        target = [cx, floor, cz]
        cmd = [
            "deno", "task", "capture", out,
            f"--lasFile={path}",
            f"--width={W}",
            f"--height={H}",
            f"--targetX={target[0]}",
            f"--targetY={target[1]}",
            f"--targetZ={target[2]}",
            f"--noCams=200",
            f"--radius={h}",
            f"--radiusRatio=1",
            f"--samplingRate={samplingRate}"
        ]
        subprocess.run(cmd, check=True)
        radiusRatio(out, lambda n: [1.0] * n)
        # Quadrants around the centroid, each target is the centroid of the points in that quadrant
        for i in [1, -1]:
            for j in [1, -1]:
                inQuad = (i * (V[:, 0] - cx) >= 0) & (j * (V[:, 2] - cz) >= 0)
                if not inQuad.any():
                    continue
                Q = V[inQuad]
                tx, tz = Q[:, [0, 2]].mean(axis=0)
                ty = floorHeight(V, tx, tz, span / 8)
                ty += 0.02 * (bbMax[1] - bbMin[1])
                tq = np.array([tx, ty, tz])
                Rq = np.linalg.norm(Q - tq, axis=1).max()
                hq = max(Rq / np.sin(alpha), 0.25 * h)
                ratio = hq / h
                cmd = [
                    "deno", "task", "capture", out,
                    f"--lasFile={path}",
                    f"--width={W}",
                    f"--height={H}",
                    f"--targetX={tx}",
                    f"--targetY={ty}",
                    f"--targetZ={tz}",
                    f"--noCams=200",
                    f"--radius={hq}",
                    f"--radiusRatio={ratio}",
                    f"--samplingRate={samplingRate}"
                ]
                subprocess.run(cmd, check=True)
                radiusRatio(out, lambda n: [ratio] * n)

def prepareDataset(lasFile, base_dir, work_dir, halfV, xyz, rgb, bbMin, bbMax, sampling_rate, regenerate=False):
    # Base dataset is generated once and never modified; each run works on a copy
    params = {"lasFile": str(Path(lasFile).resolve()), "W": W, "H": H, "FOV": FOV}
    marker = Path(base_dir, "dataset_params.json")
    if regenerate or not marker.exists() or json.loads(marker.read_text()) != params:
        generateDataset(lasFile, base_dir, halfV, xyz, rgb, bbMin, bbMax, max(1, int(sampling_rate)))
        marker.write_text(json.dumps(params, indent=2))
    else:
        print(f"Reusing dataset {base_dir}")

    src, dst = Path(base_dir), Path(work_dir)
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    (dst / "images").mkdir(parents=True)
    shutil.copytree(src / "sparse", dst / "sparse")
    shutil.copy(src / "radius_ratios.json", dst / "radius_ratios.json")
    writePoints3D(dst / "sparse/0/points3D.txt", xyz, rgb, sampling_rate)
    for img in (src / "images").iterdir():
        os.link(img, dst / "images" / img.name)

def cleanIteration(model_path, iteration):
    Path(model_path, f"chkpnt{iteration}.pth").unlink(missing_ok=True)
    shutil.rmtree(Path(model_path, "point_cloud", f"iteration_{iteration}"), ignore_errors=True)

def runGaussianSplattin(model_path, out, checkpoint = -1, checkpoint_model = "", iterations = [], sh_degree=3):
    # Na Windows je "conda" v PATH le conda.bat, ki ga subprocess brez shell-a ne najde,
    # zato uporabimo pravi conda.exe iz CONDA_EXE (nastavljen v aktiviranem okolju).
    conda = os.environ.get("CONDA_EXE", "conda")
    cmd = [
        conda, "run", "-n", "gaussian_splatting", "--no-capture-output",
        "python", "./gaussian-splatting/train.py", "-s", out, "-m", model_path,
        "--random_background", "--disable_viewer", "--sh_degree", str(sh_degree),
        "--test_iterations", "-1", "--position_lr_max_steps", "35000",
        "--densify_grad_threshold", "0.001",
    ]
    if (checkpoint != -1):
        cmd.append("--checkpoint_iterations")
        cmd.append(str(checkpoint))
    if (checkpoint_model != ""):
        cmd.append("--start_checkpoint")
        cmd.append(checkpoint_model)
    if (len(iterations) != 0):
        cmd.append("--iterations")
        cmd.append(str(max(iterations)))
        cmd.append("--save_iterations")
        cmd.extend(map(str, iterations))
    subprocess.run(cmd, check=True)

def renderGaussianSplattin(model_path, iteration, out):
    conda = os.environ.get("CONDA_EXE", "conda")
    cmd = [
        conda, "run", "-n", "gaussian_splatting", "--no-capture-output",
        "python", "./gaussian-splatting/render.py", "-s", out, "-m", model_path, 
        "--skip_test", "--iteration", str(iteration)
    ]
    subprocess.run(cmd, check=True)

def evaluateGaussianSplatting(method_dir, metric, save_maps=False):
    cmd = [
        "python", "./gaussian_splatting_utils/per_pixel_metrics.py", 
        str(method_dir), metric, "--no_png"
    ]
    if not save_maps:
        cmd.append("--no_maps")
    subprocess.run(cmd, check=True)

def pruneGaussians(checkpoint, out, size_factor=(0.05, 2), min_opacity=0.2, n_target=-1, seed=0):
    checkpoint_path = f"{out}/chkpnt{checkpoint}.pth"
    ply_path = f"{out}/point_cloud/iteration_{checkpoint}/point_cloud.ply"
    conda = os.environ.get("CONDA_EXE", "conda")
    cmd = [
        conda, "run", "-n", "gaussian_splatting", "--no-capture-output",
        "python", "prune_gaussians.py",
        "--checkpoint", checkpoint_path,
        "--out_checkpoint", checkpoint_path,
        "--ply", ply_path,
        "--size_factor_min", str(size_factor[0]),
        "--size_factor_max", str(size_factor[1]),
        "--min_opacity", str(min_opacity),
        "--target_count", str(n_target),
        "--seed", str(seed)
    ]
    subprocess.run(cmd, check=True)

def squeezGaussians(checkpoint, out):
    checkpoint_path = f"{out}/chkpnt{checkpoint}.pth"
    ply_path = f"{out}/point_cloud/iteration_{checkpoint}/point_cloud.ply"
    conda = os.environ.get("CONDA_EXE", "conda")
    cmd = [
        conda, "run", "-n", "gaussian_splatting", "--no-capture-output",
        "python", "squeeze_gaussians.py",
        "--checkpoint", checkpoint_path,
        "--out_checkpoint", checkpoint_path,
        "--points", f"{out}/points.npy",
        "--ply", ply_path,
    ]
    subprocess.run(cmd, check=True)

def addGaussians(checkpoint, out, indices_path):
    checkpoint_path = f"{out}/chkpnt{checkpoint}.pth"
    ply_path = f"{out}/point_cloud/iteration_{checkpoint}/point_cloud.ply"
    conda = os.environ.get("CONDA_EXE", "conda")
    cmd = [
        conda, "run", "-n", "gaussian_splatting", "--no-capture-output",
        "python", "add_gaussians.py",
        "--checkpoint", checkpoint_path,
        "--out_checkpoint", checkpoint_path,
        "--ply", ply_path,
        "--points", f"{out}/points.npy",
        "--colors", f"{out}/colors.npy",
        "--indices", str(indices_path),
    ]
    subprocess.run(cmd, check=True)


def randomRotate(pos, max_angle_deg, rng):
    axis = np.cross(pos, rng.normal(size=3))
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(0, max_angle_deg))
    return pos * np.cos(angle) + np.cross(axis, pos) * np.sin(angle)

def calculateQuality(scores, metric, names):
    v = np.array([scores[n][metric] for n in names])
    if metric == "psnr":
        per_view = -10 * np.log10(v + 1e-10)
    elif metric == "ssim":
        per_view = 1 - v
    else: per_view = -v
    return float(per_view.mean()), float(np.percentile(per_view, 10))

def measure_views(model_path, checkpoint, data_dir, save_maps=False):
    # Renders and computes the psnr error in one GPU process, without writing renders / gt PNGs
    conda = os.environ.get("CONDA_EXE", "conda")
    cmd = [
        conda, "run", "-n", "gaussian_splatting", "--no-capture-output",
        "python", "./gaussian_splatting_utils/measure_views.py", "-s", data_dir, "-m", model_path,
        "--iteration", str(checkpoint),
    ]
    if not save_maps:
        cmd.append("--no_maps")
    subprocess.run(cmd, check=True)

def measure(model_path, checkpoint, data_dir, metric, names=None, save_maps=False):
    method_dir = Path(model_path, "train", f"ours_{checkpoint}")
    if metric == "psnr":
        measure_views(model_path, checkpoint, data_dir, save_maps=save_maps)
    else:   # TODO: ssim / lpips are not ported to the GPU path yet
        renderGaussianSplattin(model_path, checkpoint, data_dir)
        evaluateGaussianSplatting(method_dir, metric, save_maps=save_maps)
    scores = json.loads((method_dir / "scores.json").read_text())
    names = names if names is not None else sorted(scores)
    q, p10 = calculateQuality(scores, metric, names)
    return q, p10, names, method_dir

def stalled(history):
    phase = history[-1]["phase"]
    steps = [(a, b) for a, b in zip(history, history[1:]) if a["phase"] == phase]
    if len(steps) < 3:
        return False
    recent = steps[-3:]
    if phase == "compress":
        return all((b["r"] - a["r"]) / a["r"] <  0.02 for a, b in recent)
    return all((b["q"] - a["q"]) < 0.1 for a, b in recent)

def show_quality(q, metric):
    return -q if metric == "lpips" else q

def print_round(h, metric):
    print(f"[loop {h['loop']}] {h['phase']:<8} N={h['N']}  r={h['r']:.3f}  "
          f"{metric}={show_quality(h['q'], metric):.4f}  p10={show_quality(h['p10'], metric):.4f}", flush=True)

def print_summary(history, reason, r_star, r_low, metric, path):
    print(f"\n{'krog':>4} {'faza':<9}{'N':>10}{'r':>8}{metric:>9}{'p10':>9}")
    for h in history:
        print(f"{h['loop']:>4} {h['phase']:<9}{h['N']:>10}{h['r']:>8.3f}"
              f"{show_quality(h['q'], metric):>9.4f}{show_quality(h['p10'], metric):>9.4f}")
    last = history[-1]
    ok = lambda b: "OK" if b else "NI DOSEŽENO"
    print(f"\nKonec: {reason}")
    print(f"  r = {last['r']:.3f}   cilj {r_star:.3f}, meja {r_low:.3f}   {ok(last['r'] >= r_low)}")
    path.write_text(json.dumps({"reason": reason, "r_star": r_star, "r_low": r_low,
                                "metric": metric, "history": history}, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lasFile")
    parser.add_argument("--work_root", default=".", help="Root for generated data (<root>/<name>) and models (<root>/output/<name>).")
    parser.add_argument("--regenerate", action="store_true", help="Rebuild the base dataset even if it exists.", default=False)
    parser.add_argument("--eval_dir", help="folder with eval.json + GT images/masks, default data/<name>_eval")
    parser.add_argument("--error_metric", choices=["psnr", "ssim", "lpips"], default="psnr")
    parser.add_argument("--max_loops", type=int, default=10, help="Upper limit for loop rounds.")
    parser.add_argument("--compression_ratio", type=float, default=1, help="Target compression ratio r*.")
    parser.add_argument("--ratio_tol", type=float, default=0.01, help="r may be this much below r*.")
    parser.add_argument("--sh_degree", type=int, choices=[0, 1, 2, 3], default=3,
                    help="Max SH degree of the Gaussians.")

    args = parser.parse_args()
    halfV = np.deg2rad(FOV / 2)
    halfH = np.arctan(np.tan(halfV) * W / H)

    gs_bytes = gaussianBytes(args.sh_degree)

    name = Path(args.lasFile).stem
    base_dir = str(Path(args.work_root, name))
    data_dir = f"{base_dir}_work"
    model_path = str(Path(args.work_root, "output", f"{name}_sh{args.sh_degree}_cr{int(args.compression_ratio)}"))
    timer = StageTimer(Path(model_path, "timings.json"))

    with timer.stage("read_point_cloud"):
        xyz, rgb, bbMin, bbMax, normals = readPointCloud(args.lasFile)

    sampling_rate = gs_bytes / 8 * args.compression_ratio

    with timer.stage("prepare_dataset", regenerate=args.regenerate):
        prepareDataset(args.lasFile, base_dir, data_dir, halfH, xyz, rgb, bbMin, bbMax, sampling_rate, args.regenerate)

    checkpoint = 17_000
    with timer.stage("train_initial", iterations=checkpoint):
        runGaussianSplattin(model_path, data_dir, checkpoint, iterations=[checkpoint], sh_degree=args.sh_degree)

    np.save(Path(model_path, "points.npy"), xyz)
    np.save(Path(model_path, "colors.npy"), rgb.astype(np.float32))
    if normals is not None:
        np.save(Path(model_path, "normals.npy"), normals)

    # evaluate against ground truth
    eval_dir = Path(args.eval_dir or Path(args.work_root, f"{name}_eval"))
    if (eval_dir / "eval.json").exists():
        with timer.stage("evaluate_gt", checkpoint=checkpoint):
            evaluate(model_path, args.lasFile, eval_dir, args.sh_degree)

    n_base = len(list(Path(data_dir, "images").glob("*.png")))   # before any captureViews
    base_names = [f"{i:05d}" for i in range(n_base)]

    r_low = args.compression_ratio * (1 - args.ratio_tol)
    history = []
    reason = "Max iterations reached"
    size_factor = (0.05, 3)
    min_opacity = 0.2
    add_frac = 0.1
    new_factor = 0.1
    n_views = 20
    summary_path = Path(model_path, "loop_summary.json")
    n_points = pointCount(args.lasFile)
    n_target = int(n_points * 8 / (gs_bytes * args.compression_ratio))
    for loop in range(args.max_loops):
        loop_start = time.perf_counter()
        r = compressionRatio(args.lasFile, model_path, args.sh_degree)
        with timer.stage("measure", loop=loop, checkpoint=checkpoint):
            q, p10, _, method_dir = measure(model_path, checkpoint, data_dir, args.error_metric, names=base_names, save_maps=True)
        N = gaussianCount(model_path)

        r_ok = r >= r_low
        phase = "improve" if r_ok else "compress"
        history.append({"loop": loop, "checkpoint": checkpoint, "phase": phase,
                    "N": N, "r": r, "q": q, "p10": p10, "min_opacity": min_opacity})
        print_round(history[-1], args.error_metric)   

        with timer.stage("prune", loop=loop, phase=phase):
            if phase == "compress":
                if loop >= 2:
                    pruneGaussians(checkpoint, model_path, size_factor, min_opacity, n_target, loop)
                else: 
                    pruneGaussians(checkpoint, model_path, size_factor, min_opacity)
                size_factor = (size_factor[0] * 1.2 if size_factor[0] < 0.1 else 0.1, size_factor[1])
                min_opacity = min_opacity * 1.3 if min_opacity < 0.49 else 0.5
                count = gaussianCount(model_path)
                n_removed = N - count
                add_frac = new_factor * n_removed / count
                new_factor = new_factor - 0.03 if new_factor > 0 else 0 
                n_views = 10
            else:
                pruneGaussians(checkpoint, model_path, (0.02, 4), 0.15)
                count = gaussianCount(model_path)
                n_removed = N - count
                add_frac = n_removed / count
                n_views = 30

        history[-1].update(n_after_prune=count, budget_frac=float(add_frac), n_views=n_views)

        ply_path = Path(model_path, "point_cloud", f"iteration_{checkpoint}", "point_cloud.ply")
        with timer.stage("optimize", loop=loop):
            new_poses, picked = opt_util.optimize(model_path, method_dir, ply_path,
                                                  args.error_metric, budget_frac=add_frac, n_views=n_views,
                                                  stride=2, seed=loop, radius_ratio=0.05)

        shutil.rmtree(method_dir / args.error_metric, ignore_errors=True)
        shutil.copy(method_dir / "scores.json", Path(model_path, f"scores_loop{loop}.json"))
        shutil.rmtree(method_dir, ignore_errors=True)

        indices_path = Path(model_path, f"add_idx_loop{loop}.npy")
        np.save(indices_path, picked)
        with timer.stage("add_gaussians", loop=loop, n_added=int(len(picked))):
            addGaussians(checkpoint, model_path, indices_path)

        if new_poses:
            with timer.stage("capture_views", loop=loop, n_views=len(new_poses)):
                captureViews(args.lasFile, data_dir, new_poses, "POINTS")

        with timer.stage("train", loop=loop, start=checkpoint, end=checkpoint + 3_000):
            runGaussianSplattin(model_path, data_dir, checkpoint + 3_000,
                                f"{model_path}/chkpnt{checkpoint}.pth", [checkpoint + 3_000], sh_degree=args.sh_degree)
        cleanIteration(model_path, checkpoint)
        checkpoint += 3_000

        if (eval_dir / "eval.json").exists():
            with timer.stage("evaluate_gt", loop=loop, checkpoint=checkpoint):
                evaluate(model_path, args.lasFile, eval_dir, args.sh_degree)

        history[-1]["seconds"] = round(time.perf_counter() - loop_start, 2)
        print(f"[time] loop {loop}: {history[-1]['seconds']:.1f}s", flush=True)
        summary_path.write_text(json.dumps({"history": history}, indent=2))
        
        if stalled(history):
            reason = f"No progression - quality converged {phase}" 
            break
        if loop == args.max_loops - 1:
            reason = f"Max iteration reached - achieved compression {r}" 
            break   

    Path(model_path, f"chkpnt{checkpoint}.pth").unlink(missing_ok=True)
    for f in ["points.npy", "colors.npy", "normals.npy", *Path(model_path).glob("add_idx_loop*.npy"),
                *Path(model_path).glob("events.out.tfevents.*")]:
        Path(model_path, f).unlink(missing_ok=True)
    print_summary(history, reason, args.compression_ratio, r_low, args.error_metric, summary_path)
    timer.save()
    timer.print_summary()

if __name__ == "__main__":
    main()