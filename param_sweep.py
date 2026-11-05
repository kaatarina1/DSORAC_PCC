"""Vary one 3DGS optimisation parameter at a time (others stay at default) and report for each value:
number of gaussians, compression ratio and mean PSNR / SSIM / LPIPS over the views in <eval_dir>/eval.json.

python param_sweep.py
python param_sweep.py --skip_generating --iterations 30000

Runs that already have results in <sweep_dir>/results.json are skipped, so the sweep can be resumed.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from pipeline import W, H, FOV, generateDataset
from render_eval import readGaussianPly, latestGaussianPly, compressionRatio, evaluate

# Defaults from gaussian-splatting/arguments/__init__.py (OptimizationParams)
DEFAULTS = {
    "scaling_lr": 0.005,
    "densification_interval": 100,
    "densify_from_iter": 500,
    "densify_grad_threshold": 0.0002,
}

SWEEP = {
    "scaling_lr":             [round(float(v), 4) for v in np.arange(0.005, 0.0009, -0.001)],
    "densification_interval": list(range(100, 501, 100)),
    "densify_from_iter":      list(range(500, 3001, 500)),
    "densify_grad_threshold": [round(float(v), 4) for v in np.arange(0.0002, 0.00081, 0.0001)],
}


def runName(param, value):
    return "default" if value == DEFAULTS[param] else f"{param}_{value}"


def train(dataset, model_path, extra, iterations):
    conda = os.environ.get("CONDA_EXE", "conda")
    cmd = [
        conda, "run", "-n", "gaussian_splatting", "--no-capture-output",
        "python", "./gaussian-splatting/train.py", "-s", dataset, "-m", model_path,
        "--iterations", str(iterations), "--save_iterations", str(iterations),
        "--disable_viewer", *extra,
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def printTable(results):
    print(f"\n{'parameter':<24}{'value':>10}{'gaussians':>12}{'ratio':>9}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}")
    for param, values in SWEEP.items():
        for value in values:
            r = results.get(runName(param, value))
            if r:
                print(f"{param:<24}{value:>10}{r['gaussians']:>12}{r['ratio']:>9.3f}"
                      f"{r['PSNR']:>8.2f}{r['SSIM']:>8.4f}{r['LPIPS']:>8.4f}")
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lasFile", default="data/pc/bezigrad_normals.las")
    p.add_argument("--eval_dir", help="folder with eval.json + ground truth images, default data/<name>_eval")
    p.add_argument("--iterations", type=int, default=30_000)
    p.add_argument("--skip_generating", action="store_true")
    args = p.parse_args()

    name = Path(args.lasFile).stem
    dataset = f"data/{name}"
    sweep_dir = Path("output", "sweep", name)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = Path(args.eval_dir or f"data/{name}_eval")

    if not args.skip_generating:
        halfV = np.deg2rad(FOV / 2)
        halfH = np.arctan(np.tan(halfV) * W / H)
        generateDataset(args.lasFile, dataset, halfH)

    results_path = sweep_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for param, values in SWEEP.items():
        for value in values:
            run = runName(param, value)
            if run in results:
                print(f"Skipping {run}, already done")
                continue
            model_path = str(sweep_dir / run)
            extra = [] if run == "default" else [f"--{param}", str(value)]
            train(dataset, model_path, extra, args.iterations)

            metrics = evaluate(model_path, args.lasFile, eval_dir)["mean"]
            results[run] = {
                "param": param, "value": value,
                "gaussians": len(readGaussianPly(latestGaussianPly(model_path))),
                "ratio": compressionRatio(args.lasFile, model_path),
                **{k: metrics[k] for k in ("PSNR", "SSIM", "LPIPS")},
            }
            results_path.write_text(json.dumps(results, indent=2))
            printTable(results)

    printTable(results)


if __name__ == "__main__":
    main()
