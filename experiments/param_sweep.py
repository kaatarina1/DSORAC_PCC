"""Vary one 3DGS optimisation parameter at a time (others stay at default) and report for each value:
number of gaussians, compression ratio and mean PSNR / SSIM / LPIPS over the views in <eval_dir>/eval.json.

python -m experiments.param_sweep
python -m experiments.param_sweep --skip_generating --iterations 30000

Runs that already have results in <sweep_dir>/results.json are skipped, so the sweep can be resumed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gs_pipeline.config import GS_DIR
from gs_pipeline.dataset import read_point_cloud, generate_dataset, half_fov_horizontal
from gs_pipeline.evaluation import read_gaussian_ply, latest_gaussian_ply, compression_ratio, evaluate
from gs_pipeline.gs_runner import run_in_gs_env

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


def run_name(param, value):
    return "default" if value == DEFAULTS[param] else f"{param}_{value}"


def train(dataset, model_path, extra, iterations):
    args = ["-s", dataset, "-m", model_path,
            "--iterations", iterations, "--save_iterations", iterations,
            "--disable_viewer", *extra]
    print(" ".join(["train.py", *map(str, args)]))
    run_in_gs_env(GS_DIR / "train.py", *args)


def print_table(results):
    print(f"\n{'parameter':<24}{'value':>10}{'gaussians':>12}{'ratio':>9}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}")
    for param, values in SWEEP.items():
        for value in values:
            r = results.get(run_name(param, value))
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
        xyz, _, bb_min, bb_max, _ = read_point_cloud(args.lasFile)
        generate_dataset(args.lasFile, dataset, half_fov_horizontal(), xyz, bb_min, bb_max, 1)

    results_path = sweep_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for param, values in SWEEP.items():
        for value in values:
            run = run_name(param, value)
            if run in results:
                print(f"Skipping {run}, already done")
                continue
            model_path = str(sweep_dir / run)
            extra = [] if run == "default" else [f"--{param}", str(value)]
            train(dataset, model_path, extra, args.iterations)

            metrics = evaluate(model_path, args.lasFile, eval_dir)["mean"]
            results[run] = {
                "param": param, "value": value,
                "gaussians": len(read_gaussian_ply(latest_gaussian_ply(model_path))),
                "ratio": compression_ratio(args.lasFile, model_path),
                **{k: metrics[k] for k in ("PSNR", "SSIM", "LPIPS")},
            }
            results_path.write_text(json.dumps(results, indent=2))
            print_table(results)

    print_table(results)


if __name__ == "__main__":
    main()
