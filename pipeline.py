import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from gs_pipeline import optimization as opt_util
from gs_pipeline.dataset import read_point_cloud, prepare_dataset, capture_views, half_fov_horizontal
from gs_pipeline.evaluation import compression_ratio, evaluate, gaussian_count, point_count, gaussian_bytes
from gs_pipeline.gs_runner import train_gaussians, prune_gaussians, add_gaussians, clean_iteration
from gs_pipeline.quality import measure, stalled, print_round, print_summary
from gs_pipeline.timing import StageTimer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("las_file")
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
    half_h = half_fov_horizontal()

    gs_bytes = gaussian_bytes(args.sh_degree)

    name = Path(args.las_file).stem
    base_dir = str(Path(args.work_root, name))
    data_dir = f"{base_dir}_work"
    model_path = str(Path(args.work_root, "output", f"{name}_sh{args.sh_degree}_cr{int(args.compression_ratio)}"))
    timer = StageTimer(Path(model_path, "timings.json"))

    with timer.stage("read_point_cloud"):
        xyz, rgb, bb_min, bb_max, normals = read_point_cloud(args.las_file)

    sampling_rate = gs_bytes / 8 * args.compression_ratio

    with timer.stage("prepare_dataset", regenerate=args.regenerate):
        prepare_dataset(args.las_file, base_dir, data_dir, half_h, xyz, rgb, bb_min, bb_max, sampling_rate, args.regenerate)

    checkpoint = 17_000
    with timer.stage("train_initial", iterations=checkpoint):
        train_gaussians(model_path, data_dir, checkpoint, iterations=[checkpoint], sh_degree=args.sh_degree)

    np.save(Path(model_path, "points.npy"), xyz)
    np.save(Path(model_path, "colors.npy"), rgb.astype(np.float32))
    if normals is not None:
        np.save(Path(model_path, "normals.npy"), normals)

    # evaluate against ground truth
    eval_dir = Path(args.eval_dir or Path(args.work_root, f"{name}_eval"))
    if (eval_dir / "eval.json").exists():
        with timer.stage("evaluate_gt", checkpoint=checkpoint):
            evaluate(model_path, args.las_file, eval_dir, args.sh_degree)

    n_base = len(list(Path(data_dir, "images").glob("*.png")))   # before any capture_views
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
    n_points = point_count(args.las_file)
    n_target = int(n_points * 8 / (gs_bytes * args.compression_ratio))
    for loop in range(args.max_loops):
        loop_start = time.perf_counter()
        r = compression_ratio(args.las_file, model_path, args.sh_degree)
        with timer.stage("measure", loop=loop, checkpoint=checkpoint):
            q, p10, _, method_dir = measure(model_path, checkpoint, data_dir, args.error_metric, names=base_names, save_maps=True)
        N = gaussian_count(model_path)

        r_ok = r >= r_low
        phase = "improve" if r_ok else "compress"
        history.append({"loop": loop, "checkpoint": checkpoint, "phase": phase,
                    "N": N, "r": r, "q": q, "p10": p10, "min_opacity": min_opacity})
        print_round(history[-1], args.error_metric)

        with timer.stage("prune", loop=loop, phase=phase):
            if phase == "compress":
                if loop >= 2:
                    prune_gaussians(checkpoint, model_path, size_factor, min_opacity, n_target, loop)
                else:
                    prune_gaussians(checkpoint, model_path, size_factor, min_opacity)
                size_factor = (size_factor[0] * 1.2 if size_factor[0] < 0.1 else 0.1, size_factor[1])
                min_opacity = min_opacity * 1.3 if min_opacity < 0.49 else 0.5
                count = gaussian_count(model_path)
                n_removed = N - count
                add_frac = new_factor * n_removed / count
                new_factor = new_factor - 0.03 if new_factor > 0 else 0
                n_views = 10
            else:
                prune_gaussians(checkpoint, model_path, (0.02, 4), 0.15)
                count = gaussian_count(model_path)
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
            add_gaussians(checkpoint, model_path, indices_path)

        if new_poses:
            with timer.stage("capture_views", loop=loop, n_views=len(new_poses)):
                capture_views(args.las_file, data_dir, new_poses, "POINTS")

        with timer.stage("train", loop=loop, start=checkpoint, end=checkpoint + 6_000):
            train_gaussians(model_path, data_dir, checkpoint + 6_000,
                            f"{model_path}/chkpnt{checkpoint}.pth", [checkpoint + 6_000], sh_degree=args.sh_degree)
        clean_iteration(model_path, checkpoint)
        checkpoint += 6_000

        if (eval_dir / "eval.json").exists():
            with timer.stage("evaluate_gt", loop=loop, checkpoint=checkpoint):
                evaluate(model_path, args.las_file, eval_dir, args.sh_degree)

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
