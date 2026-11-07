import os
import shutil
import subprocess
import sys
from pathlib import Path

from gs_pipeline.config import GS_DIR, GS_UTILS, GS_ENV


def run_in_gs_env(script, *args):
    # On Windows "conda" in PATH is only conda.bat, which subprocess can't start without a shell,
    # so use the real conda.exe from CONDA_EXE (set in an activated environment).
    conda = os.environ.get("CONDA_EXE", "conda")
    cmd = [conda, "run", "-n", GS_ENV, "--no-capture-output", "python", str(script), *map(str, args)]
    subprocess.run(cmd, check=True)


def clean_iteration(model_path, iteration):
    Path(model_path, f"chkpnt{iteration}.pth").unlink(missing_ok=True)
    shutil.rmtree(Path(model_path, "point_cloud", f"iteration_{iteration}"), ignore_errors=True)


def train_gaussians(model_path, data_dir, checkpoint=-1, checkpoint_model="", iterations=(), sh_degree=3):
    args = [
        "-s", data_dir, "-m", model_path,
        "--random_background", "--disable_viewer", "--sh_degree", sh_degree,
        "--test_iterations", "-1", "--position_lr_max_steps", "35000",
        "--densify_grad_threshold", "0.001",
    ]
    if checkpoint != -1:
        args += ["--checkpoint_iterations", checkpoint]
    if checkpoint_model != "":
        args += ["--start_checkpoint", checkpoint_model]
    if len(iterations) != 0:
        args += ["--iterations", max(iterations), "--save_iterations", *iterations]
    run_in_gs_env(GS_DIR / "train.py", *args)


def render_gaussians(model_path, iteration, data_dir):
    run_in_gs_env(GS_DIR / "render.py", "-s", data_dir, "-m", model_path, "--skip_test", "--iteration", iteration)


def evaluate_gaussians(method_dir, metric, save_maps=False):
    # Runs in the pipeline environment, not via run_in_gs_env: per_pixel_metrics.py needs lpips, skimage
    # and matplotlib, which were not verified to be installed in the gaussian_splatting env.
    # Check with: conda run -n gaussian_splatting python -c "import lpips, skimage, matplotlib"
    # and switch to run_in_gs_env if that succeeds.
    cmd = [sys.executable, str(GS_UTILS / "per_pixel_metrics.py"), str(method_dir), metric, "--no_png"]
    if not save_maps:
        cmd.append("--no_maps")
    subprocess.run(cmd, check=True)


def measure_views(model_path, checkpoint, data_dir, save_maps=False):
    # Renders and computes the psnr error in one GPU process, without writing renders / gt PNGs
    args = ["-s", data_dir, "-m", model_path, "--iteration", checkpoint]
    if not save_maps:
        args.append("--no_maps")
    run_in_gs_env(GS_UTILS / "measure_views.py", *args)


def prune_gaussians(checkpoint, model_path, size_factor=(0.05, 2), min_opacity=0.2, n_target=-1, seed=0):
    checkpoint_path = f"{model_path}/chkpnt{checkpoint}.pth"
    ply_path = f"{model_path}/point_cloud/iteration_{checkpoint}/point_cloud.ply"
    run_in_gs_env(
        GS_UTILS / "prune_gaussians.py",
        "--checkpoint", checkpoint_path,
        "--out_checkpoint", checkpoint_path,
        "--ply", ply_path,
        "--size_factor_min", size_factor[0],
        "--size_factor_max", size_factor[1],
        "--min_opacity", min_opacity,
        "--target_count", n_target,
        "--seed", seed,
    )


def add_gaussians(checkpoint, model_path, indices_path):
    checkpoint_path = f"{model_path}/chkpnt{checkpoint}.pth"
    ply_path = f"{model_path}/point_cloud/iteration_{checkpoint}/point_cloud.ply"
    run_in_gs_env(
        GS_UTILS / "add_gaussians.py",
        "--checkpoint", checkpoint_path,
        "--out_checkpoint", checkpoint_path,
        "--ply", ply_path,
        "--points", f"{model_path}/points.npy",
        "--colors", f"{model_path}/colors.npy",
        "--indices", indices_path,
    )
