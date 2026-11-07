"""Render the training views of a trained 3DGS model and compute the per-view and per-pixel error on the GPU,
without writing renders / gt as PNG (replaces render.py + per_pixel_metrics.py for the psnr metric).

Writes the same files the old two-step path did, so the code reading them stays unchanged:
  <model>/train/ours_<it>/scores.json      {"00000": {"psnr": <MSE>, "ssim": null, "lpips": null}, ...}
  <model>/train/ours_<it>/psnr/<idx>.npy   per-pixel squared error (H x W, float32), unless --no_maps

python gaussian_splatting_utils/measure_views.py -s <data_dir> -m <model_path> --iteration 17000
"""
import sys
import json
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gaussian-splatting"))

import numpy as np
import torch
from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    SPARSE_ADAM_AVAILABLE = False


def as_png(img):
    # Same 8-bit rounding as torchvision.utils.save_image, so the values match the old PNG-based measurement
    return (img * 255 + 0.5).clamp(0, 255).floor() / 255


@torch.no_grad()
def main():
    parser = ArgumentParser(description="Per-view error of the training views without writing PNGs")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--metric", choices=["psnr"], default="psnr", help="ssim / lpips still use the PNG path")
    parser.add_argument("--no_maps", action="store_true", help="do not save the per-pixel error maps")
    args = get_combined_args(parser)   # reads cfg_args from the model folder, like render.py
    dataset, pipe = model.extract(args), pipeline.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)   # same camera order as render.py
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")

    method_dir = Path(dataset.model_path, "train", f"ours_{scene.loaded_iter}")
    maps_dir = method_dir / args.metric
    maps_dir.mkdir(parents=True, exist_ok=True)

    scores = {}
    views = scene.getTrainCameras()
    for idx, view in enumerate(views):
        rendering = render(view, gaussians, pipe, background, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
        se = ((as_png(rendering) - as_png(view.original_image[0:3])) ** 2).mean(0)   # per-pixel squared error
        name = f"{idx:05d}"
        # "psnr" holds the MSE, exactly like per_pixel_metrics.py; gs_pipeline.quality.calculate_quality converts it to dB
        scores[name] = {"psnr": float(se.mean()), "ssim": None, "lpips": None}
        if not args.no_maps:
            np.save(maps_dir / f"{name}.npy", se.cpu().numpy())
    (method_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    print(f"Measured {len(views)} views -> {method_dir / 'scores.json'}")


if __name__ == "__main__":
    main()
