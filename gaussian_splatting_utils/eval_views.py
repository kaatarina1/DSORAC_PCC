"""Compare renders with ground truth images, print PSNR / SSIM / LPIPS.

For each camera in the eval config (any projection), compares <eval_dir>/<id>_gs.png (from render_views.py)
with <config dir>/<gt>. Metrics are averaged over valid pixels: those kept by the optional global "mask"
and per-camera "mask" (white = keep, black = ignore) and not transparent in the ground truth. Within them,
each pixel is weighted by the render's alpha (render_views.py stores the accumulated opacity), so empty
background counts 0 and half-covered pixels count half. The render is already composited over black, so the
ground truth is composited the same way (multiplied by alpha), and ignored pixels are black in both, so they
don't leak into SSIM / LPIPS windows next to them. Without masks and alpha the whole image is compared.

Next to the renders it also writes, per camera, <id>_gt_masked.png / <id>_gs_masked.png (ignored pixels
transparent) and error maps <id>_err_se.png / _err_ssim.png / _err_lpips.png (bright = worse, fixed scale
from ERROR_MAX so iterations can be compared, ignored pixels transparent).

python gaussian-splatting/utils/eval_views.py output/lego_normals/eval/iteration_30000 data/lego_normals_eval/eval.json
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import lpips
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gaussian-splatting"))
from eval_panoramas import compute_metrics, error_maps, load_mask

# Error that maps to the brightest colour: squared error 0.25 = colour off by 0.5, SSIM and LPIPS errors in [0, 1]
ERROR_MAX = {"se": 0.25, "ssim": 1.0, "lpips": 1.0}


def save_masked(path, img, weight):
    rgb = (img * 255).round().astype(np.uint8)
    alpha = (weight * 255).round().astype(np.uint8)[..., None]
    Image.fromarray(np.concatenate([rgb, alpha], -1), "RGBA").save(path)


def save_error_maps(eval_dir, id, maps, weight):
    errors = {"se": maps["se"], "ssim": 1 - maps["ssim"], "lpips": maps["lpips"]}   # all: higher = worse
    for name, err in errors.items():
        rgba = plt.get_cmap("inferno")(np.clip(err / ERROR_MAX[name], 0, 1))
        rgba[..., 3] = weight
        plt.imsave(eval_dir / f"{id}_err_{name}.png", rgba)


def load_valid(gt_img, gt_dir, cfg, cam, size):
    valid = np.ones(size[::-1], dtype=bool)             # PIL size is (w, h), arrays are (h, w)
    for name in (cfg.get("mask"), cam.get("mask")):
        if name:
            valid &= load_mask(gt_dir / name, size)
    return valid


def load_alpha(rend_img):
    if "A" not in rend_img.getbands():
        return np.ones(rend_img.size[::-1], dtype=np.float32)
    return np.asarray(rend_img.getchannel("A"), dtype=np.float32) / 255


def main():
    p = argparse.ArgumentParser()
    p.add_argument("eval_dir", help="where the renders are")
    p.add_argument("config", help="eval.json, ground truth images and masks are next to it")
    a = p.parse_args()

    eval_dir = Path(a.eval_dir)
    gt_dir = Path(a.config).parent
    with open(a.config) as f:
        cfg = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lp = lpips.LPIPS(net="vgg", spatial=True, verbose=False).to(device)
    results = {}
    for cam in cfg["cameras"]:
        id = cam["id"]
        rend_img = Image.open(eval_dir / f"{id}_gs.png")
        gt_img = Image.open(gt_dir / cam["gt"])
        size = rend_img.size
        rend = np.asarray(rend_img.convert("RGB"), dtype=np.float32) / 255
        orig = np.asarray(gt_img.convert("RGB").resize(size, Image.LANCZOS), dtype=np.float32) / 255
        weight = load_valid(gt_img, gt_dir, cfg, cam, size) * load_alpha(rend_img)
        rend *= (weight > 0)[..., None]   # already composited over black with its alpha
        orig *= weight[..., None]          # composite the ground truth the same way

        maps = error_maps(rend, orig, lp, device)
        r = results[id] = compute_metrics(rend, orig, weight, lp, device, maps)
        save_masked(eval_dir / f"{id}_gt_masked.png", orig, weight)
        save_masked(eval_dir / f"{id}_gs_masked.png", rend, weight)
        save_error_maps(eval_dir, id, maps, weight)
        print(f"{id:<28} PSNR {r['PSNR']:6.2f}  SSIM {r['SSIM']:.4f}  LPIPS {r['LPIPS']:.4f}  (valid {r['valid']:.0%})")

    mean = {k: np.mean([r[k] for r in results.values()]) for k in ("PSNR", "SSIM", "LPIPS")}
    print(f"{'MEAN':<28} PSNR {mean['PSNR']:6.2f}  SSIM {mean['SSIM']:.4f}  LPIPS {mean['LPIPS']:.4f}")
    with open(eval_dir / "metrics_views.json", "w") as f:
        json.dump({"views": results, "mean": mean}, f, indent=2)
    return results


if __name__ == "__main__":
    main()
