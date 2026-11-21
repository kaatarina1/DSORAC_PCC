"""Compare renders with ground truth images, print PSNR / SSIM / LPIPS.

For each camera in the eval config (any projection), compares <eval_dir>/<id>_gs.png (from render_views.py)
with <config dir>/<gt>. Three sets of metrics are written to metrics_views.json:

  views / mean                  (original, render-masked) Metrics are averaged over valid pixels: those kept by the
                                optional global "mask" and per-camera "mask" (white = keep, black = ignore). Within
                                them, each pixel is weighted by the render's alpha (render_views.py stores the
                                accumulated opacity), so empty background counts 0 and half-covered pixels count half,
                                and the ground truth is multiplied by the same alpha. Holes in the render are therefore
                                not counted. Kept so that earlier numbers stay comparable.
  views_gt_mask / mean_gt_mask  Only masks that come from the ground truth: the eval.json masks, a camera's optional
                                "sceneMask" (sky and what is outside the point cloud, scripts/make_scene_masks.py),
                                and for ground truth with alpha (e.g. lego) the union of the object and the render, so a
                                hole (object in the ground truth, nothing rendered) counts fully and a floater counts
                                by its alpha. Ground truth without alpha but with eval.json "background" (a plain
                                colour, e.g. human_1 on white) gets its alpha from that (background_alpha). Other
                                ground truth without alpha (photos) is compared on the whole masked area.
  views_full / mean_full        No mask at all: the whole image, ground truth composited over black like the render.

The render is composited over black, ignored pixels are black in both, so they don't leak into SSIM / LPIPS
windows next to them. "valid" in every view is the mean weight (share of the image the metric covers).

Next to the renders it also writes, per camera, <id>_gt_masked.png / <id>_gs_masked.png (ignored pixels
transparent) and error maps <id>_err_se.png / _err_ssim.png / _err_lpips.png (bright = worse, fixed scale
from ERROR_MAX so iterations can be compared, ignored pixels transparent), all for the original metric.

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
METRICS = ("PSNR", "SSIM", "LPIPS")


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


def load_valid(gt_dir, cfg, cam, size):
    valid = np.ones(size[::-1], dtype=bool)             # PIL size is (w, h), arrays are (h, w)
    for name in (cfg.get("mask"), cam.get("mask")):
        if name:
            valid &= load_mask(gt_dir / name, size)
    return valid


def load_alpha(img, size=None):
    """Alpha channel in [0, 1] (all ones without one), resized to size if given."""
    if "A" not in img.getbands():
        return np.ones((size or img.size)[::-1], dtype=np.float32)
    a = img.getchannel("A")
    if size and a.size != size:
        a = a.resize(size, Image.BILINEAR)
    return np.asarray(a, dtype=np.float32) / 255


def background_alpha(gt_img, color, size):
    """Alpha for a ground truth without alpha channel but on a plain background (eval.json "background", e.g.
    [255, 255, 255] for human_1): 0 where the pixel has exactly that colour and is connected to the image
    border, 1 elsewhere, so the same colour inside the object (white highlights) still counts as object."""
    from scipy import ndimage
    rgb = np.asarray(gt_img.convert("RGB"))
    bg = (rgb == np.asarray(color, dtype=rgb.dtype)).all(-1)
    labels, _ = ndimage.label(bg)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    alpha = Image.fromarray(((~np.isin(labels, border[border > 0])) * 255).astype(np.uint8))
    return np.asarray(alpha.resize(size, Image.BILINEAR), dtype=np.float32) / 255


def masked_metrics(rend, orig, weight, lp, device):
    """Metrics over `weight`, pixels with weight 0 black in both images."""
    keep = (weight > 0)[..., None]
    return compute_metrics(rend * keep, orig * keep, weight, lp, device)


def evaluate_dir(eval_dir, config_path, lp, device, verbose=True, keep_original=False):
    """Writes <eval_dir>/metrics_views.json (all three metric sets) and returns it. keep_original: take the
    original render-masked metric from the existing metrics_views.json instead of recomputing it (it does not
    depend on the masks added later) and skip its images; only gt_mask and full are computed."""
    eval_dir, config_path = Path(eval_dir), Path(config_path)
    gt_dir = config_path.parent
    cfg = json.loads(config_path.read_text())
    results = {"": {}, "_gt_mask": {}, "_full": {}}
    if keep_original:
        results[""] = json.loads((eval_dir / "metrics_views.json").read_text())["views"]
    for cam in cfg["cameras"]:
        id = cam["id"]
        rend_img = Image.open(eval_dir / f"{id}_gs.png")
        gt_img = Image.open(gt_dir / cam["gt"])
        size = rend_img.size
        rend = np.asarray(rend_img.convert("RGB"), dtype=np.float32) / 255     # already composited over black
        orig = np.asarray(gt_img.convert("RGB").resize(size, Image.LANCZOS), dtype=np.float32) / 255
        valid = load_valid(gt_dir, cfg, cam, size)
        rend_alpha = load_alpha(rend_img)
        gt_alpha = load_alpha(gt_img, size)
        if "A" not in gt_img.getbands() and cfg.get("background"):   # plain background instead of alpha
            gt_alpha = background_alpha(gt_img, cfg["background"], size)
        orig_black = orig * gt_alpha[..., None]                                  # ground truth over black

        # Original metric, unchanged: weighted by the render's alpha, ground truth multiplied by it
        if keep_original:
            r = results[""][id]
        else:
            weight = valid * rend_alpha
            rend_m = rend * (weight > 0)[..., None]
            orig_m = orig * weight[..., None]
            maps = error_maps(rend_m, orig_m, lp, device)
            r = results[""][id] = compute_metrics(rend_m, orig_m, weight, lp, device, maps)
            save_masked(eval_dir / f"{id}_gt_masked.png", orig_m, weight)
            save_masked(eval_dir / f"{id}_gs_masked.png", rend_m, weight)
            save_error_maps(eval_dir, id, maps, weight)

        # Masks from the ground truth only: holes count fully, floaters by their alpha. A camera's "sceneMask"
        # (scripts/make_scene_masks.py: sky and what is outside the point cloud) applies only here.
        scene = valid & load_mask(gt_dir / cam["sceneMask"], size) if cam.get("sceneMask") else valid
        g = results["_gt_mask"][id] = masked_metrics(rend, orig_black, scene * np.maximum(gt_alpha, rend_alpha),
                                                     lp, device)
        # No mask at all
        f = results["_full"][id] = masked_metrics(rend, orig_black, np.ones_like(rend_alpha), lp, device)
        if verbose:
            for name, m in (("render-masked", r), ("gt_mask", g), ("full", f)):
                print(f"{id:<28} {name:<14} PSNR {m['PSNR']:6.2f}  SSIM {m['SSIM']:.4f}  LPIPS {m['LPIPS']:.4f}"
                      f"  (valid {m['valid']:.0%})")

    out = {}
    for suffix, views in results.items():
        mean = {k: float(np.mean([v[k] for v in views.values()])) for k in METRICS}
        out[f"views{suffix}"], out[f"mean{suffix}"] = views, mean
        if verbose:
            print(f"{'MEAN ' + (suffix[1:] or 'render-masked'):<43} PSNR {mean['PSNR']:6.2f}  "
                  f"SSIM {mean['SSIM']:.4f}  LPIPS {mean['LPIPS']:.4f}")
    (eval_dir / "metrics_views.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("eval_dir", help="where the renders are")
    p.add_argument("config", help="eval.json, ground truth images and masks are next to it")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lp = lpips.LPIPS(net="vgg", spatial=True, verbose=False).to(device)
    return evaluate_dir(a.eval_dir, a.config, lp, device)


if __name__ == "__main__":
    main()
