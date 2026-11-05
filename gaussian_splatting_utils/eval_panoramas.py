"""Compare rendered panoramas with original 360 photos inside a mask, print PSNR / SSIM / LPIPS.

Expects in <eval_dir>:  mask.jpg, sN_orig.jpg, sN_mask.png  and renders named sN_<anything>.png
Mask convention: white = keep, black = ignore.

python gaussian-splatting/utils/eval_panoramas.py output/ljubljana_1_normals/eval s1_gs_billboards.png s2_gs_billboards.png ...
"""
import sys
from pathlib import Path
import numpy as np
import torch
import lpips
from PIL import Image
from skimage.metrics import structural_similarity


def load_mask(path, size):
    m = Image.open(path).convert("L").resize(size, Image.BILINEAR)
    return np.asarray(m) > 127


def evaluate(eval_dir, render_name, lp, device):
    eval_dir = Path(eval_dir)
    prefix = render_name.split("_")[0]                                     # "s1"
    rend_img = Image.open(eval_dir / render_name)
    size = rend_img.size                                                    # compare at render resolution
    orig_img = Image.open(next(eval_dir.glob(f"{prefix}_orig.*")))

    rend = np.asarray(rend_img.convert("RGB"), dtype=np.float32) / 255
    orig = np.asarray(orig_img.convert("RGB").resize(size, Image.LANCZOS), dtype=np.float32) / 255

    # 1) global mask, 2) per-image mask, 3) transparent regions of the original (if it has alpha)
    valid = load_mask(eval_dir / "mask.jpg", size)
    valid &= load_mask(eval_dir / f"{prefix}_mask.png", size)
    
    orig[~valid] = 0
    rend[~valid] = 0
    alpha = (valid * 255).astype(np.uint8)[..., None]
    for img, tag in ((orig, "orig_masked"), (rend, "render_masked")):
        rgba = np.concatenate([(img * 255).round().astype(np.uint8), alpha], -1)
        Image.fromarray(rgba, "RGBA").save(eval_dir / f"{Path(render_name).stem}_{tag}.png")

    return compute_metrics(rend, orig, valid, lp, device)


def error_maps(rend, orig, lp, device):
    """Per-pixel HxW maps of rend vs orig: squared error, SSIM and LPIPS."""
    se = ((rend - orig) ** 2).mean(-1)
    _, ssim_map = structural_similarity(rend, orig, data_range=1.0, channel_axis=-1, gaussian_weights=True,
                                        sigma=1.5, use_sample_covariance=False, full=True)
    t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].to(device) * 2 - 1
    with torch.no_grad():
        lp_map = lp(t(rend), t(orig))[0, 0].cpu().numpy()
    return {"se": se, "ssim": ssim_map.mean(-1), "lpips": lp_map}


def compute_metrics(rend, orig, weight, lp, device, maps=None):
    """rend, orig: HxWx3 float in [0, 1]; weight: HxW bool mask or float in [0, 1].
    Metrics are weighted averages over the pixels. maps: error_maps(rend, orig, ...) if already computed."""
    maps = maps or error_maps(rend, orig, lp, device)
    w = weight.astype(np.float64)
    avg = lambda m: (m * w).sum() / max(w.sum(), 1e-10)
    psnr = 10 * np.log10(1 / max(avg(maps["se"]), 1e-10))
    return {"PSNR": float(psnr), "SSIM": float(avg(maps["ssim"])), "LPIPS": float(avg(maps["lpips"])),
            "valid": float(w.mean())}


def main(eval_dir, render_names):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lp = lpips.LPIPS(net="vgg", spatial=True, verbose=False).to(device)
    results = {}
    for name in render_names:
        r = evaluate(eval_dir, name, lp, device)
        results[name] = r
        print(f"{name:<28} PSNR {r['PSNR']:6.2f}  SSIM {r['SSIM']:.4f}  LPIPS {r['LPIPS']:.4f}  (valid {r['valid']:.0%})")
    mean = {k: np.mean([r[k] for r in results.values()]) for k in ("PSNR", "SSIM", "LPIPS")}
    print(f"{'MEAN':<28} PSNR {mean['PSNR']:6.2f}  SSIM {mean['SSIM']:.4f}  LPIPS {mean['LPIPS']:.4f}")
    return results


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
