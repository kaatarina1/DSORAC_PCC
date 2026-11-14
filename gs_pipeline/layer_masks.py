"""Mask each DSORAC depth layer with SAM2 and re-composite the capture images.

capture.js --dumpLayers writes <out>/layers/<image>.bin/.json next to <out>/images/<image>.png.
This replaces every such image by a composite in which the gap-filled pixels of each layer are
kept only inside a SAM2 mask of that layer; pixels with points always keep alpha 1. The original
composite is moved to <out>/images_unmasked/ and the dumps are deleted.

Runs in the sam2 conda env (see apply_layer_masks in gs_pipeline/dataset.py):
  python -m gs_pipeline.layer_masks <out> [--checkpoint ...]
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_CHECKPOINT = "D:/Katarina/sam_test/checkpoints/sam2.1_hiera_large.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


# ----------------------------------------------------------------------------- composite.wgsl in numpy

def load_layers(stem):
    # Not Path.with_suffix: the file names contain the camera coordinates, i.e. dots
    meta = json.loads(Path(f"{stem}.json").read_text())
    L, H, W = meta["layers"], meta["height"], meta["width"]
    data = np.fromfile(f"{stem}.bin", dtype="<f4").reshape(L, H, W, 6)
    return data, meta


def layer_alphas(data, depths, radius_ratio):
    """Per-layer alpha exactly as in shaders/composite.wgsl, shape [L, H, W]."""
    L, H, W, _ = data.shape
    depths = np.asarray(depths, dtype=np.float32)
    d_min, d_max = depths.min(), depths.max()
    d_range = max(d_max - d_min, 1e-5)
    density_scale = 1.0 / (radius_ratio * radius_ratio)
    s = int(max(2.0, W * 0.1))                       # i32(max(2.0, f32(size.x) * 0.1))

    def shifted(img, dy, dx):                        # textureLoad at clamp(index + (dx, dy))
        ys = np.clip(np.arange(H) + dy, 0, H - 1)
        xs = np.clip(np.arange(W) + dx, 0, W - 1)
        return img[ys][:, xs]

    alphas = np.zeros((L, H, W), dtype=np.float32)
    for i in range(L):
        sdf, den, pts = data[i, ..., 3], data[i, ..., 4], data[i, ..., 5]
        has_data = pts > 0.01
        closeness = 1.0 - (depths[i] - d_min) / d_range
        k = (3.0 + 6.0 * closeness) * 1000
        sdf_alpha = np.exp(-(sdf * sdf * k))
        dd = lambda img: np.minimum(img * density_scale, 1.0)
        dC = dd(den)
        dN, dS = dd(shifted(den, -s, 0)), dd(shifted(den, s, 0))
        dE, dW = dd(shifted(den, 0, s)), dd(shifted(den, 0, -s))
        d_min4 = np.minimum(np.minimum(dN, dS), np.minimum(dE, dW))
        d_avg4 = (dN + dS + dE + dW) * 0.25
        t = 0.3 / (0.3 + (1.0 - 0.3) * radius_ratio)
        directional = d_min4 ** 2 + (d_avg4 ** 2 - d_min4 ** 2) * t
        support = np.maximum(dC * dC, directional)
        confidence = 0.3 + 0.7 * np.clip(support * 3.0, 0, 1)
        alpha = np.where(has_data, 1.0, sdf_alpha * confidence)
        alpha = np.where((sdf > 0.3) & ~has_data, 0.0, alpha)
        alphas[i] = np.where(alpha < 0.005, 0.0, alpha)
    return alphas


def composite(data, alphas):
    H, W = alphas.shape[1:]
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    a = np.zeros((H, W), dtype=np.float32)
    for i in range(len(alphas)):
        al = alphas[i]
        # composite.wgsl `continue`s for alpha < 0.005; the skipped pixels hold NaN colours
        # (the SOR fill leaves them undefined), and NaN * 0 would still poison the result.
        on = (al > 0)[..., None]
        rgb = np.where(on, rgb + (data[i, ..., :3] - rgb) * al[..., None], rgb)
        a = al + a * (1 - al)
    return np.concatenate([rgb, a[..., None]], axis=-1)


def to_uint8(rgba):                                  # toRGBA8 in CaptureRenderer.js
    return np.clip(np.floor(np.nan_to_num(rgba) * 255), 0, 255).astype(np.uint8)


def layer_preview(layer, alpha):
    """What SAM sees: the layer alone, as composite.wgsl would draw it, on white."""
    rgb = np.clip(np.nan_to_num(layer[..., :3]), 0, 1) * alpha[..., None] + (1 - alpha[..., None])
    return (rgb * 255).astype(np.uint8)


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


# ----------------------------------------------------------------------------- SAM2

class Sam:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, config=DEFAULT_CONFIG, auto=False):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.torch = torch
        model = build_sam2(config, checkpoint, device="cuda")
        self.predictor = SAM2ImagePredictor(model)
        self.generator = None
        if auto:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            self.generator = SAM2AutomaticMaskGenerator(model, points_per_side=32, pred_iou_thresh=0.7,
                                                        stability_score_thresh=0.85, min_mask_region_area=50)

    def box(self, img, has_data, n_points=16, rng=None):
        """One mask prompted with the bounding box of the layer's points + up to n_points of them."""
        ys, xs = np.nonzero(has_data)
        rng = rng or np.random.default_rng(0)
        pick = rng.choice(len(ys), size=min(n_points, len(ys)), replace=False)
        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            self.predictor.set_image(img)
            masks, scores, _ = self.predictor.predict(
                point_coords=np.stack([xs[pick], ys[pick]], 1).astype(np.float32),
                point_labels=np.ones(len(pick), dtype=np.int32),
                box=np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32),
                multimask_output=True)
        return masks[np.argmax(scores)].astype(bool) | has_data

    def auto(self, img, has_data, min_point_cover=0.3):
        """Union of automatic masks that the layer's points cover for at least min_point_cover."""
        from scipy.ndimage import binary_dilation
        near_points = binary_dilation(has_data, structure=disk(2))
        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            masks = self.generator.generate(img)
        out = np.zeros(has_data.shape, dtype=bool)
        for m in masks:
            seg = m["segmentation"]
            if seg.sum() and near_points[seg].mean() >= min_point_cover:
                out |= seg
        return out | has_data, len(masks)


def masked_alphas(sam, data, alphas):
    """alphas with the gap-filled pixels of every layer limited to that layer's SAM2 mask."""
    has_data = data[..., 5] > 0.01
    out = alphas.copy()
    kept = []
    for i in range(len(data)):
        if not has_data[i].any():
            continue
        filled = (alphas[i] > 0) & ~has_data[i]
        mask = sam.box(layer_preview(data[i], alphas[i]), has_data[i])
        out[i] = np.where(has_data[i], 1.0, alphas[i] * mask)
        kept.append(float((filled & mask).sum() / max(filled.sum(), 1)))
    return out, kept


# ----------------------------------------------------------------------------- apply to a capture folder

def apply(out_dir, sam):
    out_dir = Path(out_dir)
    layers_dir, images_dir, orig_dir = out_dir / "layers", out_dir / "images", out_dir / "images_unmasked"
    orig_dir.mkdir(exist_ok=True)
    metas = sorted(layers_dir.glob("*.json"))
    log_path = out_dir / "layer_masks.jsonl"
    t0 = time.perf_counter()
    for n, meta_path in enumerate(metas, 1):
        stem = str(meta_path)[:-len(".json")]
        data, meta = load_layers(stem)
        alphas = layer_alphas(data, meta["depths"], meta["radiusRatio"])
        new_alphas, kept = masked_alphas(sam, data, alphas)
        img_path = images_dir / meta["image"]
        os.replace(img_path, orig_dir / meta["image"])
        # Write a new file instead of overwriting in place, so a hard link to it is never modified
        tmp = img_path.with_name(img_path.name + ".tmp")
        Image.fromarray(to_uint8(composite(data, new_alphas))).save(tmp, format="PNG")
        os.replace(tmp, img_path)
        with open(log_path, "a") as f:
            f.write(json.dumps({"image": meta["image"], "layers": meta["layers"],
                                "filled_kept": kept}) + "\n")
        os.remove(f"{stem}.bin")
        os.remove(meta_path)
        if n % 25 == 0 or n == len(metas):
            print(f"[layer_masks] {n}/{len(metas)} images, {time.perf_counter() - t0:.0f}s", flush=True)
    try:
        layers_dir.rmdir()
    except OSError:
        pass
    return len(metas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", help="capture.js output folder with layers/ and images/")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    a = p.parse_args()
    if not any(Path(a.out_dir, "layers").glob("*.json")):
        print(f"[layer_masks] no layer dumps in {a.out_dir}", flush=True)
        return
    apply(a.out_dir, Sam(a.checkpoint, a.config))


if __name__ == "__main__":
    main()
