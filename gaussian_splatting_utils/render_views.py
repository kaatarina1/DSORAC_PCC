"""Render the views of a trained 3DGS model listed in an eval config (data/<name>_eval/eval.json).

Cameras use the viewer convention (js/CameraPosition.js): PERSPECTIVE is look-at from eye to target, up = +Y,
fovDeg is the vertical FOV; ORTHOGRAPHIC is top-down (zoom, panX, panZ); SPHERICAL is an equirectangular
panorama at eye, rotated by yawDeg. Every view is rendered at the resolution of its ground truth image
(<config dir>/<gt>), so the two line up pixel for pixel, or at evalWidth (same aspect) if the config sets it.

python gaussian-splatting/utils/render_views.py -m output/lego_normals --config data/lego_normals_eval/eval.json
"""
import os
import sys
import json
import math
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gaussian-splatting"))

import numpy as np
import torch
import torchvision
from PIL import Image
from scene.cameras import MiniCam
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from render_equirect import look_rotation, load_gaussians, render_panorama, render_rgba


def make_perspective_camera(eye, target, width, height, fovy, znear=0.01, zfar=1000.0):
    eye, target = np.asarray(eye, dtype=np.float64), np.asarray(target, dtype=np.float64)
    fwd = target - eye
    up = np.array([0., 0., 1.]) if abs(fwd[1] / np.linalg.norm(fwd)) > 0.99 else np.array([0., 1., 0.])
    R_wc = look_rotation(fwd, up)
    T = -R_wc @ eye
    fovx = 2 * math.atan(math.tan(fovy / 2) * width / height)
    world_view = torch.tensor(getWorld2View2(R_wc.T, T)).transpose(0, 1).float().cuda()
    proj = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0, 1).cuda()
    full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    return MiniCam(width, height, fovy, fovx, znear, zfar, world_view, full_proj)


def make_ortho_topdown_camera(cam, offset, width, height, canvas_aspect, distance=1e4):
    """Orthographic top-down view as in js/CameraPosition.js computeOrthoTopDownMatrix (up = +Z, zoom = half extent
    of the shorter side). The rasterizer is perspective only, so the camera is put `distance` above the scene with
    a field of view that covers the same extent; the perspective error is (object height / distance)."""
    zoom = cam["zoom"]
    half_h = zoom if canvas_aspect > 1 else zoom / canvas_aspect
    eye = np.add([cam.get("panX", 0), distance, cam.get("panZ", 0)], offset)
    target = eye - np.array([0., 1., 0.])
    fovy = 2 * math.atan(half_h / distance)   # exact at the config's y = 0 plane
    return make_perspective_camera(eye, target, width, height, fovy, znear=1.0, zfar=2 * distance)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model_path", required=True)
    p.add_argument("--config", required=True, help="eval.json with canvasWidth/canvasHeight and cameras[{id, projection, gt, ...}]")
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--out_dir", help="default: <model_path>/eval")
    p.add_argument("--offset", type=float, nargs=3, default=[0., 0., 0.],
                   help="added to eye and target, to correct a config made for a differently centred LAS")
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--white_background", action="store_true")
    a = p.parse_args()

    with open(a.config) as f:
        cfg = json.load(f)
    out_dir = Path(a.out_dir or Path(a.model_path, "eval"))
    gt_dir = Path(a.config).parent
    os.makedirs(out_dir, exist_ok=True)

    gaussians, it = load_gaussians(a.model_path, a.iteration, a.sh_degree)
    pipe = argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    bg = torch.tensor([1., 1., 1.] if a.white_background else [0., 0., 0.], device="cuda")
    offset = np.array(a.offset)

    with torch.no_grad():
        for cam in cfg["cameras"]:
            projection = cam.get("projection", "PERSPECTIVE")
            w, h = Image.open(gt_dir / cam["gt"]).size
            if "evalWidth" in cfg:   # render (and compare) smaller than the GT, keeping its aspect
                w, h = cfg["evalWidth"], round(h * cfg["evalWidth"] / w)
            if projection == "SPHERICAL":
                # Cube faces span 120 deg (overlapping, so the seams are blended), this keeps the panorama's pixel density
                img = render_panorama(gaussians, np.add(cam["eye"], offset), cam.get("yawDeg", 0.0), w, h,
                                      face=math.ceil(w * 120 / 360), fov_deg=120, white_background=a.white_background)
            else:
                if projection == "ORTHOGRAPHIC":
                    view = make_ortho_topdown_camera(cam, offset, w, h, cfg["canvasWidth"] / cfg["canvasHeight"])
                else:
                    view = make_perspective_camera(np.add(cam["eye"], offset), np.add(cam["target"], offset),
                                                   w, h, math.radians(cam.get("fovDeg", 45)))
                img = render_rgba(view, gaussians, pipe, bg).clamp(0, 1)
            out = out_dir / f"{cam['id']}_gs.png"
            torchvision.utils.save_image(img, out)
            print(f"Saved {out} ({w}x{h}, {projection}) from iteration {it}")


if __name__ == "__main__":
    main()
