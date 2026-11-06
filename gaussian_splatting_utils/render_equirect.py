"""Render equirectangular 360° panoramas from a trained 3DGS model.

The 3DGS rasterizer supports only pinhole cameras, so we render 6 cube faces
from the same position and resample them into an equirectangular image.
Camera convention matches computeSphericalCameraMatrix in js/CameraPosition.js:
view = rotateY(yaw) * (p - eye), centre column looks along -Z, top row is straight up.

Single panorama:
python gaussian-splatting/utils/render_equirect.py -m output/bezigrad_normals --pos 0 80 0 --out pano.png
From a config (see evalConfig_gs.json):
python gaussian-splatting/utils/render_equirect.py -m output/ljubljana_1_normals --config evalConfig_gs.json
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
import torch.nn.functional as F
import torchvision
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


def look_rotation(forward, up):
    """World->camera rotation in COLMAP/OpenCV convention (x right, y down, z forward)."""
    z = forward / np.linalg.norm(forward)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])                             # rows = camera axes in world


def make_camera(R_wc, position, size, fov, znear=0.01, zfar=1000.0):
    T = -R_wc @ position
    world_view = torch.tensor(getWorld2View2(R_wc.T, T)).transpose(0, 1).float().cuda()
    proj = getProjectionMatrix(znear, zfar, fov, fov).transpose(0, 1).cuda()
    full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    return MiniCam(size, size, fov, fov, znear, zfar, world_view, full_proj)


def load_gaussians(model_path, iteration=-1, sh_degree=3):
    pc_dir = Path(model_path, "point_cloud")
    it = iteration if iteration > 0 else max(int(d.name.split("_")[1]) for d in pc_dir.glob("iteration_*"))
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(pc_dir / f"iteration_{it}" / "point_cloud.ply"))
    return gaussians, it


@torch.no_grad()
def render_rgba(view, gaussians, pipe, bg):
    """4xHxW: colour over bg + accumulated opacity. With every colour 1 over a black background the
    rasterizer's sum(c_i * alpha_i * T_i) + T_final * bg becomes sum(alpha_i * T_i) = 1 - T_final = alpha."""
    img = render(view, gaussians, pipe, bg)["render"]
    alpha = render(view, gaussians, pipe, torch.zeros_like(bg), override_color=torch.ones_like(gaussians.get_xyz))["render"][:1]
    return torch.cat([img, alpha])


def rotate_frame(fwd, right, up, rotation_fix_deg):
    """Tilt the camera frame by a small rotation vector given in degrees in the camera's own (forward, right, up)
    axes, i.e. [roll, pitch, yaw]. It is applied as one rotation (Rodrigues' formula), exactly as the pose
    alignment fitted it, so the result does not depend on an order of roll / pitch / yaw."""
    w = np.radians(np.asarray(rotation_fix_deg, dtype=np.float64))
    angle = np.linalg.norm(w)
    if angle == 0:
        return fwd, right, up
    k = w / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
    fwd, right, up = R @ np.stack([fwd, right, up])      # rows of R @ B are the corrected camera axes in world
    return fwd, right, up


def render_panorama(gaussians, position, yaw_deg=0.0, width=2048, height=None,
                    face=1024, fov_deg=120.0, white_background=False, rotation_fix_deg=(0.0, 0.0, 0.0)):
    """Returns a 4xHxW tensor (RGB + alpha) in [0, 1]. rotation_fix_deg = [roll, pitch, yaw] tilt correction."""
    height = height or width // 2
    pipe = argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    bg = torch.tensor([1., 1., 1.] if white_background else [0., 0., 0.], device="cuda")

    # Scene is y-up (pipeline uses x, las_z, -las_y)
    up = np.array([0., 1., 0.])
    yaw = math.radians(yaw_deg)
    fwd = np.array([math.sin(yaw), 0., -math.cos(yaw)])
    right = np.cross(fwd, up)
    fwd, right, up = rotate_frame(fwd, right, up, rotation_fix_deg)
    dirs = [fwd, -fwd, right, -right, up, -up]
    ups = [up, up, up, up, -fwd, fwd]
    pos = np.array(position, dtype=np.float64)
    fov = math.radians(fov_deg)

    # The CUDA rasterizer draws every splat in front of the camera, even when its centre lies far outside the
    # face; it clamps such splats to the image border and stretches them into haze and streaks. Like the web
    # viewer, each face hides splats whose centre is outside 1.2x its frustum (|x/z|, |y/z| < 1.2 tan(fov/2)).
    # The cube faces cover the whole sphere, so every splat is still drawn in the face that contains its centre.
    xyz = gaussians.get_xyz
    eye = torch.tensor(pos, dtype=torch.float32, device="cuda")
    cull_lim = 1.2 * math.tan(fov / 2)
    opacity_backup = gaussians._opacity.data.clone()

    faces, rots = [], []
    for d, u in zip(dirs, ups):
        R = look_rotation(d, u)
        Rt = torch.tensor(R, dtype=torch.float32, device="cuda")
        c = (xyz - eye) @ Rt.T                              # splat centres in this face's camera frame
        z = c[:, 2]
        keep = (z > 0.01) & ((c[:, 0] / z).abs() < cull_lim) & ((c[:, 1] / z).abs() < cull_lim)
        gaussians._opacity.data[~keep] = -20.0              # sigmoid(-20) ~ 2e-9, skipped by the rasterizer
        try:
            img = render_rgba(make_camera(R, pos, face, fov), gaussians, pipe, bg)   # 4xSxS
        finally:
            gaussians._opacity.data.copy_(opacity_backup)    # next face starts from the original opacities
        faces.append(img.clamp(0, 1))
        rots.append(Rt)

    # Equirectangular directions: top row = straight up, centre column = yaw direction
    lon = (torch.arange(width, device="cuda") + 0.5) / width * 2 * math.pi - math.pi
    lat = math.pi / 2 - (torch.arange(height, device="cuda") + 0.5) / height * math.pi
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")
    f, r, u = (torch.tensor(v, dtype=torch.float32, device="cuda") for v in (fwd, right, up))
    d = (torch.cos(lat) * torch.cos(lon))[..., None] * f + (torch.cos(lat) * torch.sin(lon))[..., None] * r + torch.sin(lat)[..., None] * u

    # Neighbouring faces render the same splats slightly differently near their edges, so a hard switch
    # between faces leaves visible seams. Each face has full weight inside its 90 deg core and fades out
    # smoothly towards the edge of its fov, where the faces overlap; the panorama is the weighted average.
    assert fov_deg > 90, "faces must overlap (fov_deg > 90) to blend the seams"
    cam = torch.stack([d @ R.T for R in rots])            # 6xHxWx3 in each face's camera frame
    tan_half = math.tan(fov / 2)
    pano = torch.zeros(4, height, width, device="cuda")
    weight_sum = torch.zeros(height, width, device="cuda")
    for i, img in enumerate(faces):
        c = cam[i]
        z = c[..., 2].clamp_min(1e-6)
        xy = torch.stack([c[..., 0] / z, c[..., 1] / z], -1)
        # 1 at the 90 deg core boundary (tan 45 deg), tan_half at the edge of the face
        edge = xy.abs().amax(-1)
        w = ((tan_half - edge) / (tan_half - 1)).clamp(0, 1)
        w = w * w * (3 - 2 * w) * (c[..., 2] > 0)          # smoothstep, no weight behind the face
        s = F.grid_sample(img[None], (xy / tan_half)[None], mode="bilinear", align_corners=False)[0]
        pano += s * w
        weight_sum += w
    return pano / weight_sum.clamp_min(1e-6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model_path", required=True)
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--config", help="JSON with sphericalWidth/sphericalHeight and locations[{name, position, yawOffset}]")
    p.add_argument("--out_dir", help="where config renders are saved (default: <model_path>/eval)")
    p.add_argument("--pos", type=float, nargs=3, help="camera position in model (scene) coordinates")
    p.add_argument("--yaw", type=float, default=0.0, help="degrees, same as yawDeg/yawOffset")
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--face", type=int, default=1024, help="cube face resolution")
    p.add_argument("--fov", type=float, default=120.0, help="face FOV in degrees, >90 so faces overlap and seams are blended")
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--white_background", action="store_true")
    p.add_argument("--out", default="pano.png")
    a = p.parse_args()

    gaussians, it = load_gaussians(a.model_path, a.iteration, a.sh_degree)
    opts = dict(face=a.face, fov_deg=a.fov, white_background=a.white_background)

    if a.config:
        with open(a.config) as f:
            cfg = json.load(f)
        out_dir = Path(a.out_dir or Path(a.model_path, "eval"))
        os.makedirs(out_dir, exist_ok=True)
        for loc in cfg["locations"]:
            pano = render_panorama(gaussians, loc["position"], loc.get("yawOffset", 0.0),
                                   cfg.get("sphericalWidth", a.width), cfg.get("sphericalHeight"), **opts)
            out = out_dir / f"{loc['name']}.png"
            torchvision.utils.save_image(pano, out)
            print(f"Saved {out} from iteration {it} at {loc['position']} (yaw {loc.get('yawOffset', 0.0)})")
    else:
        if a.pos is None:
            p.error("either --config or --pos is required")
        pano = render_panorama(gaussians, a.pos, a.yaw, a.width, **opts)
        torchvision.utils.save_image(pano, a.out)
        print(f"Saved {a.out} from iteration {it} at {a.pos}")


if __name__ == "__main__":
    main()
