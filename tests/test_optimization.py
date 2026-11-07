import argparse
import json
import tempfile
from pathlib import Path
import numpy as np

# Synthetic camera used by the unit tests (same image size as the pipeline)
W, H = 1024, 512
K_TEST = np.array([[500., 0, W / 2], [0, 500., H / 2], [0, 0, 1]])
C0 = 0.28209479177387814

failed = []

def check(name, ok, detail=""):
    ok = bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        failed.append(name)
    return ok

def out_dir(model):
    d = Path(model, "_tests")
    d.mkdir(exist_ok=True)
    return d

def model_paths(args):
    method_dir = Path(args.model, "train", f"ours_{args.iteration}")
    ply = Path(args.model, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    return method_dir, ply

def point_spacing(points):
    from scipy.spatial import cKDTree
    return np.median(cKDTree(points).query(points[::100], k=2)[0][:, 1])

def write_ply(path, xyz, rgb, error=None):
    # Binary PLY; the optional "error" property shows up in CloudCompare as a scalar field
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if error is not None:
        fields.append(("error", "<f4"))
    v = np.empty(len(xyz), dtype=fields)
    v["x"], v["y"], v["z"] = np.asarray(xyz, np.float32).T
    v["red"], v["green"], v["blue"] = np.asarray(rgb, np.uint8).T
    if error is not None:
        v["error"] = error
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              + ("property float error\n" if error is not None else "") + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        v.tofile(f)

def error_colors(E, seen, pct=99, cmap="inferno"):
    # Scale by a high percentile of the seen points so a few outliers don't darken everything
    import matplotlib
    e = np.clip(E / max(np.percentile(E[seen], pct), 1e-12), 0, 1)
    rgb = (matplotlib.colormaps[cmap](e)[:, :3] * 255).astype(np.uint8)
    rgb[~seen] = 128
    return rgb

def segment_points(a, b, n):
    s = np.linspace(0, 1, n)[:, None]
    return a + s * (b - a)

def frustum_points(K, R, t, width, height, scale, n=60):
    # Wireframe pyramid from the camera centre to the image corners at depth `scale`
    C = -R.T @ t
    corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]], float)
    W = C + scale * (np.linalg.inv(K) @ corners.T).T @ R
    segs = [(C, w) for w in W] + [(W[i], W[(i + 1) % 4]) for i in range(4)]
    return np.concatenate([segment_points(a, b, n) for a, b in segs])

def plane(n, z):
    xs, ys = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    return np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)])

def ply_vertex_count(path):
    with open(path, "rb") as f:
        for line in f:
            line = line.decode("ascii", "ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                return None

# ---------- 1-4: unit tests on synthetic data ----------

def test_bilinear(args):
    from gs_pipeline.optimization import bilinear
    img_x = np.tile(np.arange(10, dtype=float), (5, 1))                # img[y, x] = x
    img_y = np.repeat(np.arange(5, dtype=float)[:, None], 10, axis=1)  # img[y, x] = y
    uv = np.array([[2.5, 1.0], [0.0, 0.0], [9.0, 4.0], [3.25, 2.75]])  # includes both corners
    ex, ey = bilinear(img_x, uv), bilinear(img_y, uv)
    check("bilinear along x", np.allclose(ex, [2.5, 0, 9, 3.25]), ex)
    check("bilinear along y", np.allclose(ey, [1, 0, 4, 2.75]), ey)

def test_camera(args):
    from gs_pipeline.optimization import look_at, project, load_cameras
    target = np.zeros(3)
    for pos in (np.array([3., 2., 5.]), np.array([0., 5., 0.])):    # second one looks straight down
        R, t = look_at(pos, target)
        uv, z, valid = project(target[None], K_TEST, R, t, W, H)
        check(f"{pos}: target lands in image centre", valid[0] and np.allclose(uv[0], [W / 2, H / 2]), uv[0])
        check(f"{pos}: depth equals distance", np.isclose(z[0], np.linalg.norm(pos - target)), z[0])
        check(f"{pos}: camera centre -R^T t equals pos", np.allclose(-R.T @ t, pos))
        check(f"{pos}: R orthonormal with det +1",
              np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1))
        uv_right = project((target + 0.5 * R[0])[None], K_TEST, R, t, W, H)[0][0]
        check(f"{pos}: camera x is right in the image", uv_right[0] > W / 2, uv_right)

    pos = np.array([3., 2., 5.])
    R, t = look_at(pos, target)
    uv_up = project((target + [0, 0.5, 0])[None], K_TEST, R, t, W, H)[0][0]
    check("world up (+y) is up in the image", uv_up[1] < H / 2, uv_up)

    # Same layout camera_to_JSON writes: position = camera centre, rotation = camera->world
    with tempfile.TemporaryDirectory() as tmp:
        entry = {"id": 0, "img_name": "00000.png", "width": W, "height": H,
                 "position": pos.tolist(), "rotation": R.T.tolist(), "fx": 500., "fy": 500.}
        path = Path(tmp, "cameras.json")
        path.write_text(json.dumps([entry]))
        K2, R2, t2, w2, h2 = load_cameras(path)[0]
    check("load_cameras round trip", np.allclose(R2, R) and np.allclose(t2, t) and np.allclose(K2, K_TEST)
          and (w2, h2) == (W, H))

def test_visibility(args):
    from gs_pipeline.optimization import look_at, project, visibility
    R, t = look_at(np.array([0., 0., 3.]), np.zeros(3))
    for n in (200, 50):
        front, back = plane(n, 0.), plane(n, -1.)
        P = np.vstack([front, back])
        uv, z, valid = project(P, K_TEST, R, t, W, H)
        vis = visibility(uv, z, valid, W, H)
        f, b = vis[:len(front)].mean(), vis[len(front):].mean()
        if n == 200:
            check("dense front plane hides the back plane", f > 0.99 and b < 0.01, f"front {f:.3f}  back {b:.3f}")
        else:
            print(f"[INFO] sparse front plane ({n}x{n}): front {f:.3f}  back {b:.3f}  "
                  "(back leaks through the gaps between splats, expected)")

def test_sampling(args):
    from gs_pipeline.optimization import importance_sample
    E = np.tile([0., 0., 1., 1., 8.], 2000)
    expected = {0.0: [0, 0, .1, .1, .8], 0.5: [.1, .1, .15, .15, .5]}
    with np.errstate(divide="ignore"):
        for alpha, expect in expected.items():
            counts = np.zeros(5)
            for s in range(200):
                idx = importance_sample(E, 100, alpha=alpha, rng=np.random.default_rng(s))
                counts += np.bincount(idx % 5, minlength=5)
            share = counts / counts.sum()
            check(f"alpha={alpha}: share per error level", np.allclose(share, expect, atol=0.03),
                  f"got {np.round(share, 3)}  expected {expect}")
        idx = importance_sample(E, 100, rng=np.random.default_rng(0))
    check("100 unique indices", len(idx) == 100 and len(np.unique(idx)) == 100)
    check("n >= M returns everything", len(importance_sample(E[:50], 100)) == 50)

# ---------- 5-8: tests on a trained model (pipeline env) ----------

def test_projection(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from gs_pipeline.optimization import load_cameras, project, visibility
    method_dir, _ = model_paths(args)
    cams = load_cameras(Path(args.model, "cameras.json"))
    n_gt = len(list((method_dir / "gt").glob("*.png")))
    check("cameras.json matches the renders", len(cams) == n_gt, f"{len(cams)} cameras, {n_gt} gt images")

    P = np.load(Path(args.model, "points.npy"))[::args.stride]
    rgb = np.load(Path(args.model, "colors.npy"))[::args.stride]
    K, R, t, w, h = cams[args.cam]
    img = np.asarray(Image.open(method_dir / "gt" / f"{args.cam:05d}.png").convert("RGB"), dtype=np.float32) / 255
    uv, z, valid = project(P, K, R, t, w, h)
    vis = visibility(uv, z, valid, w, h)
    check("points visible in this camera", vis.sum() > 100, f"{vis.sum()} of {len(P)}")

    def colour_error(uv):
        u, v = np.round(uv[vis]).astype(int).T
        return np.abs(img[v, u] - rgb[vis]).mean()
    flipped = uv.copy()
    flipped[:, 0] = w - 1 - flipped[:, 0]
    e_ok, e_flip = colour_error(uv), colour_error(flipped)
    check("point colours match the image better than a mirrored projection", e_ok < 0.8 * e_flip,
          f"error {e_ok:.3f} vs mirrored {e_flip:.3f}")

    plt.figure(figsize=(12, 6))
    plt.imshow(img)
    plt.scatter(uv[vis, 0], uv[vis, 1], s=0.2, c="r")
    plt.axis("off")
    out = out_dir(args.model) / f"projection_cam{args.cam}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

def test_point_error(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gs_pipeline.optimization import load_cameras, compute_point_error
    method_dir, _ = model_paths(args)
    cams = load_cameras(Path(args.model, "cameras.json"))
    n_maps = len(list((method_dir / args.metric).glob("*.npy")))
    if not check("one error map per camera", n_maps == len(cams), f"{n_maps} maps, {len(cams)} cameras"):
        return

    P = np.load(Path(args.model, "points.npy"))[::args.stride]
    normals_path = Path(args.model, "normals.npy")
    N = np.load(normals_path)[::args.stride] if normals_path.exists() else None
    E, C = compute_point_error(P, N, cams, method_dir, args.metric)
    seen = C > 0
    check("most points seen by some camera", seen.mean() > 0.8, f"{seen.mean():.3f}")
    check("error finite and non-negative", np.isfinite(E).all() and (E >= 0).all())
    print(f"[INFO] error over seen points: median {np.median(E[seen]):.4g}  "
          f"p95 {np.percentile(E[seen], 95):.4g}  max {E.max():.4g}")

    out = out_dir(args.model)
    hi = seen & (E > np.percentile(E[seen], 95))
    plt.figure(figsize=(10, 10))
    plt.scatter(P[:, 0], P[:, 2], s=0.1, c="lightgray")
    plt.scatter(P[hi, 0], P[hi, 2], s=0.5, c="r")
    plt.axis("equal")
    plt.title(f"top 5% {args.metric} error (top-down, x-z)")
    plt.savefig(out / f"point_error_{args.metric}.png", dpi=150, bbox_inches="tight")

    # Coloured cloud for CloudCompare / MeshLab
    write_ply(out / f"point_error_{args.metric}.ply", P, error_colors(E, seen), E)
    print(f"wrote {out / f'point_error_{args.metric}.png'} and .ply")

def test_coverage(args):
    from gs_pipeline.optimization import read_ply_xyz, uncovered_mask
    _, ply = model_paths(args)
    G = read_ply_xyz(ply)
    P_all = np.load(Path(args.model, "points.npy"))
    print(f"[INFO] {len(G)} Gaussians, {len(P_all)} LAS points")

    gb, pb = np.percentile(G, [1, 99], axis=0), np.percentile(P_all, [1, 99], axis=0)
    scale = np.linalg.norm(pb[1] - pb[0])
    check("Gaussians and LAS points share a frame (1-99% bbox)", np.abs(gb - pb).max() < 0.05 * scale,
          f"max bbox difference {np.abs(gb - pb).max():.3f}, scene size {scale:.3f}")

    spacing = point_spacing(P_all)
    P = P_all[::args.stride]
    fractions = [uncovered_mask(P, G, spacing, k).mean() for k in (1, 2, 4)]
    print(f"[INFO] spacing {spacing:.4f}; uncovered at cover_k 1/2/4: " + " / ".join(f"{f:.3f}" for f in fractions))
    check("uncovered fraction drops as cover_k grows", fractions[0] >= fractions[1] >= fractions[2])
    check("at cover_k=2 only a minority is uncovered", fractions[1] < 0.5, f"{fractions[1]:.3f}")

def test_optimize(args):
    import shutil
    from scipy.spatial import cKDTree
    from gs_pipeline.optimization import optimize, read_ply_xyz
    method_dir, ply = model_paths(args)
    new_poses, picked = optimize(args.model, method_dir, ply, args.metric, budget_frac=args.budget_frac,
                                 n_views=args.n_views, stride=args.stride)
    P_all = np.load(Path(args.model, "points.npy"))
    G = read_ply_xyz(ply)
    budget = int(args.budget_frac * len(G))

    check("some points picked", len(picked) > 0, f"{len(picked)}")
    if len(picked):
        check("picked indices unique and inside points.npy",
              len(np.unique(picked)) == len(picked) and picked.min() >= 0 and picked.max() < len(P_all))
        check("picked count within budget", len(picked) <= budget, f"{len(picked)} / budget {budget}")
        spacing = point_spacing(P_all)
        d = cKDTree(G).query(P_all[picked], workers=-1)[0]
        check("picked points are not covered by Gaussians", (d > 2.0 * spacing).all(),
              f"closest picked point is {d.min() / spacing:.2f} spacings from a Gaussian")

    check("some new views", len(new_poses) > 0, f"{len(new_poses)}")
    tree = cKDTree(P_all)
    for i, (eye, target, ratio) in enumerate(new_poses):
        print(f"[INFO] view {i}: eye {np.round(eye, 2)}  target {np.round(target, 2)}  "
              f"distance {np.linalg.norm(eye - target):.2f}  nearest point to eye {tree.query(eye)[0]:.2f}")

    out = out_dir(args.model) / "new_poses.json"
    out.write_text(json.dumps([{"eye": list(map(float, e)), "target": list(map(float, t)), "radiusRatio": float(r)}
                               for e, t, r in new_poses], indent=2))
    print(f"wrote {out}")

    if args.las and new_poses:
        from gs_pipeline.dataset import capture_views
        views_dir = "data/_test_views"
        shutil.rmtree(views_dir, ignore_errors=True)
        capture_views(args.las, views_dir, new_poses, "POINTS")
        print(f"rendered the new views into {views_dir}/images, open them and check what they show")

def test_visualize(args):
    # Writes three layers for CloudCompare: error-coloured cloud, picked points, new + existing views
    from gs_pipeline.optimization import optimize, compute_point_error, load_cameras, look_at
    method_dir, ply = model_paths(args)
    out = out_dir(args.model) / f"visualize_{args.metric}"
    out.mkdir(exist_ok=True)

    new_poses, picked = optimize(args.model, method_dir, ply, args.metric, budget_frac=args.budget_frac,
                                 n_views=args.n_views, stride=args.stride)

    # optimize() does not return E, so compute it again on the same strided subset
    P_all = np.load(Path(args.model, "points.npy")).astype(np.float32)
    normals_path = Path(args.model, "normals.npy")
    N = np.load(normals_path).astype(np.float32)[::args.stride] if normals_path.exists() else None
    P = P_all[::args.stride]
    cams = load_cameras(Path(args.model, "cameras.json"))
    E, C = compute_point_error(P, N, cams, method_dir, args.metric)
    seen = C > 0
    write_ply(out / "error.ply", P, error_colors(E, seen), E)

    # picked indexes P_all, not the strided P
    write_ply(out / "picked.ply", P_all[picked], np.tile([0, 255, 255], (len(picked), 1)))

    K, _, _, w, h = cams[0]
    scene = np.linalg.norm(np.percentile(P_all, 99, axis=0) - np.percentile(P_all, 1, axis=0))
    xyz, rgb = [], []
    for eye, target, _ in new_poses:
        R, t = look_at(np.asarray(eye, float), np.asarray(target, float))
        dist = np.linalg.norm(eye - target)
        pts = np.concatenate([frustum_points(K, R, t, w, h, 0.2 * dist),
                              segment_points(eye, target, 200)])
        xyz.append(pts)
        rgb.append(np.tile([0, 255, 0], (len(pts), 1)))
    for Kc, R, t, wc, hc in cams:
        pts = frustum_points(Kc, R, t, wc, hc, 0.02 * scene, n=20)
        xyz.append(pts)
        rgb.append(np.tile([255, 255, 255], (len(pts), 1)))
    write_ply(out / "views.ply", np.concatenate(xyz), np.concatenate(rgb))

    check("some points picked", len(picked) > 0, f"{len(picked)}")
    check("some new views", len(new_poses) > 0, f"{len(new_poses)}")
    print(f"wrote error.ply, picked.ply ({len(picked)} points), views.ply "
          f"({len(new_poses)} new green, {len(cams)} existing white) into {out}")

# ---------- 9: checkpoint after add_gaussians.py (gaussian_splatting env) ----------

def test_checkpoint(args):
    import torch
    model_params, iteration = torch.load(args.checkpoint)
    N = model_params[1].shape[0]
    names = [None, "xyz", "f_dc", "f_rest", "scaling", "rotation", "opacity",
             "max_radii2D", "xyz_gradient_accum", "denom"]
    print(f"[INFO] iteration {iteration}, {N} Gaussians")
    for i in range(1, 10):
        check(f"[{i}] {names[i]} has N rows", model_params[i].shape[0] == N, tuple(model_params[i].shape))
    for g, s in model_params[10]["state"].items():
        for key in ("exp_avg", "exp_avg_sq"):
            check(f"adam group {g} {key} has N rows", s[key].shape[0] == N, tuple(s[key].shape))
    check("params 1-6 require grad", all(model_params[i].requires_grad for i in range(1, 7)))
    if args.ply:
        n_ply = ply_vertex_count(args.ply)
        check("ply vertex count equals checkpoint rows", n_ply == N, f"ply {n_ply}, checkpoint {N}")

    if args.added:
        n = args.added
        scales = torch.exp(model_params[4][-n:]).detach().cpu().numpy()
        opacity = torch.sigmoid(model_params[6][-n:]).detach().cpu().numpy()
        rgb = (model_params[2][-n:, 0, :] * C0 + 0.5).detach().cpu().numpy()
        check("new Gaussians are isotropic", np.allclose(scales.max(1), scales.min(1)))
        check("new opacity is 0.1", np.allclose(opacity, 0.1, atol=1e-4), f"{opacity.ravel()[:5]}")
        check("new colours are in [0, 1]", (rgb >= -1e-4).all() and (rgb <= 1 + 1e-4).all())
        check("new f_rest is zero", torch.count_nonzero(model_params[3][-n:]).item() == 0)
        if args.points:
            spacing = point_spacing(np.load(args.points))
            ratio = np.median(scales[:, 0]) / spacing
            check("new scales are around the point spacing", 0.3 < ratio < 5,
                  f"median scale {np.median(scales[:, 0]):.4f} = {ratio:.2f} spacings")

TESTS = {
    "bilinear": test_bilinear,
    "camera": test_camera,
    "visibility": test_visibility,
    "sampling": test_sampling,
    "projection": test_projection,
    "point_error": test_point_error,
    "coverage": test_coverage,
    "optimize": test_optimize,
    "visualize": test_visualize,
    "checkpoint": test_checkpoint,
}

def main():
    parser = argparse.ArgumentParser(description="Tests for gs_pipeline/optimization.py and gaussian_splatting_utils/add_gaussians.py")
    parser.add_argument("test", choices=[*TESTS, "unit"], help="'unit' runs bilinear, camera, visibility, sampling")
    parser.add_argument("--model", help="output/<scene>")
    parser.add_argument("--iteration", type=int, help="selects train/ours_<it> and point_cloud/iteration_<it>")
    parser.add_argument("--metric", default="psnr", choices=["psnr", "ssim", "lpips"])
    parser.add_argument("--stride", type=int, default=20, help="use every n-th LAS point")
    parser.add_argument("--cam", type=int, default=0, help="projection: which camera to overlay")
    parser.add_argument("--n_views", type=int, default=5)
    parser.add_argument("--budget_frac", type=float, default=0.05)
    parser.add_argument("--las", help="optimize: also render the chosen views into data/_test_views")
    parser.add_argument("--checkpoint", help="checkpoint: .pth to inspect")
    parser.add_argument("--ply", help="checkpoint: matching point_cloud.ply")
    parser.add_argument("--added", type=int, help="checkpoint: how many Gaussians add_gaussians.py appended")
    parser.add_argument("--points", help="checkpoint: points.npy, to compare new scales with the spacing")
    args = parser.parse_args()

    names = ["bilinear", "camera", "visibility", "sampling"] if args.test == "unit" else [args.test]
    for name in names:
        print(f"--- {name}")
        TESTS[name](args)
    print(f"\n{'all checks passed' if not failed else f'{len(failed)} failed: ' + ', '.join(failed)}")

if __name__ == "__main__":
    main()
