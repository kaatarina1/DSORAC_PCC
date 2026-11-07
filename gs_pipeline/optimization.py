import heapq
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F

_t0 = time.perf_counter()

def log(msg):
    print(f"[optimize {time.perf_counter() - _t0:7.1f}s] {msg}", flush=True)

def uncovered_mask(P, gauss_xyz, spacing, cover_k=2.0):
    d, _ = cKDTree(gauss_xyz).query(P, workers=-1)
    return d > cover_k * spacing

def visible_weights(P, normals, K, R, t, width, height, splat=1, eps_rel=0.01):
    # P, normals: float32 tensors (N, 3) on the GPU; idx, w, uv are returned on the same device
    dev = P.device
    R_t = torch.as_tensor(R, dtype=torch.float32, device=dev)
    t_t = torch.as_tensor(t, dtype=torch.float32, device=dev)
    focal = torch.tensor([K[0, 0], K[1, 1]], dtype=torch.float32, device=dev)
    principal = torch.tensor([K[0, 2], K[1, 2]], dtype=torch.float32, device=dev)

    Pc = P @ R_t.T + t_t
    z = Pc[:, 2]
    uv = Pc[:, :2] / z[:, None] * focal + principal
    valid = (z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) \
                       & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
    idx = valid.nonzero(as_tuple=True)[0]
    uv, z = uv[idx], z[idx]

    ui, vi = uv.round().long().unbind(1)
    pix = vi * width + ui
    depth = torch.full((height * width,), float("inf"), device=dev)
    depth.scatter_reduce_(0, pix, z, reduce="amin")                     # nearest z per pixel
    depth = -F.max_pool2d(-depth.view(1, 1, height, width),
                          kernel_size=2 * splat + 1, stride=1, padding=splat)   # = 3×3 splat
    depth = depth.view(-1)

    keep = z <= depth[pix] * (1 + eps_rel)
    idx, uv = idx[keep], uv[keep]
    w = torch.ones(len(idx), device=dev)
    if normals is not None:
        cam_center = -R_t.T @ t_t
        d = cam_center - P[idx]
        w = (normals[idx] * d).sum(1).abs() / d.norm(dim=1)
    return idx, w, uv

def to_device(a, device="cuda"):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)

def project(P, K, R, t, width, height):
    Pc = P @ R.T + t
    z = Pc[:, 2]
    uv = (Pc @ K.T)[:, :2] / z[:, None]
    valid = ((z > 1e-6) #in front of the camera
            & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
            & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1))
    return uv, z, valid

def visibility(uv, z, valid, width, height, splat=1, eps_rel=0.01):
    depth = np.full((height, width), np.inf)
    idx = np.where(valid)[0]
    ui, vi = np.round(uv[idx]).astype(int).T
    for du in range(-splat, splat+1):
        for dv in range(-splat, splat+1):
            np.minimum.at(depth, (np.clip(vi+dv, 0, height - 1), np.clip(ui+du, 0, width-1)), z[idx])
    vis = np.zeros(len(z), bool)
    vis[idx] = z[idx] <= depth[vi, ui] * (1 + eps_rel)
    return vis

def compute_point_error(P, normals, cameras, method_dir, metric, device="cuda"):
    # NumPy in and out; points are uploaded once and E, C are summed on the GPU
    P_t = to_device(P, device)
    N_t = None if normals is None else to_device(normals, device)
    E = torch.zeros(len(P), dtype=torch.float64, device=device)
    C = torch.zeros(len(P), dtype=torch.float64, device=device)
    t0 = time.perf_counter()
    for i, (K, R, t, w, h) in enumerate(cameras):
        err_map = np.load(Path(method_dir, metric, f"{i:05d}.npy"))
        assert err_map.shape == (h, w), f"map {i} is {err_map.shape}, camera is {(h, w)}"
        accumulate_error(P_t, N_t, to_device(err_map, device), K, R, t, E, C)
        if (i + 1) % 10 == 0 or i + 1 == len(cameras):
            per_cam = (time.perf_counter() - t0) / (i + 1)
            log(f"error maps {i + 1}/{len(cameras)}  ({per_cam:.2f}s per camera, ~{per_cam * (len(cameras) - i - 1):.0f}s left)")
    E, C = E.cpu().numpy(), C.cpu().numpy()
    E_mean = np.zeros_like(E)
    seen = C > 0
    E_mean[seen] = E[seen] / C[seen]
    return E_mean, C

def bilinear(img, uv): 
    x, y = uv[:, 0], uv[:, 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, img.shape[1] - 1), np.minimum(y0 + 1, img.shape[0] - 1)
    ax, ay = x - x0, y - y0
    e = ((1 - ax) * (1- ay) * img[y0, x0] +
        ax * (1 - ay) * img[y0, x1] +
        (1 - ax) * ay * img[y1, x0] +
        ax * ay * img[y1, x1])
    return e

def bilinear_t(img, uv):
    # Same as bilinear, for tensors on the GPU
    x, y = uv[:, 0], uv[:, 1]
    x0, y0 = x.floor().long(), y.floor().long()
    x1 = (x0 + 1).clamp(max=img.shape[1] - 1)
    y1 = (y0 + 1).clamp(max=img.shape[0] - 1)
    ax, ay = x - x0, y - y0
    return ((1 - ax) * (1 - ay) * img[y0, x0] + ax * (1 - ay) * img[y0, x1]
            + (1 - ax) * ay * img[y1, x0] + ax * ay * img[y1, x1])

def accumulate_error(P, normals, err_map, K, R, t, E, C):
    idx, w, uv = visible_weights(P, normals, K, R, t, err_map.shape[1], err_map.shape[0])
    e = bilinear_t(err_map, uv)
    E.index_add_(0, idx, (w * e).double())
    C.index_add_(0, idx, w.double())
    return E, C, w

def load_cameras(path):
    cams = sorted(json.loads(Path(path).read_text()), key=lambda c: c["id"])
    out = []
    for c in cams:
        R = np.array(c["rotation"]).T
        t = -R @ np.array(c["position"])
        K = np.array([[c["fx"], 0, c["width"] / 2],
                      [0, c["fy"], c["height"] / 2],
                      [0, 0, 1]])
        out.append((K, R, t, c["width"], c["height"]))
    return out

def cluster_normal(Q, Nq, Eq, c):
    if Nq is not None:
        S = (Nq * Eq[:, None]).T @ Nq
        return np.linalg.eigh(S)[1][:, -1]
    d = Q - c
    Cov = (d * Eq[:, None]).T @ d
    return np.linalg.eigh(Cov)[1][:, 0]
    
def orient_towards_cameras(n, c, cameras):
    votes = 0.0
    for K, R, t, w, h in cameras:
        _, _, valid = project(c[None], K, R, t, w, h)
        if valid[0]:
            votes += np.sign(np.dot(n, -R.T @ t - c))
    return n if votes >= 0 else -n

def look_at(cam_pos, target, up=np.array([0, 1, 0])):
    z = target - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [1, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ cam_pos

def candidate_views(P, normals, E, K, width, height, cameras, tree, pct=90, eps=0.05, 
                    min_samples=20, d_minmax=(5.0, 10.0), margin=2.0, k = 5):
    hi = np.where(E > np.percentile(E, pct))[0]
    log(f"DBSCAN on {len(hi)} high-error points (eps {eps:.4f})")
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(P[hi])
    log(f"{len(set(labels) - {-1})} clusters, {(labels == -1).sum()} noise points")
    fov = 2 * np.arctan(min(width / (2 * K[0, 0]), height / (2 * K[1, 1])))
    ids, inv = np.unique(labels[labels >= 0], return_inverse=True)
    mass = np.bincount(inv, weights=E[hi][labels >= 0])
    max_cands = k * 4
    top = ids[np.argsort(-mass)[:max_cands]]
    cands = []
    for l in top:
        members = hi[labels == l]
        Q, Eq = P[members], E[members]
        Nq = None if normals is None else normals[members]
        c = (Q * Eq[:, None]).sum(0) / Eq.sum()
        n = orient_towards_cameras(cluster_normal(Q, Nq, Eq, c), c, cameras)
        r = np.linalg.norm(Q - c, axis=1).max()
        d = np.clip(margin * r / np.tan(fov / 2), *d_minmax)
        pos = c + d * n
        if tree.query(pos)[0] < 0.2 * d:
            continue
        R, t = look_at(pos, c)
        cands.append((pos, c, R, t))
    log(f"{len(cands)} candidate views after the inside-geometry check")
    return cands

def greedy_select(P, normals, E, K, width, height, cands, k=50):
    P_t = to_device(P)
    N_t = None if normals is None else to_device(normals)
    cover = []
    for j, (_, _, R, t) in enumerate(cands):
        idx, w, _ = visible_weights(P_t, N_t, K, R, t, width, height)
        cover.append((idx.cpu().numpy().astype(np.int32), w.cpu().numpy()))
        if (j + 1) % 5 == 0 or j + 1 == len(cands):
            log(f"view coverage {j + 1}/{len(cands)}")

    resid = E.astype(np.float64).copy()
    heap = [(-float((w * resid[idx]).sum()), j) for j, (idx, w) in enumerate(cover)]
    heapq.heapify(heap)

    chosen = []
    while heap and len(chosen) < k:
        _, j = heapq.heappop(heap)
        idx, w = cover[j]
        gain = float((w * resid[idx]).sum())          # fresh gain against the current resid
        if heap and gain < -heap[0][0]:               # someone else might be better → push back
            heapq.heappush(heap, (-gain, j))
            continue
        if gain <= 0:
            break
        chosen.append(cands[j])
        resid[idx] *= 1 - w
        log(f"chose view {len(chosen)}: candidate {j}, covers {gain:.4g} of remaining error")
    return chosen

def read_ply_xyz(path):
    with open(path, "rb") as f:
        names, n = [], 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                names.append(line.split()[-1])
            elif line == "end_header":
                break
        data = np.fromfile(f, dtype="<f4", count=n * len(names)).reshape(n, len(names))
    return data[:, [names.index("x"), names.index("y"), names.index("z")]]

def importance_sample(E, n, alpha=0.02, gamma=2.0, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    M = len(E)
    if n >= M:
        return np.arange(M)
    e = np.maximum(E, 0) ** gamma
    p = alpha / M + (1 - alpha) * e / max(e.sum(), 1e-12)
    keys = np.log(rng.random(M)) / p
    return np.argpartition(-keys, n)[:n]


def optimize(model_path, method_dir, ply_path, metric, budget_frac=0.05, n_views=5,
             alpha=0.01, cover_k=2.0, stride=1, radius_ratio=0.25, seed=0):
    global _t0
    _t0 = time.perf_counter()

    P_all = np.load(Path(model_path, "points.npy")).astype(np.float32)
    normals_path = Path(model_path, "normals.npy")
    N_all = np.load(normals_path).astype(np.float32) if normals_path.exists() else None

    sub = np.arange(0, len(P_all), stride)
    P = P_all[sub]
    normals = None if N_all is None else N_all[sub]

    log(f"loaded {len(P_all)} points, using {len(P)} (stride {stride}), normals: {N_all is not None}")

    tree = cKDTree(P_all)
    spacing = np.median(tree.query(P_all[::100], k=2)[0][:, 1])
    log(f"point spacing {spacing:.4f}")
    cameras = load_cameras(Path(model_path, "cameras.json"))
    log(f"{len(cameras)} cameras")

    E, C = compute_point_error(P, normals, cameras, method_dir, metric)
    seen = np.where(C > 0)[0]
    log(f"error done: {len(seen)} of {len(P)} points seen")

    gauss_xyz = read_ply_xyz(ply_path)
    budget = int(budget_frac * len(gauss_xyz))
    log(f"{len(gauss_xyz)} Gaussians, budget {budget}")
    cand = seen[uncovered_mask(P[seen], gauss_xyz, spacing, cover_k)]
    log(f"{len(cand)} uncovered candidates")
    rng = np.random.default_rng(seed)
    picked = sub[cand[importance_sample(E[cand], budget, alpha, rng=rng)]]
    log(f"picked {len(picked)} points")

    K, _, _, width, height = cameras[0]
    Ns = None if normals is None else normals[seen]
    cands = candidate_views(P[seen], Ns, E[seen], K, width, height, cameras, tree, eps=5*spacing*stride, k=n_views)
    chosen = greedy_select(P[seen], Ns, E[seen], K, width, height, cands, k=n_views)
    new_poses = [(pos, c, radius_ratio) for pos, c, _, _ in chosen]

    print(f"spacing={spacing:.4f} seen={len(seen)} uncovered={len(cand)} "
          f"picked={len(picked)} clusters={len(cands)} views={len(new_poses)}")
    return new_poses, picked