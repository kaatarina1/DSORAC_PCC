import argparse
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from prune_gaussians import robust_extent, save_ply

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to squeeze.")
    parser.add_argument("--out_checkpoint", required=True, help="Where to save the squeezed checkpoint.")
    parser.add_argument("--points", required=True, help="points.npy with the LAS points (same frame as the cameras).")
    parser.add_argument("--ply", help="Where to write the squeezed point_cloud.ply.")
    parser.add_argument("--k", type=int, default=64, help="Neighbours for the local plane fit.")
    parser.add_argument("--max_variation", type=float, default=0.02, help="Surface variation below which a neighbourhood is planar.")
    parser.add_argument("--thickness", type=float, default=0.5, help="Target thickness along the normal, in point spacings.")
    parser.add_argument("--overhang", type=float, default=2.0, help="Allowed distance of the 2-sigma ends from the cloud, in point spacings.")
    parser.add_argument("--shrink_steps", type=int, default=4)
    args = parser.parse_args()

    model_params, iteration = torch.load(args.checkpoint)
    model_params = list(model_params)
    xyz = model_params[1].detach().cpu().numpy()
    scales = torch.exp(model_params[4]).detach().cpu().numpy()
    quats = model_params[5].detach().cpu().numpy()

    points = np.load(args.points)
    tree = cKDTree(points)
    spacing = np.median(tree.query(points[::100], k=2)[0][:, 1])
    t = args.thickness * spacing
    tau = args.overhang * spacing

    extent = robust_extent(model_params[1].detach())
    idx = np.nonzero(scales.max(1) * 100 > 0 * extent)[0]

    _, nn = tree.query(xyz[idx], k=args.k, workers=-1)
    nb = points[nn]
    centroid = nb.mean(1)
    d = nb - centroid[:, None]
    C = np.einsum("mki,mkj->mij", d, d) / args.k
    evals, evecs = np.linalg.eigh(C)
    normal = evecs[:, :, 0]
    variation = evals[:, 0] / evals.sum(1).clip(1e-12)
    planar = variation < args.max_variation

    R = Rotation.from_quat(np.roll(quats[idx], -1, axis=1)).as_matrix()   # (w,x,y,z) -> (x,y,z,w)
    Sigma = R @ (scales[idx, :, None] ** 2 * R.transpose(0, 2, 1))         # R diag(s²) Rᵀ
    nnT = normal[:, :, None] * normal[:, None, :]
    P = np.eye(3) - nnT
    Sigma_new = P @ Sigma @ P + t ** 2 * nnT
    ev, R_new = np.linalg.eigh(Sigma_new)
    s_new = np.sqrt(np.clip(ev, 1e-12, None))
    R_new[np.linalg.det(R_new) < 0, :, 0] *= -1
    mu_new = xyz[idx] - np.einsum("mi,mi->m", xyz[idx] - centroid, normal)[:, None] * normal

    for _ in range(args.shrink_steps):                      # e.g. 4
        for j in range(3):
            axis = R_new[:, :, j] * (2 * s_new[:, j:j + 1])
            d_plus = tree.query(mu_new + axis, workers=-1)[0]
            d_minus = tree.query(mu_new - axis, workers=-1)[0]
            s_new[np.maximum(d_plus, d_minus) > tau, j] *= 0.7

    sel = idx[planar]
    dev = model_params[1].device
    rows = torch.from_numpy(sel).to(dev)

    def put(i, values):
        p = model_params[i].detach().clone()
        p[rows] = torch.from_numpy(np.ascontiguousarray(values)).float().to(dev)
        model_params[i] = torch.nn.Parameter(p.requires_grad_(True))

    put(1, mu_new[planar])
    put(4, np.log(s_new[planar]))
    put(5, np.roll(Rotation.from_matrix(R_new[planar]).as_quat(), 1, axis=1))

    state = model_params[10]["state"]
    for group in (0, 4, 5):                       # xyz, scaling, rotation
        state[group]["exp_avg"][rows] = 0
        state[group]["exp_avg_sq"][rows] = 0

    torch.save((tuple(model_params), iteration), args.out_checkpoint)
    print(f"squeezed {len(sel)} of {len(idx)} big Gaussians ({len(idx) - len(sel)} non-planar left as they are)")

    if args.ply:
        save_ply(args.ply,
                 model_params[1],   # xyz
                 model_params[2],   # features_dc
                 model_params[3],   # features_rest
                 model_params[6],   # opacity (logit)
                 model_params[4],   # scaling (log)
                 model_params[5])   # rotation
        print(f"wrote {args.ply}")



if __name__ == "__main__":
    main()
