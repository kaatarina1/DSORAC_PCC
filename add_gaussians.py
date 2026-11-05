import argparse
import numpy as np
import torch
from scipy.spatial import cKDTree
from prune_gaussians import save_ply

C0 = 0.28209479177387814   # SH band-0 constant, same as utils/sh_utils.RGB2S

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_checkpoint", required=True)
    parser.add_argument("--ply", help="Where to write the new point_cloud.ply.")
    parser.add_argument("--points", required=True, help="points.npy (same frame as the cameras).")
    parser.add_argument("--colors", required=True, help="colors.npy, RGB in [0, 1], same order as points.npy.")
    parser.add_argument("--indices", required=True, help="Rows of points.npy to add.")
    parser.add_argument("--init_opacity", type=float, default=0.1)

    args = parser.parse_args()

    model_params, iteration = torch.load(args.checkpoint)
    model_params = list(model_params)
    dev = model_params[1].device

    idx = np.load(args.indices)
    n = len(idx)
    if n == 0:
        print("nothing to add")
        return
    
    new_xyz = np.load(args.points)[idx].astype(np.float32)
    new_rgb = np.load(args.colors)[idx].astype(np.float32)

    old_xyz = model_params[1].detach().cpu().numpy()
    tree = cKDTree(np.vstack([old_xyz, new_xyz]))
    d, _ = tree.query(new_xyz, k=4, workers=-1)
    dist2 = np.clip((d[:, 1:] ** 2).mean(1), 1e-7, None)
    scales = np.repeat(np.log(np.sqrt(dist2))[:, None], 3, axis=1)

    f = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(dev)
    p = args.init_opacity
    new = {
        1: f(new_xyz),                                               # xyz           (n, 3)
        2: f((new_rgb - 0.5) / C0)[:, None, :],                      # features_dc   (n, 1, 3)
        3: torch.zeros((n, *model_params[3].shape[1:]), device=dev), # features_rest (n, 15, 3)
        4: f(scales),                                                # scaling (log) (n, 3)
        5: torch.tensor([1.0, 0, 0, 0], device=dev).repeat(n, 1),    # rotation      (n, 4)
        6: torch.full((n, 1), float(np.log(p / (1 - p))), device=dev), # opacity (logit) (n, 1)
    }
    for i, v in new.items():
        cat = torch.cat([model_params[i].detach(), v], dim=0)
        model_params[i] = torch.nn.Parameter(cat.requires_grad_(True))

    for i in (7, 8, 9):   # max_radii2D (N,), xyz_gradient_accum (N,1), denom (N,1)
        old = model_params[i]
        model_params[i] = torch.cat([old, torch.zeros((n, *old.shape[1:]), dtype=old.dtype, device=old.device)])

    state = model_params[10]["state"]
    for g in state.values():   # groups: 0 xyz, 1 f_dc, 2 f_rest, 3 opacity, 4 scaling, 5 rotation
        for key in ("exp_avg", "exp_avg_sq"):
            s = g[key]
            g[key] = torch.cat([s, torch.zeros((n, *s.shape[1:]), dtype=s.dtype, device=s.device)])

    torch.save((tuple(model_params), iteration), args.out_checkpoint)
    print(f"added {n} Gaussians, total {model_params[1].shape[0]}")

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