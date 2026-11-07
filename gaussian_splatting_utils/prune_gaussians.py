import argparse
import torch
from checkpoint_io import robust_extent, save_ply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Which checkpoint to prune.")
    parser.add_argument("--min_opacity", type=float, help="How transparent gaussians should be removed", default=0.001)
    parser.add_argument("--size_factor_min", type=float, help="How small gaussians should be removed", default=0.01)
    parser.add_argument("--size_factor_max", type=float, help="How big gaussians should be removed", default=2)
    parser.add_argument("--out_checkpoint", help="Where to save new checkpoint")
    parser.add_argument("--target_count", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ply")

    args = parser.parse_args()

    model_params, iteration = torch.load(args.checkpoint)
    model_params = list(model_params)

    opacity = torch.sigmoid(model_params[6]).squeeze(1)
    max_scale = torch.exp(model_params[4]).max(dim=1).values
    xyz = model_params[1]
    extent = robust_extent(xyz.detach())

    remove = (opacity < args.min_opacity) | (max_scale * 100 > args.size_factor_max * extent) | (max_scale * 100 < args.size_factor_min * extent)
    keep = ~remove

    n_total = len(keep)
    n_thresh = remove.sum().item()
    n_after_thresh = n_total - n_thresh
    if args.target_count > 0 and keep.sum() > args.target_count:
        idx = keep.nonzero().squeeze(1)
        g = torch.Generator(device=idx.device)
        g.manual_seed(args.seed)
        perm = torch.randperm(len(idx), generator=g, device=idx.device)
        shuffled = idx[perm]
        chosen = shuffled[:args.target_count]
        keep = torch.zeros_like(keep)
        keep[chosen] = True

    n_random = n_after_thresh - keep.sum().item()
    print(f"extent={extent:.4f}  total={n_total}  thresholds removed {n_thresh}  "
        f"randomly removed {n_random}  kept {keep.sum().item()}")

    for i in range(1, 10):
        model_params[i] = model_params[i][keep]

    opt_dict = model_params[10]
    for state in opt_dict["state"].values():
        state["exp_avg"] = state["exp_avg"][keep]
        state["exp_avg_sq"] = state["exp_avg_sq"][keep]

    if args.out_checkpoint:
        torch.save((tuple(model_params), iteration), args.out_checkpoint)

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