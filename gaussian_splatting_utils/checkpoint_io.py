# Shared by prune_gaussians.py, add_gaussians.py, experiments/squeeze_gaussians.py and
# experiments/test_distribution.py. Runs in the gaussian_splatting env (needs torch).
import torch


def robust_extent(xyz, lo_q=0.02, hi_q=0.98):
    n = xyz.shape[0]
    sorted_xyz = xyz.sort(dim=0).values
    lo = sorted_xyz[int(lo_q * (n - 1))]
    hi = sorted_xyz[int(round(hi_q * (n - 1)))]
    return torch.linalg.norm(hi - lo).item()


def save_ply(path, xyz, f_dc, f_rest, opacity_raw, scaling_raw, rotation):
    f_dc = f_dc.transpose(1, 2).flatten(start_dim=1)
    f_rest = f_rest.transpose(1, 2).flatten(start_dim=1)

    cols = [xyz, torch.zeros_like(xyz), f_dc, f_rest, opacity_raw, scaling_raw, rotation]
    data = torch.cat(cols, dim=1).detach().float().cpu().numpy().astype("<f4")

    names = (["x", "y", "z", "nx", "ny", "nz"]
             + [f"f_dc_{i}" for i in range(f_dc.shape[1])]
             + [f"f_rest_{i}" for i in range(f_rest.shape[1])]
             + ["opacity"]
             + [f"scale_{i}" for i in range(scaling_raw.shape[1])]
             + [f"rot_{i}" for i in range(rotation.shape[1])])
    assert data.shape[1] == len(names)

    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {data.shape[0]}\n"
              + "".join(f"property float {n}\n" for n in names)
              + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
