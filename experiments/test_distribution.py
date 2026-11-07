import torch
from pathlib import Path
from gaussian_splatting_utils.checkpoint_io import robust_extent

# Run from the repo root in the gaussian_splatting env: python -m experiments.test_distribution

def get_path(model, checkpoint): 
    return Path(f"output/{model}/chkpnt{checkpoint}.pth")

models = ["lego_normals", "bezigrad_normals", "ljubljana_1_normals"]
checkpoints = [20_000, 25_000, 30_000]

for model in models:
    print(model)
    for checkpoint in checkpoints:      
        p, _ = torch.load(get_path(model, checkpoint), weights_only=False) 
        s = torch.exp(p[4]).max(dim=1).values
        o = torch.sigmoid(p[6]).squeeze(1)
        
        xyz = p[1]
        ext = robust_extent(xyz.detach())

        print(len(s), (s > 0.01 * ext).sum().item(), (s > 0.03 * ext).sum().item(), s.quantile(0.99).item())
    print("-----------------------------------------------------")