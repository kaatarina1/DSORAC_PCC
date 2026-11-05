import sys, json
from pathlib import Path
import numpy as np
import torch
import lpips
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity

METRICS = {
    "psnr": 0.05,
    "ssim": 1.0,
    "lpips": 1.0,
}

def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255

def save(method_dir, save_png, metric, name, arr):
    np.save(method_dir / metric / f"{Path(name).stem}.npy", arr)
    if (save_png):
        plt.imsave(method_dir / metric / name, arr, cmap="inferno", vmin=0, vmax=METRICS[metric])

@torch.no_grad()
def main(method_dir, metric, save_maps=True, save_png=True):
    method_dir = Path(method_dir)
    print(f"Method directory {method_dir}")

    if save_maps:
        (method_dir / metric).mkdir(exist_ok=True)

    if metric is None or metric == "lpips":
        lp = lpips.LPIPS(net="vgg", spatial=True).cuda()
    
    res = {}
    for f in sorted((method_dir / "renders").iterdir()):
        r, g = load(f), load(method_dir / "gt" / f.name)
        se = ssim_map = lp_map = None

        if metric is None or metric == "psnr":
            print(f"Calculating PSNR")
            se = ((r - g) ** 2).mean(-1)
        
        if metric is None or metric == "ssim":
            print(f"Calculating SSIM")
            ssim, ssim_map = structural_similarity(r, g, data_range=1.0, channel_axis=-1, 
                                                gaussian_weights=True, sigma=1.5, 
                                                use_sample_covariance=False, full=True)
                
            ssim_map = ssim_map.mean(-1)

    
        if metric is None or metric == "lpips":
            t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].cuda() * 2 - 1
            print(f"Calculating LPIPS")
            lp_map = lp(t(r), t(g))[0, 0].cpu().numpy()

        res[f.stem] = {
            "psnr": float(se.mean()) if se is not None else None,
            "ssim": float(1 - ssim_map.mean()) if ssim_map is not None else None,
            "lpips": float(lp_map.mean()) if lp_map is not None else None,
        }

        if save_maps:
            print("Saving...")
            if metric == "psnr":
                save(method_dir, save_png, "psnr", f.name, se)
            elif metric == "ssim":
                save(method_dir, save_png, "ssim", f.name, 1 - ssim_map)
            elif metric == "lpips":
                save(method_dir, save_png, "lpips", f.name, lp_map)

    (method_dir / "scores.json").write_text(json.dumps(res, indent=2))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], save_maps="--no_maps" not in sys.argv, save_png="--no_png" not in sys.argv)