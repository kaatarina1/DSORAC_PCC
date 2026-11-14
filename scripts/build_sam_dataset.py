"""Build the SAM-masked base dataset of one point cloud once, so several parallel runs can copy it.

python scripts/build_sam_dataset.py D:/Katarina/pc/ljubljana_1_normals.las D:/Katarina/ablation_nokeep/sam_build

Writes <out_root>/<las stem>_sam with the same images and dataset_params.json marker that
pipeline.py --layer_masks sam would create, so a copy of it in <work_root>/ is reused, not regenerated.
Needs an activated conda env (CONDA_EXE for the sam2 env) and deno, like pipeline.py.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gs_pipeline.dataset import read_point_cloud, prepare_dataset, half_fov_horizontal   # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("las_file")
    parser.add_argument("out_root")
    args = parser.parse_args()

    name = Path(args.las_file).stem
    base_dir = Path(args.out_root, f"{name}_sam")
    work_dir = Path(args.out_root, f"{name}_sam_work")
    xyz, rgb, bb_min, bb_max, _ = read_point_cloud(args.las_file)
    # sampling_rate only thins sparse/0/points3D.txt, which every run overwrites with its own value
    prepare_dataset(args.las_file, str(base_dir), str(work_dir), half_fov_horizontal(),
                    xyz, rgb, bb_min, bb_max, sampling_rate=13, layer_masks="sam")
    shutil.rmtree(work_dir, ignore_errors=True)   # prepare_dataset also makes a run copy; not needed here
    print(f"SAM dataset ready: {base_dir}")


if __name__ == "__main__":
    main()
