"""Run pipeline.py for one scene of each type.

python run_scenes.py
python run_scenes.py --only lego bezigrad --skip_generating

Each scene runs in its own process, so a failure in one scene does not stop the others.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from render_eval import compressionRatio, latestGaussianPly

# One scene per type, each has a ground truth folder data/<name>_eval used by pipeline.py
SCENES = {
    #"lego":     "data/pc/lego_normals.las",         # synthetic object without capture noise
    "bezigrad": "data/pc/bezigrad_normals.las",     # aerial laser scan (terrain, buildings, vegetation)
    "ljubljana": "data/pc/ljubljana_1_normals.las", # terrestrial scan from a moving vehicle
}


def read_scene_results(las_file, start):
    # Same model folder as pipeline.py: output/<las stem>
    model_path = Path("output", Path(las_file).stem)
    ratio = compressionRatio(las_file, model_path)

    # Mean over the ground truth views, written by gaussian-splatting/utils/eval_views.py.
    # Older files are from a previous run (e.g. evaluation was skipped this time), so they are ignored.
    metrics_path = model_path / "eval" / latestGaussianPly(model_path).parent.name / "metrics_views.json"
    mean = None
    if metrics_path.exists() and metrics_path.stat().st_mtime >= start:
        mean = json.loads(metrics_path.read_text())["mean"]
    return ratio, mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=SCENES.keys(), help="run only these scenes")
    parser.add_argument("--skip_generating", action="store_true", help="reuse existing datasets in data/<name>")
    args = parser.parse_args()

    scene_names = args.only or list(SCENES)
    results = {}
    for name in scene_names:
        cmd = [sys.executable, "pipeline.py", SCENES[name]] #, "--skip_generating", "--skip_training"
        if args.skip_generating:
            cmd.append("--skip_generating")

        print(f"\n===== {name}: {' '.join(cmd)}", flush=True)
        start = time.perf_counter()
        code = subprocess.run(cmd).returncode
        results[name] = (code, time.perf_counter() - start)

    print("\n===== Summary")
    for name, (code, seconds) in results.items():
        status = "ok" if code == 0 else f"FAILED (exit {code})"
        print(f"{name:<12}{status:<20}{seconds / 60:.1f} min")

    if any(code != 0 for code, _ in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
