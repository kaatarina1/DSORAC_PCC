import json
from pathlib import Path

import numpy as np

from gs_pipeline.gs_runner import measure_views, render_gaussians, evaluate_gaussians


def calculate_quality(scores, metric, names):
    v = np.array([scores[n][metric] for n in names])
    if metric == "psnr":
        per_view = -10 * np.log10(v + 1e-10)
    elif metric == "ssim":
        per_view = 1 - v
    else:
        per_view = -v
    return float(per_view.mean()), float(np.percentile(per_view, 10))


def measure(model_path, checkpoint, data_dir, metric, names=None, save_maps=False):
    method_dir = Path(model_path, "train", f"ours_{checkpoint}")
    if metric == "psnr":
        measure_views(model_path, checkpoint, data_dir, save_maps=save_maps)
    else:   # TODO: ssim / lpips are not ported to the GPU path yet
        render_gaussians(model_path, checkpoint, data_dir)
        evaluate_gaussians(method_dir, metric, save_maps=save_maps)
    scores = json.loads((method_dir / "scores.json").read_text())
    names = names if names is not None else sorted(scores)
    q, p10 = calculate_quality(scores, metric, names)
    return q, p10, names, method_dir


def stalled(history):
    phase = history[-1]["phase"]
    steps = [(a, b) for a, b in zip(history, history[1:]) if a["phase"] == phase]
    if len(steps) < 3:
        return False
    recent = steps[-3:]
    if phase == "compress":
        return all((b["r"] - a["r"]) / a["r"] < 0.02 for a, b in recent)
    return all((b["q"] - a["q"]) < 0.1 for a, b in recent)


def show_quality(q, metric):
    return -q if metric == "lpips" else q


def print_round(h, metric):
    print(f"[loop {h['loop']}] {h['phase']:<8} N={h['N']}  r={h['r']:.3f}  "
          f"{metric}={show_quality(h['q'], metric):.4f}  p10={show_quality(h['p10'], metric):.4f}", flush=True)


def print_summary(history, reason, r_star, r_low, metric, path):
    print(f"\n{'krog':>4} {'faza':<9}{'N':>10}{'r':>8}{metric:>9}{'p10':>9}")
    for h in history:
        print(f"{h['loop']:>4} {h['phase']:<9}{h['N']:>10}{h['r']:>8.3f}"
              f"{show_quality(h['q'], metric):>9.4f}{show_quality(h['p10'], metric):>9.4f}")
    last = history[-1]
    ok = lambda b: "OK" if b else "NI DOSEŽENO"
    print(f"\nKonec: {reason}")
    print(f"  r = {last['r']:.3f}   cilj {r_star:.3f}, meja {r_low:.3f}   {ok(last['r'] >= r_low)}")
    path.write_text(json.dumps({"reason": reason, "r_star": r_star, "r_low": r_low,
                                "metric": metric, "history": history}, indent=2))
