import json
import time
from contextlib import contextmanager
from pathlib import Path


class StageTimer:
    # Wall-clock time per pipeline stage, printed to the console and rewritten to a JSON file
    # after every stage, so the timings survive a crash or Ctrl+C.
    def __init__(self, path):
        self.path = Path(path)
        self.start = time.perf_counter()
        self.records = []

    @contextmanager
    def stage(self, name, **info):
        print(f"[time] {name} started", flush=True)
        t0 = time.perf_counter()
        status = "failed"
        try:
            yield
            status = "ok"
        finally:
            seconds = time.perf_counter() - t0
            total = time.perf_counter() - self.start
            self.records.append({"stage": name, **info, "seconds": round(seconds, 2), "status": status})
            print(f"[time] {name}: {seconds:.1f}s ({status}), total {total:.1f}s", flush=True)
            self.save()

    def save(self):
        totals = {}
        for r in self.records:
            totals[r["stage"]] = round(totals.get(r["stage"], 0) + r["seconds"], 2)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "total_seconds": round(time.perf_counter() - self.start, 2),
            "totals_per_stage": totals,
            "stages": self.records,
        }, indent=2))

    def print_summary(self):
        totals = json.loads(self.path.read_text())["totals_per_stage"] if self.path.exists() else {}
        total = time.perf_counter() - self.start
        print(f"\n{'stage':<20}{'seconds':>10}{'share':>8}")
        for name, seconds in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"{name:<20}{seconds:>10.1f}{100 * seconds / max(total, 1e-9):>7.1f}%")
        print(f"{'total':<20}{total:>10.1f}")
