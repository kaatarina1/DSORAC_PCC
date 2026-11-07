from pathlib import Path

FOV = 45
W = 1024
H = 512

REPO_ROOT = Path(__file__).resolve().parents[1]
GS_DIR = REPO_ROOT / "gaussian-splatting"          # the submodule
GS_UTILS = REPO_ROOT / "gaussian_splatting_utils"  # scripts that run in the gaussian_splatting env
GS_ENV = "gaussian_splatting"
