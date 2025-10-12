from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATASETS_DIR = BASE_DIR / "datasets"
WEIGHTS_DIR = BASE_DIR / "weights"
CONFIGS_DIR = BASE_DIR / "configs"