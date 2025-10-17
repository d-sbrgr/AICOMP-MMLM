import time

from .config import XGBConfig


def get_run_name(cfg: XGBConfig, features: list[str], season: int) -> str:
    """Return a meaningful run name for wandb experiment tracking."""
    items = {"features": len(features), "season": season}
    items.update(cfg.run_name())
    return f"xgb_baseline_{'_'.join(f'{k}-{v}' for k, v in sorted(items.items()))}_{time.strftime('%y%m%d-%H%M%S')}"
