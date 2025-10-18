from typing import Any

import wandb

from ..run import RUN_TARGET


def wandb_sweep(
    config: dict[str, Any],
    target: RUN_TARGET,
    run_count=None,
    entity: str = "aicomp-mmlm",
    project: str = "aicomp-mmlm",
):
    sweep_id = wandb.sweep(
        sweep=config,
        entity=entity,
        project=project,
    )
    wandb.agent(sweep_id=sweep_id, function=target, count=run_count)
    if run_count:
        wandb.api.stop_sweep(sweep_id)
