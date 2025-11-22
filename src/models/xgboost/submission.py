import wandb

from ...dataloaders import get_data_loader
from ...experiments import DefaultTracker
from ...experiments.config import RunConfig
from ...submissions.submission import create_submission as create_submission_base
from .config import XGBHyperparamConfig
from .model import EnsembleXGBRegressorModel, XGBRegressorModel


def create_submission(
    season: int,
    run_id: str,
    submission_affix: str = "",
    entity: str = "aicomp-mmlm",
    project: str = "xgboost",
):
    """
    Create a submission file using a trained XGBoost model from a W&B run.

    Args:
        season: Season for which to create the submission (e.g., 2025)
        run_id: W&B run ID
        submission_affix: Optional suffix for the submission filename
        entity: W&B entity name
        project: W&B project name

    Returns:
        Path to the created submission file
    """
    run_path = f"{entity}/{project}/{run_id}"

    run = wandb.Api().run(run_path)
    config = run.config
    print("Config:", config)

    run_config = RunConfig(**config.get("run_config", {}))
    print("Run Config:", run_config)

    hyperparameters = XGBHyperparamConfig(**config.get("xgboost_config", {}))
    print("Hyperparameters:", hyperparameters)

    dataloader = get_data_loader(run_config)

    # Use ensemble model if ensemble dataloader is specified
    is_ensemble = run_config.data_loader == "season_average_ensemble"
    model_cls = EnsembleXGBRegressorModel if is_ensemble else XGBRegressorModel
    model = model_cls(dataloader, hyperparameters, None, DefaultTracker({}))

    model.fit(run_config.valid_season, run_config.start_season)
    submission_path = create_submission_base(
        season=season,
        model=model,
        filename=f"submission_xgboost{'_' + submission_affix if submission_affix else ''}_{season}.csv",
        fit=False,
    )
    print(f"Created submission at: {submission_path}")
    return submission_path
