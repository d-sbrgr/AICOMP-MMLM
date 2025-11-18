import wandb

from ...dataloaders.base_dataloader import BaseDataloader
from ...dataloaders.ensemble.ensemble_season_avg_dataloader import EnsembleSeasonAverageDataLoader
from ...dataloaders.simple.season_avg_dataloader import SeasonAverageDataLoader
from ...dataloaders.simple.sliding_window_avg_dataloader import SlidingWindowAvgDataLoader
from ...dataloaders.simple.weighted_season_avg_dataloader import WeightedSeasonAvgDataLoader
from ...experiments.config import ExperimentConfig, RunConfig
from ...experiments.wandb_tracker import WandbTracker
from ...submissions.submission import create_submission as create_submission_base
from .config import NeuralNetworkHyperparamConfig
from .model import NeuralNetworkModel


def create_submission(
    season: int,
    artifact_name: str,
    submission_affix: str = "",
    entity: str = "aicomp-mmlm",
    project: str = "neural-network",
):
    """Create a submission file using a trained neural network model from a W&B artifact."""

    run_name, artifact_version = artifact_name.split(":")
    artifact_path = f"{entity}/{project}/model-{run_name}:{artifact_version}"
    run_path = f"{entity}/{project}/{run_name}"

    run = wandb.Api().run(run_path)
    config = run.config
    print("Config:", config)

    run_config = RunConfig(**config.get("run_config", {}))
    print("Run Config:", run_config)

    hyperparameters = NeuralNetworkHyperparamConfig(**config.get("neural_network_config", {}))
    print("Hyperparameters:", hyperparameters)

    dataloader = _get_data_loader(run_config)
    model = NeuralNetworkModel(
        dataloader,
        hyperparameters,
        None,
        WandbTracker(ExperimentConfig(project=project, name="neural-network-standard"), run=run),
    )
    model.prepare_for_evaluation(
        artifact_path, valid_season=run_config.valid_season, start_season=run_config.start_season
    )

    submission_path = create_submission_base(
        season=season,
        model=model,
        filename=f"submission_neural_network{'_' + submission_affix if submission_affix else ''}_{season}.csv",
        fit=False,
    )
    print(f"Created submission at: {submission_path}")


def _get_data_loader(run_config: RunConfig) -> BaseDataloader:
    """Get the appropriate data loader based on run configuration."""
    data_loader_map = {
        "season_average": SeasonAverageDataLoader,
        "season_average_ensemble": EnsembleSeasonAverageDataLoader,
        "weighted_season_average": WeightedSeasonAvgDataLoader,
        "sliding_window_average": SlidingWindowAvgDataLoader,
    }

    loader_cls = data_loader_map.get(run_config.data_loader, SeasonAverageDataLoader)

    data_loader_config = run_config.data_loader_config
    if data_loader_config is None:
        return loader_cls(run_config.num_features)
    return loader_cls(run_config.num_features, **data_loader_config)
