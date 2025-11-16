from collections.abc import Iterable

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ...dataloaders.base_dataloader import BaseDataloader
from ...experiments import Tracker, WandbTracker
from ...utils.constants import Columns
from ..cross_validation.cv_config import CrossValidationConfig
from ..model import SupervisedModel
from .config import NeuralNetworkHyperparamConfig
from .dnn import DNN


class NeuralNetworkModel(SupervisedModel):
    """
    A wrapper class to train and validate neural network models using PyTorch Lightning.

    This model provides a flexible interface for creating various neural network architectures
    through the configuration system, while integrating seamlessly with the existing model framework.
    """

    def __init__(
        self,
        data: BaseDataloader,
        params: NeuralNetworkHyperparamConfig,
        cv: CrossValidationConfig | None,
        tracker: Tracker,
    ) -> None:
        super().__init__(data, params, cv, tracker, StandardScaler())
        self.params: NeuralNetworkHyperparamConfig = params
        self.model: DNN | None = None
        self.trainer: L.Trainer | None = None

        assert isinstance(tracker, WandbTracker), "NeuralNetworkModel only supports WandbTracker for logging."
        wandb_tracker: WandbTracker = tracker

        self.logger = WandbLogger(name=wandb_tracker.run.name, experiment=wandb_tracker.run, log_model=True)

        if params.seed is not None:
            L.seed_everything(params.seed, workers=True)

    def fit(self, season: int, start_season: int = 2003) -> None:
        """
        Fit the neural network model on data provided by the given dataloader.

        The model uses data up to but not including the specified season.

        Args:
            season (int): Season for which the model should be validated.
            start_season (int): Start season for which the model should be trained on.
        """
        self.data.setup()
        X_train, y_train = self.data.train_data(season, start_season)
        X_train = self._drop_and_sort_features(X_train)

        self._fit_scaler(X_train)
        X_scaled = self._transform(X_train)

        X_valid, y_valid = self.data.valid_data()
        X_valid = self._transform(X_valid)

        X_train_tensor = torch.FloatTensor(X_scaled.values)
        y_train_tensor = torch.FloatTensor(y_train.values)
        X_valid_tensor = torch.FloatTensor(X_valid.values)
        y_valid_tensor = torch.FloatTensor(y_valid.values)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers,
        )

        # Initialize model
        input_dim = X_scaled.shape[1]
        self.model = DNN(input_dim=input_dim, config=self.params)

        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                monitor="val_loss",
                dirpath="checkpoints",
                filename="nn-{epoch:02d}-{val_loss:.4f}",
                save_top_k=3,
                mode="min",
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="epoch", log_momentum=True, log_weight_decay=True),
        ]

        if self.params.early_stopping:
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=self.params.early_stopping_min_delta,
                patience=self.params.early_stopping_patience,
                verbose=False,
                mode="min",
            )
            callbacks.append(early_stop_callback)

        # Setup trainer
        self.trainer = L.Trainer(
            max_epochs=self.params.max_epochs,
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=True,
            logger=self.logger,
            gradient_clip_val=self.params.gradient_clip_val,
            accumulate_grad_batches=self.params.accumulate_grad_batches,
            deterministic=self.params.seed is not None,
            log_every_n_steps=1,
        )

        # Train the model
        self.trainer.fit(self.model, train_loader, val_loader)

    def validate(self) -> None:
        # Nothing to do here, as validation is handled during training with PyTorch Lightning
        pass

    def predict(self, matchups: pd.DataFrame) -> Iterable[float]:
        """
        Make predictions for the given matchups.

        Args:
            matchups (pd.DataFrame): Matchups to make predictions for.

        Note:
            Fit must be called first.
        """
        assert self.model is not None, "Call fit() before predict()"

        X = self.data.test_data(matchups)
        X = self._drop_and_sort_features(X)
        X_scaled = self._transform(X)

        X_tensor = torch.FloatTensor(X_scaled.values)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).numpy()

        return pd.Series(preds, index=X.index)

    def prepare_for_evaluation(self, artifact_path: str, valid_season: int, start_season: int) -> None:
        import wandb

        """Prepare the model for evaluation by loading from a W&B artifact and fitting the scaler.

        Both the validation and start season are only used for the fitting of the scaler.
        Args:
            artifact_path (str): W&B artifact path to load the model from.
            valid_season (int): Season used for validation.
            start_season (int): Start season for which the scaler should be fitted on training data.
        """

        art = wandb.Api().artifact(artifact_path, type="model")
        artifact_dir = art.download()

        self.model = DNN.load_from_checkpoint(artifact_dir + "/model.ckpt")

        self.data.setup()
        X_train, _ = self.data.train_data(prediction_season=valid_season, start_season=start_season)
        X_train = self._drop_and_sort_features(X_train)
        self._fit_scaler(X_train)

    def _drop_and_sort_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop ID columns and sort features alphabetically for consistency."""
        id_cols = [Columns.SEASON, Columns.T1_TEAM_ID, Columns.T2_TEAM_ID]
        df = df.drop(columns=[c for c in id_cols if c in df.columns])
        return df
