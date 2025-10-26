from collections.abc import Iterable

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ..dataloaders.base_dataloader import BaseDataloader
from ..experiments import Tracker
from .cross_validation.cv_config import CrossValidationConfig
from .hyperparam_config import HyperparamConfig


class Model:
    def __init__(self):
        pass

    def fit(self, season: int) -> None:
        """Fits the model to the data it requires.

        The training data may differ per model and hence needs to be loaded by each implementation.
        """
        raise NotImplementedError()

    def validate(self) -> None:
        """Validates the model on the validation set.

        The validation data may differ per model and hence needs to be loaded by each implementation.
        """
        pass

    def predict(self, matchups: pd.DataFrame) -> Iterable[float]:
        """Predicts the outcome of the given matchups.

        Args:
            matchups (pd.DataFrame): A dataframe containing the matchups to predict. The dataframe must contain
                the following columns:
                    - 'Season': The season of the matchup.
                    - 'TeamID1': The ID of the first team.
                    - 'TeamID2': The ID of the second team.
        Returns:
            pd.Series: A series containing the predicted outcome of the matchups. The index of the series must
                match the index of the input dataframe.
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.__class__.__name__.removesuffix("Model")


class SupervisedModel(Model):
    """
    Base class for supervised learning models with standardized components.

    This class provides a strongly-typed interface for models that use:
    - A data loader for loading training/validation/test data
    - Hyperparameter configuration
    - Cross-validation configuration
    - Experiment tracking

    Attributes:
        data: The data loader for loading training/validation/test data
        params: The hyperparameter configuration
        cv_cfg: The cross-validation configuration (optional)
        tracker: The experiment tracker for logging metrics
    """

    def __init__(
        self,
        data: BaseDataloader,
        params: HyperparamConfig,
        cv: CrossValidationConfig | None,
        tracker: Tracker,
        scaler: StandardScaler | MinMaxScaler | RobustScaler | None = None,
    ) -> None:
        """
        Initialize a supervised model with standardized components.

        Args:
            data: The data loader for loading training/validation/test data
            params: The hyperparameter configuration
            cv: The cross-validation configuration (can be None to disable CV)
            tracker: The experiment tracker for logging metrics
        """
        super().__init__()
        self.data = data
        self.params = params
        self.cv_cfg = cv
        self.tracker = tracker
        self._scaler = scaler

    def _fit_scaler(self, X: pd.DataFrame) -> None:
        """Fit the scaler on training data.

        Args:
            X: Training features (after dropping ID columns)
        """
        if self._scaler is None:
            return

        self._scaler.fit(X)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling transformation to features.

        Args:
            X: Features to transform

        Returns:
            Scaled features as DataFrame with same structure
        """
        if self._scaler is None:
            return X

        X_scaled = self._scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
