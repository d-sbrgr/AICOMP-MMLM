import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from torch import Tensor

from .bce_with_entropy_loss import BCEWithEntropyLoss
from .config import NeuralNetworkHyperparamConfig


class DNN(L.LightningModule):
    """
    A flexible PyTorch Lightning module that can be configured to create
    various neural network architectures.

    This module supports:
    - Arbitrary number of hidden layers with configurable sizes
    - Multiple activation functions
    - Batch normalization and dropout
    - Multiple optimizers and learning rate schedulers
    - Custom loss functions
    """

    def __init__(self, input_dim: int, config: NeuralNetworkHyperparamConfig):
        """
        Initialize the neural network.

        Args:
            input_dim: Number of input features
            config: Configuration object with architecture and training parameters
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.input_dim = input_dim

        self.layers = self._build_network()

        self._initialize_weights()
        self.loss_fn = self._get_loss_function()

        self.train_accuracy = torchmetrics.Accuracy(task="binary")
        self.val_accuracy = torchmetrics.Accuracy(task="binary")
        self.train_brier = torchmetrics.MeanSquaredError()
        self.val_brier = torchmetrics.MeanSquaredError()

    def _build_network(self) -> nn.Module:
        """Build the neural network architecture based on configuration."""
        layers = []

        # Input dropout
        if self.config.input_dropout > 0:
            layers.append(nn.Dropout(p=self.config.input_dropout))

        # Build hidden layers
        prev_size = self.input_dim
        for hidden_size in self.config.hidden_layers:
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))

            # Batch normalization
            if self.config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            # Activation function
            layers.append(self._get_activation())

            # Dropout
            if self.config.dropout > 0:
                layers.append(nn.Dropout(p=self.config.dropout))

            prev_size = hidden_size

        # Output layer (single output for regression)
        layers.append(nn.Linear(prev_size, 1))

        # Sigmoid activation for probability output (0-1)
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def _get_activation(self) -> nn.Module:
        """Get activation function based on configuration."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
        }

        activation = activations.get(self.config.activation.lower())
        if activation is None:
            raise ValueError(f"Unknown activation function: {self.config.activation}")

        return activation

    def _get_loss_function(self) -> nn.Module:
        """Get loss function based on configuration."""
        loss_functions = {
            "mse": nn.MSELoss(),
            "mae": nn.L1Loss(),
            "huber": nn.HuberLoss(),
            "bce": nn.BCELoss(),
            "bce_entropy": BCEWithEntropyLoss(),
        }

        loss_fn = loss_functions.get(self.config.loss_function.lower())
        if loss_fn is None:
            raise ValueError(f"Unknown loss function: {self.config.loss_function}")

        return loss_fn

    def _initialize_weights(self) -> None:
        """Initialize network weights based on configuration."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.weight_init == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif self.config.weight_init == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif self.config.weight_init == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                elif self.config.weight_init == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                elif self.config.weight_init == "normal":
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                elif self.config.weight_init == "uniform":
                    nn.init.uniform_(module.weight, a=-0.05, b=0.05)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network."""
        return self.layers(x).squeeze(-1)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self.train_accuracy(y_hat, y)
        self.train_brier(y_hat, y)

        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_brier", self.train_brier, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Validation step."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self.val_accuracy(y_hat, y)
        self.val_brier(y_hat, y)

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_brier", self.val_brier, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self) -> dict:
        """Configure optimizer and learning rate scheduler."""
        # Select optimizer
        if self.config.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        # If no scheduler, return optimizer only
        if self.config.scheduler is None:
            return {"optimizer": optimizer}

        # Configure learning rate scheduler
        if self.config.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )
        elif self.config.scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.config.scheduler_gamma,
            )
        elif self.config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.max_epochs,
            )
        elif self.config.scheduler == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.config.scheduler_gamma,
                patience=self.config.scheduler_patience,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
