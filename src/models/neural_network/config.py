from dataclasses import asdict, dataclass, field
from typing import Any

from ..hyperparam_config import HyperparamConfig


@dataclass
class NeuralNetworkHyperparamConfig(HyperparamConfig):
    """
    Configuration for PyTorch Lightning Neural Network models.

    Supports flexible architecture definition with configurable layers,
    activation functions, dropout, and training hyperparameters.
    """

    # Architecture configuration
    hidden_layers: list[int] = field(default_factory=lambda: [128, 64, 32])
    """List of hidden layer sizes. Empty list means direct input to output."""

    activation: str = "relu"
    """Activation function: 'relu', 'leaky_relu', 'tanh', 'sigmoid', 'gelu', 'elu'"""

    dropout: float = 0.2
    """Dropout probability (0.0 to disable)"""

    batch_norm: bool = True
    """Whether to use batch normalization after each hidden layer"""

    input_dropout: float = 0.0
    """Dropout probability for input layer (0.0 to disable)"""

    # Training hyperparameters
    learning_rate: float = 0.001
    """Initial learning rate for optimizer"""

    batch_size: int = 512
    """Batch size for training"""

    max_epochs: int = 200
    """Maximum number of training epochs"""

    optimizer: str = "adam"
    """Optimizer: 'adam', 'sgd', 'rmsprop', 'adamw'"""

    weight_decay: float = 0.0
    """L2 regularization coefficient"""

    momentum: float = 0.9
    """Momentum for SGD optimizer (ignored for others)"""

    # Learning rate scheduler
    scheduler: str | None = None
    """Learning rate scheduler: None, 'step', 'exponential', 'cosine', 'reduce_on_plateau'"""

    scheduler_step_size: int = 30
    """Step size for StepLR scheduler"""

    scheduler_gamma: float = 0.1
    """Multiplicative factor for learning rate decay"""

    scheduler_patience: int = 10
    """Patience for ReduceLROnPlateau scheduler"""

    # Early stopping
    early_stopping: bool = True
    """Whether to use early stopping"""

    early_stopping_patience: int = 15
    """Number of epochs with no improvement after which training will be stopped"""

    early_stopping_min_delta: float = 1e-4
    """Minimum change in monitored metric to qualify as an improvement"""

    # Loss function
    loss_function: str = "mse"
    """Loss function: 'mse', 'mae', 'huber', 'bce' (binary cross entropy), 'bce_entropy'"""

    # Validation
    validation_split: float = 0.1
    """Fraction of training data to use for validation during training"""

    # Model initialization
    weight_init: str = "xavier_uniform"
    """Weight initialization: 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
    'normal', 'uniform'"""

    # Training behavior
    gradient_clip_val: float | None = None
    """Gradient clipping value (None to disable)"""

    accumulate_grad_batches: int = 1
    """Number of batches to accumulate gradients over"""

    num_workers: int = 32
    """Number of workers for data loading (0 for main process only)"""

    seed: int | None = None
    """Random seed for reproducibility (None for no seeding)"""

    def as_params(self) -> dict[str, Any]:
        """Convert config to dictionary of parameters."""
        return asdict(self)

    def run_name(self) -> dict[str, Any]:
        """Return key hyperparameters for run naming."""
        return {
            "hidden_layers": "_".join(map(str, self.hidden_layers)),
            "activation": self.activation,
            "lr": self.learning_rate,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
        }
