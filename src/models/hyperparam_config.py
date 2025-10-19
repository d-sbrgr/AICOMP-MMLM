from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class HyperparamConfig:
    def as_params(self) -> dict[str, Any]:
        return asdict(self)

    def run_name(self) -> dict[str, Any]:
        return {"model": self.__class__.__name__}
