from dataclasses import dataclass, field
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from Diffusion.model import TrainingConfig


@dataclass
class ExperimentConfig:
    batch_size_train: int = 64
    batch_size_accumulation_multiple: int = 4
    batch_size_test: int = 64
    lr: float = 0.001
    max_steps: int = 1000
    scale: int = 4
    size: tuple[int, int] = field(default_factory=lambda: (64, 64))

    B_1: float = 1e-4
    B_T: float = 0.02
    T: int = 1000


    mode_type: Optional[str] = None
    diffusion_params: Optional[TrainingConfig] = None

    PATH: str = "model.pt"
    device: str = field(init=False)

    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"



cs = ConfigStore.instance()
cs.store(name="config", node=ExperimentConfig)