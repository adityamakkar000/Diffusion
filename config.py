from dataclasses import dataclass, field
import torch

@dataclass
class ExperimentConfig:
    batch_size_train: int = 4
    batch_size_accumulation_multiple: int = 4
    batch_size_test: int = 4
    lr: float = 0.001
    max_steps: int = 1000
    scale: int = 4
    size: tuple[int, int] = field(default_factory=lambda: (64, 64))

    B_1: float = 1e-4
    B_T: float = 0.02
    T: int = 1000

    diffusion_params: dict = field(default_factory=lambda: {
        'timeStep': 1000,
        'originalSize': (64,64),
        'inChannels': 3,
        'channels': [32, 64, 128],
        'strides': [2, 2],
        'n_heads': [1],
        'attn': [True, False, False],
        'resNetBlocks': [2, 2, 2],
        'dropout': [0.2],
    })
    PATH: str = "model.pt"

    device: str = field(init=False)

    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
