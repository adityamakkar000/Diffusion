from dataclasses import dataclass
import huggingface_hub
from diffusers import UNet2DModel


@dataclass
class TrainingConfig:
    image_size = 64  # the generated image resolution
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub

    seed = 0


config = TrainingConfig()


model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(
        32,
        64,
        128,
    ),
    down_block_types=(
        "AttnDownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "UpBlock2D",
        "AttnUpBlock2D",
        ),
)
