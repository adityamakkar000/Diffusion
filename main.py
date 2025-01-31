import torch
import torch.nn.functional as F
from torch import Tensor
from setupDataset import get_dataloaders
from models.DDPM.model import UNET, UNetConfig
from models.DiT.model import DiT, DiTConfig
import time
import os
import math
import hydra
from omegaconf import DictConfig, OmegaConf, MISSING
from hydra.core.config_store import ConfigStore
from typing import Tuple, Union, Optional
from tqdm import tqdm
from dataclasses import dataclass
import wandb
from datetime import datetime


@dataclass
class TrainingConfig:
    batch_size_train: int = MISSING
    batch_size_accumulation_multiple: int = MISSING
    batch_size_test: int = MISSING
    lr: float = MISSING
    min_lr: float = MISSING
    warmup_steps: int = MISSING
    end_steps: int = MISSING
    size: Tuple[int, int] = MISSING
    B_1: float = MISSING
    B_T: float = MISSING
    T: int = MISSING
    checkpoint_interval: int = MISSING
    max_steps: int = MISSING
    dataset: str = MISSING


@dataclass
class Config:
    training: TrainingConfig = MISSING
    model: str = MISSING
    unet_model_config: Optional[UNetConfig] = None
    dit_model_config: Optional[DiTConfig] = None
    save_dir: str = MISSING
    path: Union[None, str] = MISSING
    description: Union[None, str] = MISSING


cs = ConfigStore.instance()
cs.store(name="base", node=Config)


@hydra.main(version_base=None, config_path="./configs")
def main(cfg: DictConfig) -> None:
    save_dir = f"./runs/{cfg.save_dir}"
    if "path" in cfg and cfg.path is not None:
        load = True
        path = f"./runs/{cfg.path}"
        assert os.path.exists(path), "path does not exist"
        print(f"using model found at {path}")
    else:
        load = False

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    OmegaConf.save(config=cfg, f=os.path.join(save_dir, "config.yaml"))
    print(OmegaConf.to_yaml(cfg))

    assert cfg.model in ["UNET", "DiT"], "model must be UNET or DiT"
    assert not (cfg.model == "DiT" and cfg.dit_model_config is None), (
        "DiT model config must be provided"
    )
    assert not (cfg.model == "UNET" and cfg.unet_model_config is None), (
        "UNET model config must be provided"
    )

    batch_size_train = cfg.training.batch_size_train
    batch_size_accumlation_multiple = cfg.training.batch_size_accumulation_multiple
    batch_size_test = cfg.training.batch_size_test
    lr = cfg.training.lr

    ds_split = cfg.training.dataset
    size = cfg.training.size
    if ds_split == "cifar":
        size = (32, 32)

    B_1 = cfg.training.B_1
    B_T = cfg.training.B_T
    T = cfg.training.T

    checkpoint_interval = cfg.training.checkpoint_interval

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    train_data, test_data = get_dataloaders(
        ds_split, size, batch_size_train, batch_size_test
    )

    dataset = {"train": train_data, "test": test_data}
    batch_size = {"train": batch_size_train, "test": batch_size_test}

    # linear based noise scheduler
    beta_array = torch.linspace(B_1, B_T, T, dtype=torch.float32).to(device)
    alpha_array = 1.0 - beta_array
    alpha_bar_array = torch.cumprod(alpha_array, dim=0, dtype=torch.float32)

    def get_training_batch(split: "str" = "train") -> Tuple[Tensor, Tensor, Tensor]:
        x_0 = dataset[split]().to(device)
        t = torch.randint(1, T, (batch_size[split],)).int().to(device)
        alpha_bar = alpha_bar_array[t].view(-1, 1, 1, 1)
        z = torch.randn_like(x_0)
        x_t = (torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * z).to(device)

        return x_t, t, z

    if cfg.model == "UNET":
        model = UNET(T=T, **cfg.unet_model_config)
    elif cfg.model == "DiT":
        model = DiT(T=T, length=size[0], **cfg.dit_model_config)
    else:
        raise ValueError("model must be UNET or DiT")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    lr = cfg.training.lr
    min_lr = cfg.training.min_lr
    warmup_steps = cfg.training.warmup_steps
    end_steps = cfg.training.end_steps

    def get_lr(step):
        if step < warmup_steps:
            return lr * (step + 1) / (warmup_steps + 1)
        if step > end_steps:
            return min_lr
        return min_lr + 0.5 * (lr - min_lr) * (
            1 + (math.cos((step - warmup_steps) / (end_steps - warmup_steps) * math.pi))
        )

    def training_step(split: "str", model) -> Tensor:
        if split == "train":
            model.train()
            optimizer.zero_grad()
        if split == "test":
            model.eval()
            loss_final = 0

        with torch.enable_grad() if split == "train" else torch.no_grad():
            for b in range(batch_size_accumlation_multiple):
                x_t, t, z = get_training_batch(split)
                predicted_noise = model(x_t, t)

                if cfg.model == "HF":
                    predicted_noise = predicted_noise["sample"]

                loss = (
                    1 / (T * batch_size_accumlation_multiple * batch_size[split])
                ) * F.mse_loss(predicted_noise, z, reduction="sum")

                if split == "train":
                    loss.backward()

                if split == "test":
                    loss_final += loss.detach().item()

        if split == "train":
            optimizer.step()
            return loss.detach().item()

        if split == "test":
            return loss_final

    lowest_loss = 100000
    max_steps = cfg.training.max_steps
    step_inital = 0

    if load:
        checkpoint = torch.load(
            f"{path}/model.pt", weights_only=True, map_location=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lowest_loss = checkpoint["loss"]
        step_inital = checkpoint["step"] if cfg.path == cfg.save_dir else 0
        print(
            "loaded model from checkpoint at step",
            checkpoint["step"],
            "with loss",
            checkpoint["loss"],
        )

    print("Files loaded, setting up model ...\n\n")
    print("device", device)
    print("model params", sum(p.numel() for p in model.parameters()))
    print("starting training ... \n\n")

    start = time.time()

    wandb_log = True if cfg.description is not None else False

    resume = True if load and cfg.path == cfg.save_dir else False

    if wandb_log:
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        if resume:
            wandb.init(
                project="diffusion",
                name=f"{cfg.save_dir}_{cfg.description}_{current_time}",
                id=cfg.save_dir,
                resume_from=f"{cfg.save_dir}?_step={step_inital}",
                config=OmegaConf.to_container(cfg),
            )
        else:
            wandb.init(
                project="diffusion",
                name=f"{cfg.save_dir}_{cfg.description}_{current_time}",
                id=cfg.save_dir,
                resume="allow",
                config=OmegaConf.to_container(cfg),
            )

    for step in tqdm(range(step_inital, max_steps), desc="Training"):
        current_lr = get_lr(step)
        for param in optimizer.param_groups:
            param["lr"] = current_lr
        loss = training_step("train", model)
        if not (step % checkpoint_interval == 0) and wandb_log:
            wandb.log({"train loss": loss, "step": step, "lr": current_lr})

        if step % checkpoint_interval == 0:
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.time() - start

            loss_eval = training_step("test", model)
            if wandb_log:
                wandb.log(
                    {
                        "lr": current_lr,
                        "train loss": loss,
                        "val loss": loss_eval,
                        "step": step,
                    }
                )

            saved = False

            if not os.path.exists(f"{save_dir}/checkpoints"):
                os.makedirs(f"{save_dir}/checkpoints")

            # best save
            if loss_eval < lowest_loss:
                lowest_loss = loss_eval
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": round(loss_eval, 6),
                    },
                    f"{save_dir}/model.pt",
                )
                saved = True

            # increment save
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": round(loss_eval, 6),
                },
                f"{save_dir}/checkpoints/model.pt",
            )

            if not wandb_log:
                percentage_complete = 100.0 * (step + 1) / max_steps
                metric_string = f"Step {step} | Eval Loss: {loss_eval:.6f} | Time: {end:.2f}s | {percentage_complete:.2f}% complete "
                if saved:
                    metric_string += " | saved model"

                print(metric_string)

        start = time.time()


wandb.finish()

if __name__ == "__main__":
    main()
