import torch
import torch.nn.functional as F
import torch.distributed as dist
import time
import os
import math
import hydra
import wandb
from torch import Tensor
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from setupDataset import get_dataloaders
from models.DDPM.model import UNET, UNetConfig
from models.DiT.model import DiT, DiTConfig
from omegaconf import DictConfig, OmegaConf, MISSING
from hydra.core.config_store import ConfigStore
from typing import Tuple, Union, Optional
from tqdm import tqdm
from dataclasses import dataclass
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
    torch.manual_seed(1234)

    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process =( ddp_rank == 0)
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.manual_seed(1234)

    save_dir = f"./runs/{cfg.save_dir}"

    if "path" in cfg and cfg.path is not None:
        load = True
        path = f"./runs/{cfg.path}"
        assert os.path.exists(path), "path does not exist"
        print(f"using model found at {path}")
    else:
        load = False

    if master_process:
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
    B_1 = cfg.training.B_1
    B_T = cfg.training.B_T
    T = cfg.training.T
    lr = cfg.training.lr
    min_lr = cfg.training.min_lr
    warmup_steps = cfg.training.warmup_steps
    end_steps = cfg.training.end_steps
    checkpoint_interval = cfg.training.checkpoint_interval
    lowest_loss = 100000
    max_steps = cfg.training.max_steps // ddp_world_size
    step_inital = 0
    if ds_split == "cifar":
        size = (32, 32)

    def get_idx(dataset, rank):
        l1 = len(dataset["train"])
        l2 = len(dataset["test"])

        chunk_size_1 = l1 // ddp_world_size
        chunk_size_2 = l2 // ddp_world_size

        start_idx_1 = rank * chunk_size_1
        end_idx_1 = (rank + 1) * chunk_size_1 if rank != ddp_world_size - 1 else l1

        start_idx_2 = rank * chunk_size_2
        end_idx_2 = (rank + 1) * chunk_size_2 if rank != ddp_world_size - 1 else l2

        return {
            "train": (start_idx_1, end_idx_1),
            "test": (start_idx_2, end_idx_2)
        }

    train_data, test_data = get_dataloaders(
        ds_split, size, batch_size_train, batch_size_test
    )
    dataset = {"train": train_data, "test": test_data}
    idx = get_idx(dataset, ddp_rank)
    batch_size = {
        "train": (batch_size_train),
        "test": (batch_size_test),
    }

    # linear based noise sckjheduler
    beta_array = torch.linspace(B_1, B_T, T, dtype=torch.float32).to(device)
    alpha_array = 1.0 - beta_array
    alpha_bar_array = torch.cumprod(alpha_array, dim=0, dtype=torch.float32)

    def get_training_batch(split: "str" = "train") -> Tuple[Tensor, Tensor, Tensor]:

        x_0 = dataset[split](idx[split]).to(device)
        t = torch.randint(1, T, (batch_size[split],)).int().to(device)
        alpha_bar = alpha_bar_array[t].view(-1, 1, 1, 1)
        z = torch.randn_like(x_0)
        x_t = (torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * z).to(device)

        return x_t, t, z

    if cfg.model == "UNET":
        model = UNET(T=T, **cfg.unet_model_config).to(device)
    elif cfg.model == "DiT":
        model = DiT(T=T, length=size[0], **cfg.dit_model_config).to(device)
    else:
        raise ValueError("model must be UNET or DiT")

    if device != 'mps' and device != 'cpu':
        try:
            model = torch.compile(model)
            if master_process:
                print("model compiled with torch compile")
        except:
            if master_process:
                print("model could not compile")

    optimizer = torch.optim.Adam(model.parameters(), lr)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank)
        if master_process:
            print("DDP model wrapped successfully")

    def get_lr(step):
        if step < warmup_steps:
            return lr * (step + 1) / (warmup_steps + 1)
        if step > end_steps:
            return min_lr
        return min_lr + 0.5 * (lr - min_lr) * (
            1 + (math.cos((step - warmup_steps) / (end_steps - warmup_steps) * math.pi))
        )

    def training_step(split: "str", model) -> Tensor:
        loss_final = 0.0
        if split == "train":
            model.train()
            optimizer.zero_grad()
        if split == "test":
            model.eval()

        with torch.enable_grad() if split == "train" else torch.no_grad():
            for b in range(batch_size_accumlation_multiple):
                x_t, t, z = get_training_batch(split)
                predicted_noise = model(x_t, t)

                if cfg.model == "HF":
                    predicted_noise = predicted_noise["sample"]

                loss = (
                    1
                    / (
                        T
                        * batch_size[split]
                        * batch_size_accumlation_multiple
                        * ddp_world_size
                    )
                ) * F.mse_loss(predicted_noise, z, reduction="sum")

                if split == "train":
                    if ddp:
                        model.require_backward_grad_sync = (
                            b == batch_size_accumlation_multiple - 1
                        )
                    loss.backward()
                loss_final += loss.detach()

        if ddp:
            dist.all_reduce(loss_final, op=dist.ReduceOp.SUM)

        results = {"loss": loss_final}

        if split == "train":
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            results["norm"] = norm

        return results

    if load:
        checkpoint = torch.load(
            f"{path}/model.pt", weights_only=True, map_location=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lowest_loss = checkpoint["loss"]
        step_inital = checkpoint["step"] if cfg.path == cfg.save_dir else 0
        if master_process:
            print(
                "loaded model from checkpoint at step",
                checkpoint["step"],
                "with loss",
                checkpoint["loss"],
            )

    if master_process:
        print("Files loaded, setting up model ...\n\n")
        print("device", device)
        print("model params", sum(p.numel() for p in model.parameters()))
        print("starting training ... \n\n")

        wandb_log = True if cfg.description is not None else False
        if wandb_log:
            current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            resume = True if load and cfg.path == cfg.save_dir else False
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

        start = time.time()

    for step in tqdm(range(step_inital, max_steps), desc="Training"):
        current_lr = get_lr(step)
        for param in optimizer.param_groups:
            param["lr"] = current_lr
        results = training_step("train", model)

        if not (step % checkpoint_interval == 0) and master_process and wandb_log:
            wandb.log(
                {
                    "train loss": results["loss"].item(),
                    "step": step,
                    "lr": current_lr,
                    "norm": results["norm"],
                }
            )

        if step % checkpoint_interval == 0:
            if torch.cuda.is_available():
                if ddp:
                    if master_process:
                        torch.cuda.synchronize()
                else:
                    torch.cuda.synchronize()

            if master_process:
                end = time.time() - start

            loss_eval = training_step("test", model)["loss"]
            if master_process:
                if wandb_log:
                    wandb.log(
                        {
                            "lr": current_lr,
                            "train loss": results["loss"].item(),
                            "norm": results["norm"],
                            "val loss": loss_eval.item(),
                            "step": step,
                        }
                    )

            saved = False

            if master_process:
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
                            "loss": round(loss_eval.item(), 6),
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
                        "loss": round(loss_eval.item(), 6),
                    },
                    f"{save_dir}/checkpoints/model.pt",
                )

                if not wandb_log:
                    percentage_complete = 100.0 * (step + 1) / max_steps
                    imgs_per_sec = (
                        batch_size["train"]
                        * batch_size_accumlation_multiple
                        * ddp_world_size
                        / end
                    )
                    metric_string = f"Step {step} | Train Loss: {results['loss'].item():.4f} | Eval Loss: {loss_eval.item():.4f} | Time: {end:.2f}s | img/s: {imgs_per_sec:.2f} | {percentage_complete:.2f}% complete "
                    if saved:
                        metric_string += " | saved model"

                    print(metric_string)

            start = time.time()

    if master_process:
        if wandb_log:
            wandb.finish()
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
