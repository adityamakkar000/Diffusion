import torch
import torch.nn.functional as F
from torch import Tensor
from setupDataset import get_dataloaders
from models.DDPM.model import UNET
from models.hf_diff.diff import createHFDiffusion
import time
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Tuple
from tqdm import tqdm


@hydra.main(version_base=None, config_path="./configs")
def main(cfg: DictConfig) -> None:

    if "path" in cfg and cfg.path is not None:
        load = True
        path = f"./runs/{cfg.path}"
        assert os.path.exists(path), "path does not exist"
        print(f"using model found at {path}")
    else:
        load = False
        path = f"./runs/{cfg.save_dir}"
        if not os.path.exists(path):
            os.makedirs(path)
        OmegaConf.save(config=cfg, f=os.path.join(path, "config.yaml"))

    print(OmegaConf.to_yaml(cfg))

    batch_size_train = cfg.training.batch_size_train
    batch_size_accumlation_multiple = cfg.training.batch_size_accumulation_multiple
    batch_size_test = cfg.training.batch_size_test
    lr = cfg.training.lr
    max_steps = cfg.training.max_steps
    scale = cfg.training.scale
    size = cfg.training.size

    B_1 = cfg.training.B_1
    B_T = cfg.training.B_T
    T = cfg.training.T

    checkpoint_interval = cfg.training.checkpoint_interval

    diffusion_params = cfg.model_config
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    train_data, test_data = get_dataloaders(size, batch_size_train, batch_size_test)

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

    if cfg.model == "self":
        model = UNET(**diffusion_params).to(device)
    elif cfg.model == "HF":
        model = createHFDiffusion(diffusion_params).to(device)
    else:
        raise NotImplementedError("model not implemented")

    optimizer = torch.optim.Adam(model.parameters(), lr)

    lr = cfg.lr
    min_lr = cfg.min_lr
    warmp_up_steps = cfg.warmp_up_steps
    end_steps = cfg.end_steps

    def get_lr(step):
        if step < warmp_up_steps:
            return lr * (step + 1) / (warmp_up_steps + 1)
        if step > end_steps:
            return min_lr
        return min_lr + 0.5 * (lr - min_lr) * (
            1
            + (
                torch.cos(
                    (step - warmp_up_steps) / (end_steps - warmp_up_steps) * torch.pi
                )
            ).item()
        )

    def training_step(split: "str", model, return_loss=False) -> Tensor:

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

            if return_loss:
                return loss.detach().item() * batch_size_accumlation_multiple
        if split == "test":
            return loss_final

    lowest_loss = 100000
    step = 0

    if load:
        checkpoint = torch.load(
            f"{path}/model.pt", weights_only=True, map_location=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lowest_loss = checkpoint["loss"]
        # step = checkpoint["step"]
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

    for _ in tqdm(range(step, max_steps), desc="Training"):

        # set learning rate
        lr = get_lr(_)
        for param in optimizer.param_groups:
            param["lr"] = lr

        if _ % checkpoint_interval == 0:
            loss = training_step("train", model, return_loss=True)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.time() - start

            batch_idx = (
                (_ + 1) * batch_size_accumlation_multiple * batch_size_train
            ) % len(train_data)
            percentage_complete = 100.0 * (_ + 1) / max_steps
            batch_percentage_complete = 100.0 * (batch_idx) / len(train_data)

            loss_eval = training_step("test", model)

            saved = False
            if loss_eval < lowest_loss:
                lowest_loss = loss_eval
                torch.save(
                    {
                        "step": _,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": round(loss_eval, 6),
                    },
                    f"{path}/model.pt",
                )
                saved = True

            metric_string = f"Step {_} | Train Loss: {loss:.6f} | Eval Loss: {loss_eval:.6f} | Time: {end:.2f}s | {percentage_complete:.2f}% complete | {batch_percentage_complete:.2f}% dataset complete "
            if saved:
                metric_string += " | saved model"

            print(metric_string)

            start = time.time()

        else:
            training_step("train", model)


if __name__ == "__main__":
    main()
