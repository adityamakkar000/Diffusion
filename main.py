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


@hydra.main(version_base=None, config_path="./configs")
def main(cfg: DictConfig) -> None:

    if "load" in cfg:
        load = True
        path = f"runs/{cfg.path}"
        assert os.path.exists(path), "path does not exist"
        cfg = OmegaConf.load(os.path.join(path, "config.yaml"))
    else:
        load = False
        path = f"runs/{cfg.save_dir}"
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

    # linear based noise scheduler
    beta_array = torch.linspace(B_1, B_T, T)
    alpha_bar_array = torch.zeros(T).to(device)
    alpha_bar_array[0] = 1 - beta_array[0]
    for i in range(1, T):
        alpha_bar_array[i] = alpha_bar_array[i - 1] * (1 - beta_array[i])
   
        def get_alpha(t: Tensor) -> float:
        alpha_bar = alpha_bar_array[t].view(-1, 1, 1, 1)
        assert (
            alpha_bar.dim() == 4
            and alpha_bar.shape[0] == t.shape[0]
            and alpha_bar.shape[1] == alpha_bar.shape[2] == alpha_bar.shape[3]
        )
        return alpha_bar

    def get_training_batch(split: "str" = "train") -> Tuple[Tensor, Tensor, Tensor]:
        x_0 = dataset[split]()
        x_0 = x_0.to(device)
        t = torch.randint(1, T, (batch_size_test,)).int().to(device)
        alpha_bar = get_alpha(t)
        z = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * z
        x_t = x_t.to(device)

        return x_t, t, z

    if cfg.model == "self":
        model = UNET(**diffusion_params).to(device)
    elif cfg.model == "HF":
        model = createHFDiffusion(diffusion_params).to(device)
    else:
        raise NotImplementedError("model not implemented")

    optimizer = torch.optim.Adam(model.parameters(), lr)

    lowest_loss = 100000
    if load:
        checkpoint = torch.load(
            f"{path}/model.pt", weights_only=True, map_location=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lowest_loss = checkpoint["loss"]
        print(
            "loaded model from checkpoint at step",
            checkpoint["step"],
            "with loss",
            checkpoint["loss"],
        )

    def training_step(split: "str", model) -> Tensor:

        if split == "train":
            optimizer.zero_grad()
        if split == "test":
            loss_final = 0

        for b in range(batch_size_accumlation_multiple):

            x_t, t, z = get_training_batch(split)
            predicted_noise = model(x_t, t)

            if cfg.model == "HF":
                predicted_noise = predicted_noise["sample"]

            loss = (
                1 / (T * batch_size_accumlation_multiple * batch_size_train)
            ) * F.mse_loss(predicted_noise, z, reduction="sum")

            if split == "train":
                loss.backward()

            if split == "test":
                loss_final += loss.detach().item()

        if split == "train":
            optimizer.step()
            return loss.detach().item() * batch_size_accumlation_multiple
        if split == "test":
            return loss_final

    print("Files loaded, setting up model ...\n\n")
    print("device", device)
    print("model params", sum(p.numel() for p in model.parameters()))
    print("starting training ... \n\n")

    start = time.time()
    _ = 0
    while True:
        _ += 1
        optimizer.zero_grad()
        model.train()
        loss = training_step("train", model)
        optimizer.step()

        if _ % checkpoint_interval == 0:


            if device == "cuda":
                torch.cuda.synchronize()
            end = time.time() - start

            batch_idx = (
                (_ + 1) * batch_size_accumlation_multiple * batch_size_train
            ) % len(train_data)
            percentage_complete = 100.0 * (_ + 1) / max_steps
            batch_percentage_complete = 100.0 * (batch_idx) / len(train_data)

            with torch.no_grad():
                model.eval()
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

            metric_string = f"Step {_} | Train Loss: {loss:.6f} | Eval Loss: {loss_eval:.6f} | Time: {end:.2f}s | {percentage_complete:.2f}% complete"
            if saved:
                metric_string += " | saved model"

            print(metric_string)

            start = time.time()


if __name__ == "__main__":
    main()
