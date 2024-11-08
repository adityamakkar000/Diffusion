import torch
import torch.nn.functional as F
from torch import Tensor

from setupDataset import get_dataloaders
import matplotlib.pyplot as plt
from DDPM.model import UNET

import time
from itertools import cycle

import hydra
from omegaconf import DictConfig, OmegaConf




@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    
    config = ExperimentConfig()

    batch_size_train = config.batch_size_train
    batch_size_accumlation_multiple = config.batch_size_accumulation_multiple
    batch_size_test = config.batch_size_test
    lr = config.lr
    max_steps = config.max_steps
    scale = config.scale
    size = config.size

    B_1 = config.B_1
    B_T = config.B_T
    T = config.T

    diffusion_params = config.diffusion_params

    PATH = config.PATH

    device = config.device
    train_data, test_data = get_dataloaders(size, batch_size_train, batch_size_test)

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


    model = UNET(**diffusion_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    if args.load:
        checkpoint = torch.load(PATH, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(
            "loaded model from checkpoint at step",
            checkpoint["step"],
            "with loss",
            checkpoint["loss"],
        )

    print("files loaded, setting up model ...\n\n")
    print("device", device)
    print("model params", sum(p.numel() for p in model.parameters()))
    print("starting training ... \n\n")

    start = time.time()
    for _ in range(max_steps):

        optimizer.zero_grad()
        for b in range(batch_size_accumlation_multiple):
            data = train_data()
            x_0 = data.to(device)

            t = torch.randint(1, T, (batch_size_train,)).int().to(device)
            alpha_bar = get_alpha(t)
            z = torch.randn_like(x_0)
            x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * z
            x_t = x_t.to(device)



            predicted_noise = model(x_t, t)

            loss = (
                1 / (T * batch_size_accumlation_multiple * batch_size_train)
            ) * F.mse_loss(predicted_noise, z, reduction="sum")
            loss.backward()


            # delete intermediate variables

            if b < batch_size_accumlation_multiple - 1:
                del loss
            del x_0
            del x_t
            del predicted_noise
            del z

        optimizer.step()

        if _ % 10 == 0:

            loss_metric = loss.detach().item() * batch_size_accumlation_multiple
            del loss

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
            loss_eval = 0
            for b in range(batch_size_accumlation_multiple):
                x_0 = test_data()
                x_0 = x_0.to(device)
                t = torch.randint(1, T, (batch_size_test,)).int().to(device)
                alpha_bar = get_alpha(t)
                z = torch.randn_like(x_0)
                x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * z
                x_t = x_t.to(device)

                predicted_noise = model(x_t, t)

                loss = (
                    1 / (T * batch_size_test * batch_size_accumlation_multiple)
                ) * F.mse_loss(predicted_noise, z, reduction="sum")
                loss_eval += loss.detach().item()

                # delete intermediate variables
                del loss
                del x_0
                del x_t
                del predicted_noise
                del z



            model.train()


            print(
                f"Step {_}/{max_steps} | Train Loss: {loss_metric:.6f} | Eval Loss: {loss_eval:.6f} | Time: {end:.2f}s | {percentage_complete:.2f}% complete"
            )

            torch.save(
                {
                    "step": _,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": round(loss_metric, 6),
                },
                PATH,
            )

            start = time.time()
