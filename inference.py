import torch
import matplotlib.pyplot as plt
from models.DDPM.model import UNET
from models.hf_diff.diff import createHFDiffusion
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


@hydra.main(version_base=None, config_path="./configs")
def main(cfg: DictConfig) -> None:
    if "path" in cfg and cfg.path is not None:
        path = f"./runs/{cfg.path}"
        assert os.path.exists(path), f"path, {path}, does not exist"
        cfg = OmegaConf.load(os.path.join(path, "config.yaml"))

    else:
        ValueError("load not in config")

    if not os.path.exists(f"{path}/samples"):
        os.makedirs(f"{path}/samples")

    print(OmegaConf.to_yaml(cfg))
    print(f"Saving images at {path}")

    lr = cfg.training.lr
    size = cfg.training.size
    if cfg.training.dataset == "cifar":
        size = (32, 32)
    B_1 = cfg.training.B_1
    B_T = cfg.training.B_T
    T = cfg.training.T
    diffusion_params = cfg.model_config

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    # linear based noise scheduler
    beta_array = torch.linspace(B_1, B_T, T, dtype=torch.float32).to(device)
    alpha_array = 1.0 - beta_array
    alpha_bar_array = torch.cumprod(alpha_array, dim=0, dtype=torch.float32)

    def save_image(x, path):
        img = (x + 1.0) * 255.0 / 2.0
        img = img.type(torch.uint8)
        plt.imsave(path, img.permute(1, 2, 0).cpu().detach().numpy(), format="png")

    if cfg.model == "self":
        model = UNET(**diffusion_params).to(device)
    elif cfg.model == "HF":
        model = createHFDiffusion(diffusion_params).to(device)
    else:
        raise NotImplementedError("model not implemented")

    optimizer = torch.optim.Adam(model.parameters(), lr)

    checkpoint = torch.load(f"{path}/model.pt", weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(
        "loaded model from checkpoint at step",
        checkpoint["step"],
        "with loss",
        checkpoint["loss"],
    )

    print("Files loaded, setting up model ...\n\n")
    print("device", device)
    print("model params", sum(p.numel() for p in model.parameters()))
    print("starting inference ... \n\n")

    with torch.no_grad():
        model.eval()

        batch_size = 1
        x_t = torch.randn(batch_size, 3, size[0], size[1]).to(
            device
        )  # intitally set to normal distrubtion

        for t in tqdm(reversed(range(T)), desc="Generating images"):
            timesteps = t * torch.ones(batch_size).to(device).long()
            eps_pred = model(x_t, timesteps)

            alpha_bar_t = alpha_bar_array[timesteps].view(-1, 1, 1, 1)
            beta_t = beta_array[timesteps].view(-1, 1, 1, 1)
            alpha_t = alpha_array[timesteps].view(-1, 1, 1, 1)

            mean = (1 / alpha_t.sqrt()) * (
                x_t - (beta_t / ((1 - alpha_bar_t).sqrt())) * eps_pred
            )
            # mean = mean.clamp(-1, 1)  # numerical stability

            epsilon = torch.zeros_like(x_t)

            if t > 0:
                alpha_bar_t_sub1 = alpha_bar_array[timesteps - 1].view(-1, 1, 1, 1)
                beta_tilde = beta_t * (1 - alpha_bar_t_sub1) / (1 - alpha_bar_t)
                z = torch.randn_like(x_t)
                epsilon = (beta_tilde).sqrt() * z
            x_t = mean + epsilon

            if t % 100 == 0:
                save_image(x_t[0], f"{path}/samples/{t}.png")

        img = torch.clamp(x_t, -1, 1)
        save_image(img[0], f"{path}/generated_image.png")
        print(f"saved image at {path}/generated_image.png")


if __name__ == "__main__":
    main()
