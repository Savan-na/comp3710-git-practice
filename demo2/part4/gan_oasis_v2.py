# ============================================================
# COMP3710 DEMO 2 - PART 4 TASK 3
# OASIS GAN V2 - 64x64 WGAN-GP
# ============================================================

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image


DEFAULT_DATA_DIR = Path("/home/groups/comp3710/OASIS/keras_png_slices_train")
DEFAULT_OUTPUT_DIR = Path("demo2/part4/gan_outputs_v2")
IMAGE_SIZE = 64


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class OASISDataset(Dataset):
    def __init__(self, image_dir):
        self.files = sorted(Path(image_dir).glob("*.png"))
        if not self.files:
            raise RuntimeError(f"No PNG files found in {image_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = read_image(str(self.files[index]), mode=ImageReadMode.GRAY).float()
        image = F.interpolate(
            image.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return image / 127.5 - 1.0


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.network = nn.Sequential(
            UpBlock(512, 256),
            UpBlock(256, 128),
            UpBlock(128, 64),
            UpBlock(64, 32),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.project(z).view(z.size(0), 512, 4, 4)
        return self.network(x)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.features = nn.Sequential(
            block(1, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512),
        )
        self.output = nn.Linear(512 * 4 * 4, 1)

    def forward(self, x):
        return self.output(self.features(x).flatten(1)).squeeze(1)


def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def gradient_penalty(critic, real, fake, device):
    batch = real.size(0)
    alpha = torch.rand(batch, 1, 1, 1, device=device)
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(mixed)

    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=mixed,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.flatten(1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


@torch.no_grad()
def generated_images(generator, noise):
    was_training = generator.training
    generator.eval()
    result = generator(noise)
    if was_training:
        generator.train()
    return result


@torch.no_grad()
def diversity(images):
    flat = (((images.clamp(-1, 1) + 1) / 2).flatten(1))
    if flat.size(0) < 2:
        return 0.0
    return torch.pdist(flat).mean().item()


def save_grid(generator, noise, path):
    images = generated_images(generator, noise)
    save_image((images + 1) / 2, str(path), nrow=5)
    return diversity(images)


def save_loss_plot(history, path):
    plt.figure(figsize=(10, 6))
    plt.plot(history["epochs"], history["critic_loss"], label="Critic loss")
    plt.plot(history["epochs"], history["generator_loss"], label="Generator loss")
    plt.plot(history["epochs"], history["gradient_penalty"], label="Gradient penalty")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("OASIS GAN V2 - WGAN-GP")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_checkpoint(path, epoch, generator, critic, opt_g, opt_c, history, args):
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "critic": critic.state_dict(),
            "optimizer_g": opt_g.state_dict(),
            "optimizer_c": opt_c.state_dict(),
            "history": history,
            "latent_dim": args.latent_dim,
            "image_size": IMAGE_SIZE,
            "args": vars(args),
        },
        path,
    )


def train(args):
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. Submit this script through Slurm.")

    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMP3710 DEMO 2 - PART 4 TASK 3")
    print("OASIS GAN V2 - 64x64 WGAN-GP")
    print("=" * 70)
    print("GPU:", torch.cuda.get_device_name(0))

    dataset = OASISDataset(args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    print("Training images:", len(dataset))
    print("Image tensor shape:", dataset[0].shape)

    real_samples = next(iter(loader))[:25]
    save_image((real_samples + 1) / 2, str(output_dir / "real_oasis_64.png"), nrow=5)

    generator = Generator(args.latent_dim).to(device)
    critic = Critic().to(device)
    generator.apply(init_weights)
    critic.apply(init_weights)

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr_c, betas=(0.0, 0.9))

    fixed_noise = torch.randn(25, args.latent_dim, device=device)

    history = {
        "epochs": [],
        "critic_loss": [],
        "generator_loss": [],
        "gradient_penalty": [],
        "diversity": [],
    }

    total_start = time.time()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        generator.train()
        critic.train()

        c_sum = 0.0
        g_sum = 0.0
        gp_sum = 0.0
        c_updates = 0
        g_updates = 0
        epoch_start = time.time()

        for batch_index, real in enumerate(loader, start=1):
            real = real.to(device, non_blocking=True)
            batch = real.size(0)

            # Critic update
            opt_c.zero_grad(set_to_none=True)

            z = torch.randn(batch, args.latent_dim, device=device)
            with torch.no_grad():
                fake = generator(z)

            real_score = critic(real)
            fake_score = critic(fake)
            gp = gradient_penalty(critic, real, fake, device)

            c_loss = fake_score.mean() - real_score.mean() + args.lambda_gp * gp

            if not torch.isfinite(c_loss):
                raise RuntimeError("Critic loss became non-finite.")

            c_loss.backward()
            opt_c.step()

            c_sum += c_loss.item()
            gp_sum += gp.item()
            c_updates += 1

            # Generator update
            if global_step % args.n_critic == 0:
                opt_g.zero_grad(set_to_none=True)
                z = torch.randn(batch, args.latent_dim, device=device)
                fake = generator(z)
                g_loss = -critic(fake).mean()

                if not torch.isfinite(g_loss):
                    raise RuntimeError("Generator loss became non-finite.")

                g_loss.backward()
                opt_g.step()

                g_sum += g_loss.item()
                g_updates += 1

            global_step += 1

            if batch_index == 1 or batch_index % 100 == 0:
                print(
                    f"Epoch {epoch:03d}/{args.epochs:03d} "
                    f"Batch {batch_index:03d}/{len(loader):03d} "
                    f"C={c_loss.item():.4f} "
                    f"GP={gp.item():.4f} "
                    f"G(avg)={g_sum / max(g_updates, 1):.4f}"
                )

        mean_c = c_sum / max(c_updates, 1)
        mean_g = g_sum / max(g_updates, 1)
        mean_gp = gp_sum / max(c_updates, 1)
        fixed_images = generated_images(generator, fixed_noise)
        fixed_diversity = diversity(fixed_images)

        history["epochs"].append(epoch)
        history["critic_loss"].append(mean_c)
        history["generator_loss"].append(mean_g)
        history["gradient_penalty"].append(mean_gp)
        history["diversity"].append(fixed_diversity)

        print(
            f"Epoch {epoch:03d} complete | "
            f"C={mean_c:.4f} | G={mean_g:.4f} | "
            f"GP={mean_gp:.4f} | diversity={fixed_diversity:.4f} | "
            f"time={time.time() - epoch_start:.1f}s"
        )

        if epoch == 1 or epoch % args.save_every == 0 or epoch == args.epochs:
            sample_path = output_dir / f"generated_epoch_{epoch:03d}.png"
            d = save_grid(generator, fixed_noise, sample_path)
            print("Saved:", sample_path)
            print("Fixed-noise diversity:", f"{d:.6f}")

            save_loss_plot(history, output_dir / "gan_v2_training_losses.png")

            save_checkpoint(
                output_dir / "gan_oasis_v2_latest.pt",
                epoch,
                generator,
                critic,
                opt_g,
                opt_c,
                history,
                args,
            )

            with open(output_dir / "training_history.json", "w") as f:
                json.dump(history, f, indent=2)

    final_checkpoint = output_dir / "gan_oasis_v2_final.pt"

    save_checkpoint(
        final_checkpoint,
        args.epochs,
        generator,
        critic,
        opt_g,
        opt_c,
        history,
        args,
    )

    final_noise = torch.randn(25, args.latent_dim, device=device)
    final_diversity = save_grid(
        generator,
        final_noise,
        output_dir / "gan_v2_final_generated_brains.png",
    )

    save_loss_plot(history, output_dir / "gan_v2_training_losses.png")

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 70)
    print("GAN V2 TRAINING COMPLETE")
    print("=" * 70)
    print("Training time:", f"{(time.time() - total_start) / 60:.2f} minutes")
    print("Final critic loss:", f"{history['critic_loss'][-1]:.6f}")
    print("Final generator loss:", f"{history['generator_loss'][-1]:.6f}")
    print("Final gradient penalty:", f"{history['gradient_penalty'][-1]:.6f}")
    print("Final diversity:", f"{final_diversity:.6f}")
    print("Final checkpoint:", final_checkpoint)
    print("Final generated brains:", output_dir / "gan_v2_final_generated_brains.png")
    print("Loss plot:", output_dir / "gan_v2_training_losses.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--lr-g", type=float, default=1e-4)
    parser.add_argument("--lr-c", type=float, default=1e-4)
    parser.add_argument("--lambda-gp", type=float, default=10.0)
    parser.add_argument("--n-critic", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3710)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
