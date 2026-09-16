# ============================================================
# DEMO 2 - PART 4 TASK 3: OASIS GENERATIVE ADVERSARIAL NETWORK
# ============================================================
#
# Course:
# COMP3710 Lab Demonstration 2 - Part 4 Task 3
#
# Goal:
# Generate realistic and diverse OASIS brain MRI slices using a GAN.
#
# Evidence produced:
#   1. Real OASIS MRI examples
#   2. Generated MRI samples during training
#   3. Generator / discriminator loss plot
#   4. Diversity measurement for generated samples
#   5. Final GAN checkpoint
#
# ============================================================

import argparse
import copy
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


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DATA_DIR = Path(
    "/home/groups/comp3710/OASIS/keras_png_slices_train"
)

DEFAULT_OUTPUT_DIR = Path(
    "demo2/part4/gan_outputs"
)

IMAGE_SIZE = 256
IMAGE_CHANNELS = 1


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    # What: make random operations reproducible.
    # Why: helps compare GAN runs more reliably.
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# OASIS DATASET
# ============================================================

class OASISDataset(Dataset):
    """
    Loads the preprocessed OASIS MRI PNG slices.

    Input PNG:
        uint8 grayscale image, 256 x 256

    Output tensor:
        float32 [1, 256, 256]
        range [-1, 1]
    """

    def __init__(self, image_dir: Path):
        self.image_dir = Path(image_dir)
        self.image_files = sorted(self.image_dir.glob("*.png"))

        if len(self.image_files) == 0:
            raise RuntimeError(
                f"No PNG images found in: {self.image_dir}"
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]

        # What: read MRI as one-channel grayscale image.
        # Why: OASIS slices are grayscale medical images.
        image = read_image(
            str(image_path),
            mode=ImageReadMode.GRAY
        )

        if image.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(
                f"Unexpected image shape {image.shape} "
                f"for {image_path}"
            )

        # What: map uint8 [0,255] to float [-1,1].
        # Why: generator output uses Tanh, which produces [-1,1].
        image = image.float() / 127.5 - 1.0

        return image


# ============================================================
# GENERATOR
# ============================================================

class Generator(nn.Module):
    """
    Latent vector z:
        [B, latent_dim]

    Output:
        [B, 1, 256, 256]
    """

    def __init__(self, latent_dim=128):
        super().__init__()

        self.latent_dim = latent_dim

        self.project = nn.Sequential(
            nn.Linear(latent_dim, 1024 * 4 * 4),
            nn.BatchNorm1d(1024 * 4 * 4),
            nn.ReLU(True),
        )

        self.generator = nn.Sequential(

            # 4 -> 8
            nn.ConvTranspose2d(
                1024, 512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 8 -> 16
            nn.ConvTranspose2d(
                512, 256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 16 -> 32
            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 32 -> 64
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64 -> 128
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 128 -> 256
            nn.ConvTranspose2d(
                32, IMAGE_CHANNELS,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.Tanh()
        )

    def forward(self, z):
        x = self.project(z)
        x = x.view(z.size(0), 1024, 4, 4)
        return self.generator(x)


# ============================================================
# DISCRIMINATOR
# ============================================================

def snconv(in_channels, out_channels):
    """
    Spectral-normalised convolution.

    Spectral normalisation helps stabilise adversarial training
    and reduces the chance that the discriminator becomes
    excessively strong.
    """

    return nn.utils.spectral_norm(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )
    )


class Discriminator(nn.Module):
    """
    Input:
        [B, 1, 256, 256]

    Output:
        [B]
        unbounded real/fake score
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # 256 -> 128
            snconv(1, 32),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 -> 64
            snconv(32, 64),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 -> 32
            snconv(64, 128),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 -> 16
            snconv(128, 256),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 -> 8
            snconv(256, 512),
            nn.LeakyReLU(0.2, inplace=True),

            # 8 -> 4
            snconv(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.utils.spectral_norm(
            nn.Linear(1024 * 4 * 4, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x.squeeze(1)


# ============================================================
# INITIALISATION
# ============================================================

def initialise_weights(module):

    if isinstance(
        module,
        (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
    ):
        if hasattr(module, "weight") and module.weight is not None:
            try:
                nn.init.normal_(module.weight, 0.0, 0.02)
            except Exception:
                pass

        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(
        module,
        (nn.BatchNorm1d, nn.BatchNorm2d)
    ):
        if module.weight is not None:
            nn.init.normal_(module.weight, 1.0, 0.02)

        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ============================================================
# EMA GENERATOR
# ============================================================

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):

    for ema_parameter, parameter in zip(
        ema_model.parameters(),
        model.parameters()
    ):
        ema_parameter.mul_(decay).add_(
            parameter,
            alpha=1.0 - decay
        )

    for ema_buffer, buffer in zip(
        ema_model.buffers(),
        model.buffers()
    ):
        ema_buffer.copy_(buffer)


# ============================================================
# VISUALISATION
# ============================================================

@torch.no_grad()
def save_generated_samples(
    generator,
    fixed_noise,
    output_path,
):

    generator.eval()

    generated = generator(fixed_noise)

    # What: convert [-1,1] back to [0,1].
    # Why: PNG saving expects image intensities suitable for display.
    display_images = (
        generated.clamp(-1, 1) + 1.0
    ) / 2.0

    save_image(
        display_images,
        str(output_path),
        nrow=4
    )

    # Measure average pairwise distance between generated samples.
    flat = display_images.flatten(1)

    if flat.size(0) > 1:
        diversity = torch.pdist(flat).mean().item()
    else:
        diversity = 0.0

    return diversity


def save_loss_plot(
    epochs,
    generator_losses,
    discriminator_losses,
    output_path
):

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        generator_losses,
        label="Generator loss"
    )

    plt.plot(
        epochs,
        discriminator_losses,
        label="Discriminator loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("OASIS GAN Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180
    )

    plt.close()


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    epoch,
    generator,
    generator_ema,
    discriminator,
    optimizer_g,
    optimizer_d,
    generator_losses,
    discriminator_losses,
    args,
):

    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "generator_ema": generator_ema.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "generator_losses": generator_losses,
            "discriminator_losses": discriminator_losses,
            "latent_dim": args.latent_dim,
            "args": vars(args),
        },
        path
    )


# ============================================================
# TRAINING
# ============================================================

def train(args):

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("COMP3710 DEMO 2 - PART 4 TASK 3")
    print("OASIS GAN TRAINING")
    print("=" * 70)

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    dataset = OASISDataset(
        Path(args.data_dir)
    )

    print(
        "Training images:",
        len(dataset)
    )

    print(
        "Image shape:",
        dataset[0].shape
    )

    print(
        "Image range:",
        float(dataset[0].min()),
        "to",
        float(dataset[0].max())
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(
            args.num_workers > 0
        )
    )

    # Save real MRI examples for comparison.
    real_batch = next(iter(loader))

    save_image(
        (real_batch[:16] + 1.0) / 2.0,
        str(
            output_dir /
            "real_oasis_samples.png"
        ),
        nrow=4
    )

    generator = Generator(
        latent_dim=args.latent_dim
    ).to(device)

    discriminator = Discriminator().to(device)

    generator.apply(initialise_weights)
    discriminator.apply(initialise_weights)

    # EMA copy normally gives visually more stable samples.
    generator_ema = copy.deepcopy(
        generator
    ).eval()

    for parameter in generator_ema.parameters():
        parameter.requires_grad_(False)

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(0.0, 0.9)
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(0.0, 0.9)
    )

    fixed_noise = torch.randn(
        16,
        args.latent_dim,
        device=device
    )

    generator_losses = []
    discriminator_losses = []
    completed_epochs = []

    print()
    print(
        "Generator parameters:",
        sum(
            p.numel()
            for p in generator.parameters()
        )
    )

    print(
        "Discriminator parameters:",
        sum(
            p.numel()
            for p in discriminator.parameters()
        )
    )

    print()
    print(
        f"Starting {args.epochs} epochs..."
    )

    total_start = time.time()

    for epoch in range(
        1,
        args.epochs + 1
    ):

        generator.train()
        discriminator.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        epoch_start = time.time()

        for batch_index, real_images in enumerate(loader):

            real_images = real_images.to(
                device,
                non_blocking=True
            )

            batch_size = real_images.size(0)


            # ====================================================
            # TRAIN DISCRIMINATOR
            # ====================================================

            optimizer_d.zero_grad(
                set_to_none=True
            )

            noise = torch.randn(
                batch_size,
                args.latent_dim,
                device=device
            )

            with torch.no_grad():
                fake_images = generator(noise)

            real_scores = discriminator(
                real_images
            )

            fake_scores = discriminator(
                fake_images
            )

            # Hinge adversarial loss.
            discriminator_loss = (
                F.relu(
                    1.0 - real_scores
                ).mean()
                +
                F.relu(
                    1.0 + fake_scores
                ).mean()
            )

            if not torch.isfinite(
                discriminator_loss
            ):
                raise RuntimeError(
                    "Discriminator loss became non-finite."
                )

            discriminator_loss.backward()

            optimizer_d.step()


            # ====================================================
            # TRAIN GENERATOR
            # ====================================================

            optimizer_g.zero_grad(
                set_to_none=True
            )

            noise = torch.randn(
                batch_size,
                args.latent_dim,
                device=device
            )

            generated_images = generator(
                noise
            )

            generated_scores = discriminator(
                generated_images
            )

            generator_loss = (
                -generated_scores.mean()
            )

            if not torch.isfinite(
                generator_loss
            ):
                raise RuntimeError(
                    "Generator loss became non-finite."
                )

            generator_loss.backward()

            optimizer_g.step()

            update_ema(
                generator_ema,
                generator,
                decay=args.ema_decay
            )

            epoch_g_loss += (
                generator_loss.item()
            )

            epoch_d_loss += (
                discriminator_loss.item()
            )

            if (
                batch_index == 0
                or
                (batch_index + 1) % 100 == 0
            ):
                print(
                    f"Epoch {epoch:03d}/{args.epochs:03d} "
                    f"Batch {batch_index + 1:03d}/{len(loader):03d} "
                    f"D={discriminator_loss.item():.4f} "
                    f"G={generator_loss.item():.4f}"
                )

        mean_g_loss = (
            epoch_g_loss /
            len(loader)
        )

        mean_d_loss = (
            epoch_d_loss /
            len(loader)
        )

        generator_losses.append(
            mean_g_loss
        )

        discriminator_losses.append(
            mean_d_loss
        )

        completed_epochs.append(epoch)

        epoch_seconds = (
            time.time() - epoch_start
        )

        print(
            f"\nEpoch {epoch:03d} complete | "
            f"G loss={mean_g_loss:.4f} | "
            f"D loss={mean_d_loss:.4f} | "
            f"time={epoch_seconds:.1f}s"
        )

        # Save evidence at epoch 1 and every 5 epochs.
        if (
            epoch == 1
            or
            epoch % 5 == 0
            or
            epoch == args.epochs
        ):

            sample_path = (
                output_dir /
                f"generated_epoch_{epoch:03d}.png"
            )

            diversity = save_generated_samples(
                generator_ema,
                fixed_noise,
                sample_path
            )

            print(
                f"Generated samples saved: "
                f"{sample_path}"
            )

            print(
                f"Generated-sample diversity: "
                f"{diversity:.6f}"
            )

            save_loss_plot(
                completed_epochs,
                generator_losses,
                discriminator_losses,
                output_dir /
                "gan_training_losses.png"
            )

            save_checkpoint(
                output_dir /
                "gan_oasis_latest.pt",
                epoch,
                generator,
                generator_ema,
                discriminator,
                optimizer_g,
                optimizer_d,
                generator_losses,
                discriminator_losses,
                args,
            )

        print()

    total_seconds = (
        time.time() - total_start
    )

    final_checkpoint = (
        output_dir /
        "gan_oasis_final.pt"
    )

    save_checkpoint(
        final_checkpoint,
        args.epochs,
        generator,
        generator_ema,
        discriminator,
        optimizer_g,
        optimizer_d,
        generator_losses,
        discriminator_losses,
        args,
    )

    final_diversity = save_generated_samples(
        generator_ema,
        torch.randn(
            16,
            args.latent_dim,
            device=device
        ),
        output_dir /
        "gan_final_generated_brains.png"
    )

    save_loss_plot(
        completed_epochs,
        generator_losses,
        discriminator_losses,
        output_dir /
        "gan_training_losses.png"
    )

    print("=" * 70)
    print("GAN TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Training time:",
        f"{total_seconds / 60.0:.2f} minutes"
    )

    print(
        "Final generator loss:",
        f"{generator_losses[-1]:.6f}"
    )

    print(
        "Final discriminator loss:",
        f"{discriminator_losses[-1]:.6f}"
    )

    print(
        "Final sample diversity:",
        f"{final_diversity:.6f}"
    )

    print(
        "Final checkpoint:",
        final_checkpoint
    )

    print(
        "Final generated brains:",
        output_dir /
        "gan_final_generated_brains.png"
    )

    print(
        "Loss plot:",
        output_dir /
        "gan_training_losses.png"
    )


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR)
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR)
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4
    )

    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=3710
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    train(args)
