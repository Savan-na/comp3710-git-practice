from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image


# ============================================================
# COMP3710 DEMO 2 - PART 4, TASK 1
# VARIATIONAL AUTOENCODER (VAE) FOR OASIS BRAIN MRI
# ============================================================
#
# Current progress:
#   1. OASIS dataset + DataLoaders                       [DONE]
#   2. Convolutional VAE with 2D latent space            [DONE]
#   3. Shape validation                                  [DONE]
#   4. VAE loss = reconstruction BCE + KL divergence     [DONE]
#   5. Full training + validation + best checkpoint      [CURRENT]
#
# Still required later for Task 1 full marks:
#   6. MRI reconstruction visualisation
#   7. 2D latent manifold visualisation
# ============================================================


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

# Official COMP3710 OASIS location on Rangpur.
DATA_ROOT = Path("/home/groups/comp3710/OASIS")

TRAIN_DIR = DATA_ROOT / "keras_png_slices_train"
VAL_DIR = DATA_ROOT / "keras_png_slices_validate"
TEST_DIR = DATA_ROOT / "keras_png_slices_test"

# Save outputs beside this Python file.
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_CHECKPOINT = SCRIPT_DIR / "vae_oasis_pipeline_test.pt"
BEST_CHECKPOINT = SCRIPT_DIR / "vae_oasis_best.pt"

# OASIS MRI slices.
IMAGE_SIZE = 256
IMAGE_CHANNELS = 1

# DataLoader settings.
BATCH_SIZE = 64
NUM_WORKERS = 1

# A 2D latent space lets us visualise the manifold directly later.
LATENT_DIM = 2

# Shape test only needs a few samples.
SHAPE_TEST_BATCH_SIZE = 4

# Training choices.
# These are implementation choices, not official course requirements.
LEARNING_RATE = 1e-3
FULL_TRAINING_EPOCHS = 20

# Safe default:
#   "pipeline_test" -> 1 epoch, 2 train batches, 2 validation batches
#   "full_train"    -> complete training/validation sets
RUN_MODE = "full_train"


# ============================================================
# DATASET
# ============================================================

class OASISDataset(Dataset):
    """Dataset for preprocessed 2D OASIS MRI slices.

    Returns each MRI as:
        shape: (1, 256, 256)
        dtype: torch.float32
        range: [0, 1]
    """

    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)

        # What: ensure the requested directory exists.
        # Why: fail clearly instead of silently creating an empty dataset.
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"OASIS directory not found: {self.image_dir}"
            )

        # What: store PNG files in a deterministic order.
        # Why: stable ordering helps reproducible inspection/evaluation.
        self.image_paths = sorted(self.image_dir.glob("*.png"))

        if not self.image_paths:
            raise RuntimeError(
                f"No PNG images found in: {self.image_dir}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        # What: load as one-channel grayscale.
        # Why: OASIS MRI input is 1 x 256 x 256, not RGB.
        image = read_image(
            str(image_path),
            mode=ImageReadMode.GRAY
        )

        # What: convert uint8 [0,255] -> float32 [0,1].
        # Why: matches the sigmoid decoder output range.
        image = image.float() / 255.0

        expected_shape = (
            IMAGE_CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE
        )

        if tuple(image.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected image shape {tuple(image.shape)} "
                f"for {image_path}; expected {expected_shape}"
            )

        return image


# ============================================================
# DATA LOADERS
# ============================================================

def build_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
):
    """Create OASIS train, validation and test DataLoaders."""

    train_dataset = OASISDataset(TRAIN_DIR)
    val_dataset = OASISDataset(VAL_DIR)
    test_dataset = OASISDataset(TEST_DIR)

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_options
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_options
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ============================================================
# VARIATIONAL AUTOENCODER
# ============================================================

class VAE(nn.Module):
    """Convolutional VAE for 256x256 grayscale OASIS MRI slices."""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder:
        # 1x256x256 -> 32x128x128 -> 64x64x64
        # -> 128x32x32 -> 256x16x16
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.encoded_channels = 256
        self.encoded_size = 16

        self.feature_size = (
            self.encoded_channels
            * self.encoded_size
            * self.encoded_size
        )

        # A VAE predicts the latent distribution parameters q(z|x).
        self.fc_mu = nn.Linear(
            self.feature_size,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            self.feature_size,
            latent_dim
        )

        # Map z back to the convolutional decoder feature map.
        self.decoder_input = nn.Linear(
            latent_dim,
            self.feature_size
        )

        # Decoder:
        # 256x16x16 -> 128x32x32 -> 64x64x64
        # -> 32x128x128 -> 1x256x256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32, 1,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            # Input pixels are in [0,1], so reconstruction is too.
            nn.Sigmoid(),
        )

    def encode(self, x):
        """Encode MRI into mu and log-variance."""

        features = self.encoder(x)
        features = torch.flatten(features, start_dim=1)

        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample z = mu + sigma*epsilon, epsilon ~ N(0,1)."""

        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std

        return z

    def decode(self, z):
        """Decode latent vectors into MRI reconstructions."""

        features = self.decoder_input(z)

        features = features.view(
            -1,
            self.encoded_channels,
            self.encoded_size,
            self.encoded_size
        )

        return self.decoder(features)

    def forward(self, x):
        """Complete VAE forward pass."""

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)

        return reconstruction, mu, logvar


# ============================================================
# VAE LOSS
# ============================================================

def vae_loss(
    reconstruction,
    target,
    mu,
    logvar
):
    """Return total, reconstruction and KL losses per image."""

    batch_size = target.size(0)

    # Reconstruction term.
    # reduction="sum" sums all pixels in the batch; dividing by
    # batch_size gives the mean reconstruction loss per MRI slice.
    reconstruction_loss = F.binary_cross_entropy(
        reconstruction,
        target,
        reduction="sum"
    ) / batch_size

    # KL divergence from q(z|x) = N(mu,sigma^2)
    # to the standard normal prior p(z) = N(0,1).
    kl_loss = (
        -0.5
        * torch.sum(
            1
            + logvar
            - mu.pow(2)
            - logvar.exp()
        )
        / batch_size
    )

    total_loss = reconstruction_loss + kl_loss

    return (
        total_loss,
        reconstruction_loss,
        kl_loss
    )


# ============================================================
# ONE TRAINING EPOCH
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    max_batches=None
):
    """Train one epoch. max_batches=None means full DataLoader."""

    model.train()

    total_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    kl_loss_sum = 0.0
    total_images = 0

    for batch_index, images in enumerate(train_loader):

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        images = images.to(
            device,
            non_blocking=True
        )

        batch_size = images.size(0)

        optimizer.zero_grad(
            set_to_none=True
        )

        reconstruction, mu, logvar = model(images)

        (
            total_loss,
            reconstruction_loss,
            kl_loss
        ) = vae_loss(
            reconstruction,
            images,
            mu,
            logvar
        )

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                "Training loss became non-finite."
            )

        total_loss.backward()
        optimizer.step()

        # Weight each batch mean by its actual number of images.
        total_loss_sum += total_loss.item() * batch_size
        reconstruction_loss_sum += (
            reconstruction_loss.item() * batch_size
        )
        kl_loss_sum += kl_loss.item() * batch_size
        total_images += batch_size

    if total_images == 0:
        raise RuntimeError(
            "No training images were processed."
        )

    return (
        total_loss_sum / total_images,
        reconstruction_loss_sum / total_images,
        kl_loss_sum / total_images
    )


# ============================================================
# VALIDATION / EVALUATION
# ============================================================

def evaluate(
    model,
    data_loader,
    device,
    max_batches=None
):
    """Evaluate without gradients or parameter updates."""

    model.eval()

    total_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    kl_loss_sum = 0.0
    total_images = 0

    with torch.no_grad():

        for batch_index, images in enumerate(data_loader):

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

            images = images.to(
                device,
                non_blocking=True
            )

            batch_size = images.size(0)

            reconstruction, mu, logvar = model(images)

            (
                total_loss,
                reconstruction_loss,
                kl_loss
            ) = vae_loss(
                reconstruction,
                images,
                mu,
                logvar
            )

            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    "Evaluation loss became non-finite."
                )

            total_loss_sum += total_loss.item() * batch_size
            reconstruction_loss_sum += (
                reconstruction_loss.item() * batch_size
            )
            kl_loss_sum += kl_loss.item() * batch_size
            total_images += batch_size

    if total_images == 0:
        raise RuntimeError(
            "No evaluation images were processed."
        )

    return (
        total_loss_sum / total_images,
        reconstruction_loss_sum / total_images,
        kl_loss_sum / total_images
    )


# ============================================================
# TRAINING CONTROLLER
# ============================================================

def train_vae(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    learning_rate,
    checkpoint_path,
    max_train_batches=None,
    max_val_batches=None
):
    """Train VAE, log losses and save best validation checkpoint."""

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    best_val_loss = float("inf")

    history = {
        "train_total": [],
        "train_reconstruction": [],
        "train_kl": [],
        "val_total": [],
        "val_reconstruction": [],
        "val_kl": [],
    }

    for epoch in range(1, epochs + 1):

        (
            train_total,
            train_reconstruction,
            train_kl
        ) = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches
        )

        (
            val_total,
            val_reconstruction,
            val_kl
        ) = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            max_batches=max_val_batches
        )

        history["train_total"].append(train_total)
        history["train_reconstruction"].append(
            train_reconstruction
        )
        history["train_kl"].append(train_kl)

        history["val_total"].append(val_total)
        history["val_reconstruction"].append(
            val_reconstruction
        )
        history["val_kl"].append(val_kl)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Train total={train_total:.4f} | "
            f"Train recon={train_reconstruction:.4f} | "
            f"Train KL={train_kl:.4f} | "
            f"Val total={val_total:.4f} | "
            f"Val recon={val_reconstruction:.4f} | "
            f"Val KL={val_kl:.4f}"
        )

        # Save the best observed validation model, not simply the
        # final epoch.
        if val_total < best_val_loss:

            best_val_loss = val_total

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "validation_loss":
                        best_val_loss,
                    "history":
                        history,
                    "latent_dim":
                        model.latent_dim,
                },
                checkpoint_path
            )

            print(
                "Best checkpoint saved:",
                checkpoint_path
            )

    return history


# ============================================================
# DATASET INSPECTION
# ============================================================

def inspect_dataset(
    train_loader,
    train_dataset,
    val_dataset,
    test_dataset
):
    """Print basic dataset and tensor information."""

    images = next(iter(train_loader))

    print("\n--- OASIS Dataset ---")
    print("Training slices:", len(train_dataset))
    print("Validation slices:", len(val_dataset))
    print("Testing slices:", len(test_dataset))

    print("\n--- First Training Batch ---")
    print("Batch shape:", images.shape)
    print("dtype:", images.dtype)
    print("Pixel minimum:", images.min().item())
    print("Pixel maximum:", images.max().item())

    return images


# ============================================================
# VAE ARCHITECTURE TEST
# ============================================================

def test_vae_shapes(
    model,
    images,
    device
):
    """Verify encoder, latent and decoder tensor dimensions."""

    model.eval()

    images = images[
        :SHAPE_TEST_BATCH_SIZE
    ].to(device)

    with torch.no_grad():
        reconstruction, mu, logvar = model(images)

    print("\n--- VAE Shape Test ---")
    print("Input shape:", images.shape)
    print("Reconstruction shape:", reconstruction.shape)
    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)

    assert reconstruction.shape == images.shape

    assert mu.shape == (
        images.size(0),
        LATENT_DIM
    )

    assert logvar.shape == (
        images.size(0),
        LATENT_DIM
    )

    print("VAE shape test: PASSED")


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n--- Device ---")
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    ) = build_dataloaders()

    # The test loader is intentionally prepared now.
    # It will be used in the reconstruction/manifold stage.
    _ = test_loader

    images = inspect_dataset(
        train_loader,
        train_dataset,
        val_dataset,
        test_dataset
    )

    model = VAE(
        latent_dim=LATENT_DIM
    ).to(device)

    test_vae_shapes(
        model,
        images,
        device
    )

    if RUN_MODE == "pipeline_test":

        print(
            "\n--- VAE Pipeline Test ---"
        )
        print(
            "1 epoch, 2 train batches, "
            "2 validation batches"
        )

        train_vae(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=1,
            learning_rate=LEARNING_RATE,
            checkpoint_path=PIPELINE_CHECKPOINT,
            max_train_batches=2,
            max_val_batches=2
        )

        print("\nPipeline test completed.")

    elif RUN_MODE == "full_train":

        print("\n--- Full VAE Training ---")

        train_vae(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=FULL_TRAINING_EPOCHS,
            learning_rate=LEARNING_RATE,
            checkpoint_path=BEST_CHECKPOINT,
            max_train_batches=None,
            max_val_batches=None
        )

        print("\nFull training completed.")

    else:
        raise ValueError(
            "RUN_MODE must be "
            "'pipeline_test' or 'full_train'."
        )


if __name__ == "__main__":
    main()
