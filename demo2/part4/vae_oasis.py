from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image


# ============================================================
# COMP3710 DEMO 2 - PART 4, TASK 1
# VARIATIONAL AUTOENCODER (VAE) FOR OASIS BRAIN MRI
# ============================================================
#
# Current stage:
#   1. Load the preprocessed OASIS MRI dataset.
#   2. Construct train / validation / test DataLoaders.
#   3. Implement a convolutional Variational Autoencoder.
#   4. Verify all tensor shapes with a forward pass.
#
# Training, VAE loss, reconstruction visualisation and
# latent-manifold visualisation are added in later stages.
# ============================================================


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

# Official COMP3710 OASIS dataset location on Rangpur.
DATA_ROOT = Path("/home/groups/comp3710/OASIS")

TRAIN_DIR = DATA_ROOT / "keras_png_slices_train"
VAL_DIR = DATA_ROOT / "keras_png_slices_validate"
TEST_DIR = DATA_ROOT / "keras_png_slices_test"

# OASIS images are preprocessed 256x256 grayscale PNG slices.
IMAGE_SIZE = 256
IMAGE_CHANNELS = 1

# Mini-batch size reserved for later training.
BATCH_SIZE = 64

# One worker is conservative and appropriate for Rangpur testing.
NUM_WORKERS = 1

# A 2D latent space allows direct visualisation of the VAE manifold.
LATENT_DIM = 2

# Use only a few images for the architecture sanity check.
SHAPE_TEST_BATCH_SIZE = 4


# ============================================================
# DATASET
# ============================================================

class OASISDataset(Dataset):
    """Dataset for preprocessed 2D OASIS MRI slices.

    Each item is returned as a PyTorch tensor with:

        shape: (1, 256, 256)
        dtype: torch.float32
        range: [0, 1]

    Parameters
    ----------
    image_dir:
        Directory containing the preprocessed MRI PNG files.
    """

    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)

        # What:
        # Ensure the requested dataset directory exists.
        #
        # Why:
        # A clear error here is easier to diagnose than silently
        # constructing an empty dataset.
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"OASIS directory not found: {self.image_dir}"
            )

        # What:
        # Store all MRI files in a deterministic order.
        #
        # Why:
        # Stable ordering makes inspection and evaluation reproducible.
        self.image_paths = sorted(
            self.image_dir.glob("*.png")
        )

        if not self.image_paths:
            raise RuntimeError(
                f"No PNG images found in: {self.image_dir}"
            )

    def __len__(self):
        """Return the number of MRI slices."""

        return len(self.image_paths)

    def __getitem__(self, index):
        """Load and preprocess one MRI slice."""

        image_path = self.image_paths[index]

        # What:
        # Load the MRI explicitly as a one-channel grayscale image.
        #
        # Why:
        # The OASIS input is grayscale, so the model input shape is
        # (1, 256, 256), not three-channel RGB.
        image = read_image(
            str(image_path),
            mode=ImageReadMode.GRAY
        )

        # What:
        # Convert uint8 pixel values from [0, 255] to float32 [0, 1].
        #
        # Why:
        # This provides a stable numerical range and matches the
        # sigmoid output used by the VAE decoder.
        image = image.float() / 255.0

        return image


# ============================================================
# DATA LOADERS
# ============================================================

def build_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
):
    """Create the official OASIS train, validation and test loaders."""

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
    """Convolutional VAE for 256x256 grayscale OASIS MRI slices.

    Input
    -----
    Tensor:
        (batch_size, 1, 256, 256)

    Latent representation
    ---------------------
    mu:
        (batch_size, latent_dim)

    logvar:
        (batch_size, latent_dim)

    Output
    ------
    Reconstructed MRI:
        (batch_size, 1, 256, 256)
    """

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        self.latent_dim = latent_dim

        # ========================================================
        # ENCODER
        #
        # 1 x 256 x 256
        #       ↓
        # 32 x 128 x 128
        #       ↓
        # 64 x 64 x 64
        #       ↓
        # 128 x 32 x 32
        #       ↓
        # 256 x 16 x 16
        # ========================================================

        self.encoder = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),
        )

        self.encoded_channels = 256
        self.encoded_size = 16

        self.feature_size = (
            self.encoded_channels
            * self.encoded_size
            * self.encoded_size
        )

        # ★ CORE:
        # Unlike a normal autoencoder, a VAE predicts the
        # parameters of a latent probability distribution.
        self.fc_mu = nn.Linear(
            self.feature_size,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            self.feature_size,
            latent_dim
        )

        # ========================================================
        # LATENT VECTOR -> DECODER FEATURE MAP
        # ========================================================

        self.decoder_input = nn.Linear(
            latent_dim,
            self.feature_size
        )

        # ========================================================
        # DECODER
        #
        # 256 x 16 x 16
        #       ↓
        # 128 x 32 x 32
        #       ↓
        # 64 x 64 x 64
        #       ↓
        # 32 x 128 x 128
        #       ↓
        # 1 x 256 x 256
        # ========================================================

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32,
                1,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            # What:
            # Restrict reconstructed pixels to [0, 1].
            #
            # Why:
            # The input MRI tensors are also normalized to [0, 1].
            nn.Sigmoid(),
        )

    def encode(self, x):
        """Encode input MRI into latent distribution parameters."""

        features = self.encoder(x)

        features = torch.flatten(
            features,
            start_dim=1
        )

        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample latent z using the reparameterisation trick.

        The VAE represents:

            q(z | x) = N(mu, sigma^2)

        Instead of sampling z directly, use:

            z = mu + sigma * epsilon

        where:

            epsilon ~ N(0, 1)

        This keeps the sampling operation differentiable with
        respect to the network parameters.
        """

        std = torch.exp(
            0.5 * logvar
        )

        epsilon = torch.randn_like(std)

        z = mu + epsilon * std

        return z

    def decode(self, z):
        """Decode latent vectors into reconstructed MRI slices."""

        features = self.decoder_input(z)

        features = features.view(
            -1,
            self.encoded_channels,
            self.encoded_size,
            self.encoded_size
        )

        reconstruction = self.decoder(
            features
        )

        return reconstruction

    def forward(self, x):
        """Run the complete VAE forward pass."""

        mu, logvar = self.encode(x)

        z = self.reparameterize(
            mu,
            logvar
        )

        reconstruction = self.decode(z)

        return reconstruction, mu, logvar


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

    images = next(
        iter(train_loader)
    )

    print("\n--- OASIS Dataset ---")

    print(
        "Training slices:",
        len(train_dataset)
    )

    print(
        "Validation slices:",
        len(val_dataset)
    )

    print(
        "Testing slices:",
        len(test_dataset)
    )

    print("\n--- First Training Batch ---")

    print(
        "Batch shape:",
        images.shape
    )

    print(
        "dtype:",
        images.dtype
    )

    print(
        "Pixel minimum:",
        images.min().item()
    )

    print(
        "Pixel maximum:",
        images.max().item()
    )

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

    # Only a few images are needed to test the architecture.
    images = images[
        :SHAPE_TEST_BATCH_SIZE
    ].to(device)

    with torch.no_grad():

        reconstruction, mu, logvar = model(
            images
        )

    print("\n--- VAE Shape Test ---")

    print(
        "Input shape:",
        images.shape
    )

    print(
        "Reconstruction shape:",
        reconstruction.shape
    )

    print(
        "mu shape:",
        mu.shape
    )

    print(
        "logvar shape:",
        logvar.shape
    )

    # Shape assertions make architectural mistakes fail clearly.
    assert reconstruction.shape == images.shape

    assert mu.shape == (
        images.size(0),
        LATENT_DIM
    )

    assert logvar.shape == (
        images.size(0),
        LATENT_DIM
    )

    print(
        "VAE shape test: PASSED"
    )


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


if __name__ == "__main__":
    main()