from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from vae_oasis import VAE, LATENT_DIM


# ============================================================
# COMP3710 DEMO 2 - PART 4, TASK 1
# VAE 2D LATENT MANIFOLD VISUALISATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

CHECKPOINT_PATH = SCRIPT_DIR / "vae_oasis_best.pt"
OUTPUT_PATH = SCRIPT_DIR / "vae_manifold.png"

GRID_SIZE = 12
LATENT_MIN = -3.0
LATENT_MAX = 3.0


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # Load trained VAE
    # --------------------------------------------------------

    model = VAE(
        latent_dim=LATENT_DIM
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Loaded checkpoint from epoch:",
        checkpoint["epoch"]
    )

    # --------------------------------------------------------
    # Create a regular 2D latent grid
    # --------------------------------------------------------

    values = torch.linspace(
        LATENT_MIN,
        LATENT_MAX,
        GRID_SIZE
    )

    latent_points = []

    for y in values.flip(0):
        for x in values:

            latent_points.append(
                torch.tensor(
                    [x, y],
                    dtype=torch.float32
                )
            )

    latent_points = torch.stack(
        latent_points
    ).to(device)

    # --------------------------------------------------------
    # Decode latent coordinates into MRI slices
    # --------------------------------------------------------

    with torch.no_grad():

        generated = model.decode(
            latent_points
        ).cpu()

    # --------------------------------------------------------
    # Display decoded images as a 2D manifold
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        GRID_SIZE,
        GRID_SIZE,
        figsize=(12, 12)
    )

    for index, axis in enumerate(
        axes.flat
    ):

        axis.imshow(
            generated[index, 0],
            cmap="gray",
            vmin=0.0,
            vmax=1.0
        )

        axis.axis("off")

    plt.suptitle(
        "OASIS VAE 2D Latent Manifold"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Manifold saved:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
