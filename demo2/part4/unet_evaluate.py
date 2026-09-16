from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from unet_oasis import (
    UNet,
    NUM_CLASSES,
    build_dataloaders,
    evaluate
)


SCRIPT_DIR = Path(__file__).resolve().parent

CHECKPOINT_PATH = SCRIPT_DIR / "unet_oasis_best.pt"
OUTPUT_PATH = SCRIPT_DIR / "unet_test_prediction.png"


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset
    ) = build_dataloaders()

    # --------------------------------------------------------
    # Load trained UNet
    # --------------------------------------------------------

    model = UNet(
        in_channels=1,
        num_classes=NUM_CLASSES
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

    print(
        "Best validation DSC:",
        checkpoint["validation_dice"]
    )

    # --------------------------------------------------------
    # Evaluate complete test set
    # --------------------------------------------------------

    test_loss, test_dice = evaluate(
        model,
        test_loader,
        device
    )

    print("\n--- Test Results ---")
    print(
        f"Test loss: {test_loss:.4f}"
    )

    for class_index, score in enumerate(
        test_dice
    ):
        print(
            f"Class {class_index} DSC: "
            f"{score:.4f}"
        )

    # --------------------------------------------------------
    # Run inference on one test MRI for visualisation
    # --------------------------------------------------------

    images, masks = next(
        iter(test_loader)
    )

    image = images[:1].to(device)
    target = masks[:1].to(device)

    with torch.no_grad():

        logits = model(image)

        prediction = torch.argmax(
            logits,
            dim=1
        )

    image = image[0, 0].cpu()
    target = target[0].cpu()
    prediction = prediction[0].cpu()

    # --------------------------------------------------------
    # Save MRI / ground truth / prediction
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )

    axes[0].imshow(
        image,
        cmap="gray"
    )
    axes[0].set_title(
        "Input MRI"
    )
    axes[0].axis("off")

    axes[1].imshow(
        target,
        cmap="viridis",
        vmin=0,
        vmax=NUM_CLASSES - 1
    )
    axes[1].set_title(
        "Ground Truth"
    )
    axes[1].axis("off")

    axes[2].imshow(
        prediction,
        cmap="viridis",
        vmin=0,
        vmax=NUM_CLASSES - 1
    )
    axes[2].set_title(
        "UNet Prediction"
    )
    axes[2].axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nSegmentation visualisation saved:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
