from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode


# ============================================================
# COMP3710 DEMO 2 - PART 4, TASK 2
# UNET FOR OASIS BRAIN MRI SEGMENTATION
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path("/home/groups/comp3710/OASIS")

TRAIN_IMAGE_DIR = DATA_ROOT / "keras_png_slices_train"
TRAIN_MASK_DIR = DATA_ROOT / "keras_png_slices_seg_train"

VAL_IMAGE_DIR = DATA_ROOT / "keras_png_slices_validate"
VAL_MASK_DIR = DATA_ROOT / "keras_png_slices_seg_validate"

TEST_IMAGE_DIR = DATA_ROOT / "keras_png_slices_test"
TEST_MASK_DIR = DATA_ROOT / "keras_png_slices_seg_test"

IMAGE_SIZE = 256
NUM_CLASSES = 4

RAW_LABELS = [0, 85, 170, 255]

BATCH_SIZE = 16
NUM_WORKERS = 1

LEARNING_RATE = 1e-3
MAX_EPOCHS = 20

TARGET_DSC = 0.90

CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "unet_oasis_best.pt"
)


# ============================================================
# DATASET
# ============================================================

class OASISSegmentationDataset(Dataset):
    """Paired OASIS MRI and segmentation-mask dataset."""

    def __init__(self, image_dir, mask_dir):

        self.image_paths = sorted(
            Path(image_dir).glob("*.png")
        )

        self.mask_paths = sorted(
            Path(mask_dir).glob("*.png")
        )

        if len(self.image_paths) != len(self.mask_paths):
            raise RuntimeError(
                "MRI and segmentation-mask counts do not match."
            )

        if not self.image_paths:
            raise RuntimeError(
                f"No MRI images found in {image_dir}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image = read_image(
            str(self.image_paths[index]),
            mode=ImageReadMode.GRAY
        )

        image = image.float() / 255.0

        mask = read_image(
            str(self.mask_paths[index]),
            mode=ImageReadMode.GRAY
        ).squeeze(0)

        # Raw labels:
        # 0, 85, 170, 255
        #
        # Categorical labels:
        # 0, 1, 2, 3
        class_mask = torch.empty_like(
            mask,
            dtype=torch.long
        )

        class_mask[mask == 0] = 0
        class_mask[mask == 85] = 1
        class_mask[mask == 170] = 2
        class_mask[mask == 255] = 3

        return image, class_mask


# ============================================================
# DATA LOADERS
# ============================================================

def build_dataloaders():

    train_dataset = OASISSegmentationDataset(
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR
    )

    val_dataset = OASISSegmentationDataset(
        VAL_IMAGE_DIR,
        VAL_MASK_DIR
    )

    test_dataset = OASISSegmentationDataset(
        TEST_IMAGE_DIR,
        TEST_MASK_DIR
    )

    options = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available()
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **options
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **options
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **options
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset
    )


# ============================================================
# UNET BLOCK
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# UNET
# ============================================================

class UNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        num_classes=NUM_CLASSES
    ):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(
            in_channels,
            32
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(
            32,
            64
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(
            64,
            128
        )
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(
            128,
            256
        )
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(
            256,
            512
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )
        self.dec4 = DoubleConv(
            512,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )
        self.dec3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )
        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )
        self.dec1 = DoubleConv(
            64,
            32
        )

        # Four categorical segmentation classes.
        self.output = nn.Conv2d(
            32,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        e4 = self.enc4(
            self.pool3(e3)
        )

        b = self.bottleneck(
            self.pool4(e4)
        )

        d4 = self.up4(b)
        d4 = torch.cat(
            [d4, e4],
            dim=1
        )
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat(
            [d3, e3],
            dim=1
        )
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat(
            [d2, e2],
            dim=1
        )
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat(
            [d1, e1],
            dim=1
        )
        d1 = self.dec1(d1)

        logits = self.output(d1)

        return logits


# ============================================================
# SOFT DICE LOSS
# ============================================================

def soft_dice_loss(
    logits,
    targets,
    num_classes=NUM_CLASSES,
    epsilon=1e-6
):
    """Differentiable Dice loss using one-hot targets."""

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    one_hot_targets = F.one_hot(
        targets,
        num_classes=num_classes
    )

    one_hot_targets = one_hot_targets.permute(
        0,
        3,
        1,
        2
    ).float()

    intersection = torch.sum(
        probabilities * one_hot_targets,
        dim=(0, 2, 3)
    )

    denominator = (
        torch.sum(
            probabilities,
            dim=(0, 2, 3)
        )
        +
        torch.sum(
            one_hot_targets,
            dim=(0, 2, 3)
        )
    )

    dice = (
        2.0 * intersection + epsilon
    ) / (
        denominator + epsilon
    )

    return 1.0 - dice.mean()


# ============================================================
# TRAINING LOSS
# ============================================================

def segmentation_loss(
    logits,
    targets
):

    ce_loss = F.cross_entropy(
        logits,
        targets
    )

    dice_loss = soft_dice_loss(
        logits,
        targets
    )

    total_loss = (
        ce_loss
        + dice_loss
    )

    return total_loss


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, masks in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = segmentation_loss(
            logits,
            masks
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Training loss became non-finite."
            )

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    return total_loss / total_samples


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    data_loader,
    device
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    intersections = torch.zeros(
        NUM_CLASSES,
        device=device
    )

    denominators = torch.zeros(
        NUM_CLASSES,
        device=device
    )

    with torch.no_grad():

        for images, masks in data_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            logits = model(images)

            loss = segmentation_loss(
                logits,
                masks
            )

            batch_size = images.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            predictions = torch.argmax(
                logits,
                dim=1
            )

            for class_index in range(
                NUM_CLASSES
            ):

                pred_class = (
                    predictions
                    == class_index
                )

                target_class = (
                    masks
                    == class_index
                )

                intersections[class_index] += (
                    pred_class
                    & target_class
                ).sum()

                denominators[class_index] += (
                    pred_class.sum()
                    + target_class.sum()
                )

    dice_scores = (
        2.0 * intersections + 1e-6
    ) / (
        denominators + 1e-6
    )

    return (
        total_loss / total_samples,
        dice_scores.cpu().tolist()
    )


# ============================================================
# FULL TRAINING
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    device
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_mean_dice = -1.0

    print("\n--- Full UNet Training ---")

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device
        )

        (
            val_loss,
            val_dice
        ) = evaluate(
            model,
            val_loader,
            device
        )

        mean_dice = sum(
            val_dice
        ) / NUM_CLASSES

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS:02d} | "
            f"Train loss={train_loss:.4f} | "
            f"Val loss={val_loss:.4f} | "
            f"DSC="
            f"[{val_dice[0]:.4f}, "
            f"{val_dice[1]:.4f}, "
            f"{val_dice[2]:.4f}, "
            f"{val_dice[3]:.4f}] | "
            f"Mean={mean_dice:.4f}"
        )

        # Save best validation model.
        if mean_dice > best_mean_dice:

            best_mean_dice = mean_dice

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "validation_dice":
                        val_dice,
                    "mean_dice":
                        mean_dice
                },
                CHECKPOINT_PATH
            )

            print(
                "Best checkpoint saved:",
                CHECKPOINT_PATH
            )

        # Course target:
        # DSC > 0.90 for every segmentation label.
        if min(val_dice) > TARGET_DSC:

            print(
                "\nTarget reached: "
                "all validation DSC scores > 0.90"
            )

            break


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
        test_dataset
    ) = build_dataloaders()

    print("\n--- OASIS Segmentation Dataset ---")

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    print(
        "Testing samples:",
        len(test_dataset)
    )

    model = UNet(
        in_channels=1,
        num_classes=NUM_CLASSES
    ).to(device)

    train_model(
        model,
        train_loader,
        val_loader,
        device
    )


if __name__ == "__main__":
    main()
