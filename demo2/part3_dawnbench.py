from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# COMP3710 Demo 2 - Part 3.2: DAWNBench Challenge
# CIFAR-10 classification using a self-implemented ResNet-18.
# Modes allow testing/evaluation without repeatedly retraining the full model.


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args():
    """Read command-line settings.

    mode:
        inspect -> verify dataset and batch shapes only.
        smoke   -> train/evaluate a few batches to test the pipeline.
        train   -> run full training and save the best checkpoint.
        eval    -> load the checkpoint and evaluate without retraining.
        demo    -> load the checkpoint, run inference, then train one epoch.
    """

    parser = argparse.ArgumentParser(
        description="COMP3710 DAWNBench CIFAR-10 ResNet-18"
    )

    parser.add_argument(
        "--mode",
        choices=("inspect", "smoke", "train", "eval", "demo"),
        default="inspect"
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA mixed precision."
    )

    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for more reproducible experiments.

    seed: integer random seed used by Python, NumPy and PyTorch.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(data_dir, batch_size, workers):
    """Create the official CIFAR-10 train/test DataLoaders.

    data_dir:
        Local folder where torchvision caches CIFAR-10.

    batch_size:
        Number of images in one mini-batch.

    workers:
        Number of background processes used for data loading.

    Returns:
        train_loader, test_loader, train_dataset, test_dataset
    """

    # [Added optimisation]
    # Random crop and horizontal flip increase training-data variation.
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Test images are normalized but never randomly augmented.
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # ★ CORE:
    # CIFAR-10 already provides an official 50,000-image training split.
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    # ★ CORE:
    # CIFAR-10 already provides an official 10,000-image test split.
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    loader_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        **loader_options
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **loader_options
    )

    return train_loader, test_loader, train_dataset, test_dataset


class BasicBlock(nn.Module):
    """Basic residual block used by ResNet-18.

    in_channels:
        Number of input feature-map channels.

    out_channels:
        Number of output feature-map channels.

    stride:
        Spatial step size. stride=2 reduces image height and width by half.
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        # ★ CORE:
        # Projection shortcut matches tensor shapes when channels/size change.
        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        out = self.bn2(
            self.conv2(out)
        )

        # ★ CORE:
        # Skip connection adds the original signal to the learned residual.
        out = self.relu(out + identity)

        return out


class ResNet18CIFAR(nn.Module):
    """Self-implemented ResNet-18 for CIFAR-10.

    Input:
        (batch_size, 3, 32, 32)

    Output:
        (batch_size, 10) raw class logits.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.in_channels = 64

        # CIFAR-10 uses a 3x3 stride-1 stem because images are only 32x32.
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ★ CORE:
        # ResNet-18 contains 2 + 2 + 2 + 2 residual blocks.
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._initialise_weights()

    def _make_layer(self, out_channels, blocks, stride):
        """Build one ResNet stage.

        out_channels:
            Number of output channels in this stage.

        blocks:
            Number of residual BasicBlocks.

        stride:
            Stride used by the first block.
        """

        layers = [
            BasicBlock(
                self.in_channels,
                out_channels,
                stride
            )
        ]

        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.in_channels,
                    out_channels,
                    stride=1
                )
            )

        return nn.Sequential(*layers)

    def _initialise_weights(self):
        """Initialise convolution and BatchNorm parameters."""

        for module in self.modules():

            if isinstance(module, nn.Conv2d):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

            elif isinstance(module, nn.BatchNorm2d):

                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        return self.fc(x)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    amp_enabled,
    max_batches=None
):
    """Train for one epoch.

    Returns:
        average_loss, accuracy, elapsed_seconds
    """

    model.train()

    loss_sum = 0.0
    correct = 0
    total = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for batch_index, (images, labels) in enumerate(loader):

        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        # ★ CORE:
        # Mixed precision uses FP16 where safe to reduce GPU compute/memory cost.
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled
        ):

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        loss_sum += loss.item() * labels.size(0)

        correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()

        total += labels.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return (
        loss_sum / total,
        correct / total,
        elapsed
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    amp_enabled,
    max_batches=None
):
    """Evaluate without updating weights.

    Returns:
        average_loss, accuracy, elapsed_seconds
    """

    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for batch_index, (images, labels) in enumerate(loader):

        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled
        ):

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

        loss_sum += loss.item() * labels.size(0)

        correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()

        total += labels.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return (
        loss_sum / total,
        correct / total,
        elapsed
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_accuracy
):
    """Save model and training state."""

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_accuracy": best_accuracy,
        },
        path,
    )


def load_checkpoint(
    path,
    model,
    device,
    optimizer=None,
    scheduler=None,
    scaler=None
):
    """Load a saved model checkpoint."""

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    if scheduler is not None:
        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

    if scaler is not None:
        scaler.load_state_dict(
            checkpoint["scaler"]
        )

    return checkpoint


def main():

    args = parse_args()

    set_seed(args.seed)

    project_dir = Path(__file__).resolve().parent

    data_dir = (
        project_dir
        / "data"
    )

    checkpoint_path = (
        project_dir
        / "resnet18_cifar10_best.pt"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    amp_enabled = (
        device.type == "cuda"
        and not args.no_amp
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, test_loader, train_dataset, test_dataset = build_loaders(
        data_dir,
        args.batch_size,
        args.workers
    )

    print("\n--- CIFAR-10 ---")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Training images:", len(train_dataset))
    print("Testing images:", len(test_dataset))
    print("Batch size:", args.batch_size)
    print("Mixed precision:", amp_enabled)

    # --------------------------------------------------------
    # MODE: inspect
    # --------------------------------------------------------

    if args.mode == "inspect":

        images, labels = next(
            iter(train_loader)
        )

        print(
            "Batch images:",
            tuple(images.shape)
        )

        print(
            "Batch labels:",
            tuple(labels.shape)
        )

        return

    model = ResNet18CIFAR(
        num_classes=10
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # [Added - implementation choice]
    # Use a low learning rate during the first few epochs for stable optimisation,
    # then increase to the main ResNet learning rate before cosine decay.

    base_lr = 0.1
    warmup_epochs = 5

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )

    # Warm up from 10% of the base LR:
    # 0.01 -> 0.1 over the first five epochs.
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs - 1
    )

    # ★ CORE: gradually reduce the learning rate after warm-up.
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - warmup_epochs),
        eta_min=1e-4
    )

    # ★ CORE: run warm-up first, then automatically switch to cosine decay.
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            warmup_scheduler,
            cosine_scheduler
        ],
        milestones=[warmup_epochs]
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled
    )

    # --------------------------------------------------------
    # MODE: smoke
    # --------------------------------------------------------

    if args.mode == "smoke":

        print("\n--- Smoke Test ---")

        # [Debug] Verify the untrained model before parameter updates.
        images, labels = next(iter(train_loader))
        images = images.to(device)
        labels = labels.to(device)

        model.eval()

        with torch.no_grad():
            initial_logits = model(images)
            initial_loss = criterion(initial_logits, labels)

        print(
            "Initial logits range:",
            f"{initial_logits.min().item():.4f}",
            "to",
            f"{initial_logits.max().item():.4f}"
        )

        print(
            "Initial loss:",
            f"{initial_loss.item():.4f}"
        )

        train_loss, train_acc, train_time = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
            max_batches=5
        )



        test_loss, test_acc, test_time = evaluate(
            model,
            test_loader,
            criterion,
            device,
            amp_enabled,
            max_batches=5
        )

        print(
            f"Train 5 batches | "
            f"loss={train_loss:.4f} "
            f"acc={train_acc:.4f} "
            f"time={train_time:.2f}s"
        )

        print(
            f"Test 5 batches | "
            f"loss={test_loss:.4f} "
            f"acc={test_acc:.4f} "
            f"time={test_time:.2f}s"
        )

        return

    # --------------------------------------------------------
    # MODE: eval / demo
    # --------------------------------------------------------

    if args.mode in ("eval", "demo"):

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = load_checkpoint(
            checkpoint_path,
            model,
            device
        )

        print(
            "\nLoaded checkpoint:",
            checkpoint_path
        )

        print(
            "Saved best accuracy:",
            f"{checkpoint['best_accuracy']:.4f}"
        )

        test_loss, test_acc, inference_time = evaluate(
            model,
            test_loader,
            criterion,
            device,
            amp_enabled
        )

        print(
            f"Inference | "
            f"loss={test_loss:.4f} "
            f"acc={test_acc:.4f} "
            f"time={inference_time:.2f}s"
        )

        if args.mode == "eval":
            return

        # Course demo requirement:
        # run one complete training epoch on Rangpur.
        demo_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True
        )

        demo_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled
        )

        train_loss, train_acc, epoch_time = train_one_epoch(
            model,
            train_loader,
            criterion,
            demo_optimizer,
            demo_scaler,
            device,
            amp_enabled
        )

        print(
            f"Demo 1 epoch | "
            f"loss={train_loss:.4f} "
            f"acc={train_acc:.4f} "
            f"time={epoch_time:.2f}s"
        )

        return

    # --------------------------------------------------------
    # MODE: train
    # --------------------------------------------------------

    print("\n--- Full Training ---")

    best_accuracy = 0.0

    # Sum of time spent specifically inside training epochs.
    training_compute_time = 0.0

    # Wall-clock measurement for the complete training loop,
    # including evaluation and checkpoint operations.
    run_start = time.perf_counter()

    time_to_90 = None
    time_to_94 = None


    for epoch in range(1, args.epochs + 1):

        # Record the learning rate actually used for this epoch.
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc, epoch_time = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled
        )

        training_compute_time += epoch_time

        test_loss, test_acc, _ = evaluate(
            model,
            test_loader,
            criterion,
            device,
            amp_enabled
        )

        elapsed_wall = time.perf_counter() - run_start

        # ★ CORE: record the first time each DAWNBench accuracy target is reached.
        if test_acc >= 0.90 and time_to_90 is None:
            time_to_90 = elapsed_wall

            print(
                f">>> Reached 90% accuracy at epoch {epoch} "
                f"after {time_to_90:.2f}s"
            )

        if test_acc >= 0.94 and time_to_94 is None:
            time_to_94 = elapsed_wall

            print(
                f">>> Reached 94% accuracy at epoch {epoch} "
                f"after {time_to_94:.2f}s"
            )

        # Advance the learning-rate schedule for the next epoch.
        scheduler.step()

        if test_acc > best_accuracy:

            best_accuracy = test_acc

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_accuracy
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr={current_lr:.5f} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_acc:.4f} | "
            f"epoch={epoch_time:.2f}s | "
            f"best={best_accuracy:.4f}"
        )


    total_wall_time = time.perf_counter() - run_start


    print("\n--- Final Result ---")

    print(
        "Best test accuracy:",
        f"{best_accuracy:.4f}"
    )

    print(
        "Training compute time:",
        f"{training_compute_time:.2f}s"
    )

    print(
        "Training-loop wall time:",
        f"{total_wall_time:.2f}s"
    )

    if time_to_90 is not None:
        print(
            "Time to 90% accuracy:",
            f"{time_to_90:.2f}s"
        )
    else:
        print("Time to 90% accuracy: NOT REACHED")

    if time_to_94 is not None:
        print(
            "Time to 94% accuracy:",
            f"{time_to_94:.2f}s"
        )
    else:
        print("Time to 94% accuracy: NOT REACHED")

    print(
        "Best checkpoint:",
        checkpoint_path
    )


if __name__ == "__main__":
    main()