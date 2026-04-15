import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (256, 256)
DEFAULT_IMAGES_DIR = Path("dataset/content/dataset/images")
DEFAULT_MASKS_DIR = Path("dataset/content/dataset/masks")
DEFAULT_TEST_DIR = Path("test")
DEFAULT_MODEL_PATH = Path("best_unet_rust.pth")


class RustSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=IMAGE_SIZE, augment=False, fnames=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.augment = augment

        all_fnames = sorted(
            image_path.stem
            for image_path in self.images_dir.iterdir()
            if image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        self.fnames = list(fnames) if fnames is not None else all_fnames

        self.to_tensor_img = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        )
        self.to_tensor_mask = transforms.Compose(
            [
                transforms.Resize(img_size, interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        image_path = self.images_dir / f"{fname}.png"
        mask_path = self.masks_dir / f"{fname}.png"

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment and random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        image = self.to_tensor_img(image)
        mask = self.to_tensor_mask(mask)
        mask = (mask > 0.5).float()
        return image, mask


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.down1 = DoubleConv(n_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool(c1)

        c2 = self.down2(p1)
        p2 = self.pool(c2)

        c3 = self.down3(p2)
        p3 = self.pool(c3)

        c4 = self.down4(p3)
        p4 = self.pool(c4)

        bottleneck = self.bottleneck(p4)

        u4 = self.up4(bottleneck)
        u4 = torch.cat([u4, c4], dim=1)
        c4 = self.conv4(u4)

        u3 = self.up3(c4)
        u3 = torch.cat([u3, c3], dim=1)
        c3 = self.conv3(u3)

        u2 = self.up2(c3)
        u2 = torch.cat([u2, c2], dim=1)
        c2 = self.conv2(u2)

        u1 = self.up1(c2)
        u1 = torch.cat([u1, c1], dim=1)
        c1 = self.conv1(u1)

        return self.out_conv(c1)


def dice_coeff(logits, target, eps=1e-7):
    pred = torch.sigmoid(logits)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def iou_score(logits, target, eps=1e-7):
    pred = torch.sigmoid(logits)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def build_dataloaders(images_dir, masks_dir, batch_size, val_ratio=0.2, seed=42):
    base_dataset = RustSegDataset(images_dir, masks_dir, img_size=IMAGE_SIZE, augment=False)
    print(f"Total samples: {len(base_dataset)}")

    val_len = int(len(base_dataset) * val_ratio)
    train_len = len(base_dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(base_dataset, [train_len, val_len], generator=generator)
    train_fnames = [base_dataset.fnames[idx] for idx in train_subset.indices]
    val_fnames = [base_dataset.fnames[idx] for idx in val_subset.indices]

    train_dataset = RustSegDataset(images_dir, masks_dir, img_size=IMAGE_SIZE, augment=True, fnames=train_fnames)
    val_dataset = RustSegDataset(images_dir, masks_dir, img_size=IMAGE_SIZE, augment=False, fnames=val_fnames)
    print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")

    num_workers = 0 if os.name == "nt" else 2
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_dice += dice_coeff(logits, masks).item() * images.size(0)
        total_iou += iou_score(logits, masks).item() * images.size(0)

    size = len(loader.dataset)
    return total_loss / size, total_dice / size, total_iou / size


def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, masks)

            total_loss += loss.item() * images.size(0)
            total_dice += dice_coeff(logits, masks).item() * images.size(0)
            total_iou += iou_score(logits, masks).item() * images.size(0)

    size = len(loader.dataset)
    return total_loss / size, total_dice / size, total_iou / size


def train_model(model, train_loader, val_loader, model_path, epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_dice = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_dice, val_iou = eval_one_epoch(model, val_loader, criterion)

        print(
            f"[Epoch {epoch:02d}] "
            f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} IoU: {val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved to {model_path} with Val Dice = {best_val_dice:.4f}")


def load_model_weights(model, model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model weights: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()


def show_validation_predictions(model, val_loader):
    images, masks = next(iter(val_loader))
    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    with torch.no_grad():
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    idx = random.randint(0, images.size(0) - 1)
    img_np = images[idx].cpu().permute(1, 2, 0).numpy()
    mask_np = masks[idx].cpu().squeeze().numpy()
    pred_np = preds[idx].cpu().squeeze().numpy()

    overlay = img_np.copy()
    overlay[pred_np > 0.5] = [1.0, 0.0, 0.0]
    blend = np.clip(img_np * 0.5 + overlay * 0.5, 0, 1)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 4, 1)
    plt.title("Image")
    plt.imshow(img_np)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("GT Mask")
    plt.imshow(mask_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Pred Mask")
    plt.imshow(pred_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Prediction Overlay")
    plt.imshow(blend)
    plt.axis("off")
    plt.show()


def predict_random_test_image(model, test_dir):
    test_dir = Path(test_dir)
    files = sorted([path for path in test_dir.iterdir() if path.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    if not files:
        raise ValueError(f"No test images found in {test_dir}")

    image_path = random.choice(files)
    print(f"Random test image: {image_path}")

    image = Image.open(image_path).convert("RGB")
    resized = np.array(image.resize(IMAGE_SIZE, Image.BILINEAR))
    image_tensor = torch.tensor(resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        pred = (torch.sigmoid(logits) > 0.5).float()

    pred_np = pred[0].cpu().squeeze().numpy()
    overlay = resized.copy()
    red_mask = np.zeros_like(overlay)
    red_mask[:, :, 0] = 255
    overlay[pred_np > 0.5] = (
        overlay[pred_np > 0.5] * 0.5 + red_mask[pred_np > 0.5] * 0.5
    ).astype(np.uint8)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title(f"Random Image\n{image_path.name}")
    plt.imshow(resized)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Pred Mask")
    plt.imshow(pred_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Train and inspect a U-Net rust segmentation model.")
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR))
    parser.add_argument("--masks-dir", default=str(DEFAULT_MASKS_DIR))
    parser.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "predict", "all"],
        default="all",
        help="train: train only, eval: validation preview only, predict: test image only, all: train then preview",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Using device: {DEVICE}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    train_loader, val_loader = build_dataloaders(args.images_dir, args.masks_dir, args.batch_size)
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    model_path = Path(args.model_path)

    if args.mode in {"train", "all"}:
        train_model(model, train_loader, val_loader, model_path, args.epochs)

    if args.mode in {"eval", "predict"} and not model_path.exists():
        raise FileNotFoundError(f"Could not find model weights: {model_path}")

    if args.mode in {"eval", "predict", "all"}:
        load_model_weights(model, model_path)

    if args.mode in {"eval", "all"}:
        show_validation_predictions(model, val_loader)

    if args.mode in {"predict", "all"}:
        predict_random_test_image(model, args.test_dir)


if __name__ == "__main__":
    main()
