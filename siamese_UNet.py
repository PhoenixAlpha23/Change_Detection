import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from albumentations import (
    Compose, Resize, RandomRotate90, HorizontalFlip, VerticalFlip,
    ShiftScaleRotate, ColorJitter, RandomBrightnessContrast, GaussianBlur,
    Normalize, RandomCrop
)
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


class LEVIRCDDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.dir_b = os.path.join(images_dir, 'B')
        self.dir_a = os.path.join(images_dir, 'A')
        self.dir_m = os.path.join(images_dir, 'label')
        self.transform = transform
        self.list_b = sorted([f for f in os.listdir(self.dir_b) if f.endswith(('png','jpg','jpeg'))])
        self.list_a = sorted([f for f in os.listdir(self.dir_a) if f.endswith(('png','jpg','jpeg'))])
        self.list_m = sorted([f for f in os.listdir(self.dir_m) if f.endswith(('png','jpg','jpeg'))])

    def __len__(self):
        return len(self.list_b)

    def __getitem__(self, idx):
        # Load images
        img_b = cv2.cvtColor(cv2.imread(os.path.join(self.dir_b, self.list_b[idx])), cv2.COLOR_BGR2RGB)
        img_a = cv2.cvtColor(cv2.imread(os.path.join(self.dir_a, self.list_a[idx])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.dir_m, self.list_m[idx]), cv2.IMREAD_GRAYSCALE)

        # Normalize images to [0, 1]
        img_b = img_b.astype(np.float32) / 255.0
        img_a = img_a.astype(np.float32) / 255.0

        # Normalize mask to [0, 1] and convert to float32
        mask = mask.astype(np.float32) / 255.0

        if self.transform:
            augmented = self.transform(image=img_b, image1=img_a, mask=mask)
            img_b = augmented['image']
            img_a = augmented['image1']
            mask = augmented['mask']
        # concat pre/post change as 6-channel input
        x = torch.cat([img_b, img_a], dim=0)
        mask = mask.unsqueeze(0).float()    

        return x, mask


def get_transforms(train=True, size=256):
    if train:
        return Compose([
            Resize(size, size),
            RandomRotate90(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            RandomBrightnessContrast(p=0.5),
            GaussianBlur(p=0.2),
            Normalize(mean=(0.485,0.456,0.406)*2, std=(0.229,0.224,0.225)*2),
            ToTensorV2()
        ], additional_targets={'image1':'image'})
    else:
        return Compose([
            Resize(size, size),
            Normalize(mean=(0.485,0.456,0.406)*2, std=(0.229,0.224,0.225)*2),
            ToTensorV2()
        ], additional_targets={'image1':'image'})


def boundary_loss(pred, mask):
    # pred: sigmoid output [B,1,H,W], mask: [B,1,H,W]
    # morphological boundary: dilate minus orig
    pred_d = F.max_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mask_d = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    b_pred = pred_d - pred
    b_mask = mask_d - mask
    return F.l1_loss(b_pred, b_mask)


def compute_metrics(pred, mask, threshold=0.5):
    pred_bin = (pred > threshold).float()
    mask_bin = (mask > 0.5).float()
    # Flatten
    pred_flat = pred_bin.view(-1)
    mask_flat = mask_bin.view(-1)
    tp = (pred_flat * mask_flat).sum()
    tn = ((1 - pred_flat) * (1 - mask_flat)).sum()
    fp = (pred_flat * (1 - mask_flat)).sum()
    fn = ((1 - pred_flat) * mask_flat).sum()
    eps = 1e-6
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return iou.item(), accuracy.item(), precision.item(), recall.item()


def train_one_epoch(model, loader, device, criterion_bce, lambda1, lambda2):
    model.train()
    running_loss = 0
    for x, mask in loader:
        x, mask = x.to(device), mask.to(device)
        optimizer.zero_grad()
        logits = model(x)
        bce = criterion_bce(logits, mask)
        pred = torch.sigmoid(logits)
        b_loss = boundary_loss(pred, mask)
        loss = lambda1 * b_loss + lambda2 * bce
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


def validate(model, loader, device, criterion_bce, lambda1, lambda2):
    model.eval()
    val_loss = 0
    metrics = {'iou':0, 'acc':0, 'prec':0, 'rec':0}
    with torch.no_grad():
        for x, mask in loader:
            x, mask = x.to(device), mask.to(device)
            logits = model(x)
            bce = criterion_bce(logits, mask)
            pred = torch.sigmoid(logits)
            b_loss = boundary_loss(pred, mask)
            loss = lambda1 * b_loss + lambda2 * bce
            val_loss += loss.item() * x.size(0)
            iou, acc, prec, rec = compute_metrics(pred, mask)
            metrics['iou'] += iou * x.size(0)
            metrics['acc'] += acc * x.size(0)
            metrics['prec'] += prec * x.size(0)
            metrics['rec'] += rec * x.size(0)
    n = len(loader.dataset)
    for k in metrics: metrics[k] /= n
    return val_loss / n, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda1', type=float, default=1.0, help='Boundary loss weight')
    parser.add_argument('--lambda2', type=float, default=1.0, help='BCE loss weight')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Datasets
    train_ds = LEVIRCDDataset(os.path.join(args.data_dir, 'train'), transform=get_transforms(True))
    val_ds   = LEVIRCDDataset(os.path.join(args.data_dir, 'val'),   transform=get_transforms(False))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model: U-Net with ResNet34 backbone
    model = smp.Unet(
        encoder_name='resnet34', encoder_weights='imagenet', in_channels=6, classes=1,
        activation=None  # logits output
    ).to(device)

    # Loss and optimizer
    # Compute pos_weight from data (optional) or set to 1
    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_iou = 0
    for epoch in range(1, args.epochs+1):
        train_loss = train_one_epoch(model, train_loader, device, criterion_bce, args.lambda1, args.lambda2)
        val_loss, metrics = validate(model, val_loader, device, criterion_bce, args.lambda1, args.lambda2)
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val IoU: {metrics['iou']:.4f}, Acc: {metrics['acc']:.4f}, Prec: {metrics['prec']:.4f}, Rec: {metrics['rec']:.4f}")

        # Save best
        if metrics['iou'] > best_iou:
            best_iou = metrics['iou']
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved with IoU: {best_iou:.4f}")
