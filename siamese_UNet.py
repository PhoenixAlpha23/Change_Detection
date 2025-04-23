import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from albumentations import (
    Compose, Resize, RandomRotate90, HorizontalFlip, VerticalFlip,
    ShiftScaleRotate, ColorJitter, RandomBrightnessContrast, GaussianBlur,
    Normalize, RandomCrop
)
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.metrics import precision_recall_curve, f1_score


class LEVIRCDDataset(Dataset):
    def __init__(self, images_dir, transform=None, compute_weights=False):
        self.dir_b = os.path.join(images_dir, 'B')
        self.dir_a = os.path.join(images_dir, 'A')
        self.dir_m = os.path.join(images_dir, 'label')
        self.transform = transform
        self.list_b = sorted([f for f in os.listdir(self.dir_b) if f.endswith(('png','jpg','jpeg'))])
        self.list_a = sorted([f for f in os.listdir(self.dir_a) if f.endswith(('png','jpg','jpeg'))])
        self.list_m = sorted([f for f in os.listdir(self.dir_m) if f.endswith(('png','jpg','jpeg'))])
        self.weights = None
        
        # Compute sample weights for handling class imbalance
        if compute_weights:
            self.weights = []
            for idx in range(len(self.list_m)):
                mask = cv2.imread(os.path.join(self.dir_m, self.list_m[idx]), cv2.IMREAD_GRAYSCALE)
                pos_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
                # Give higher weight to samples with changes
                weight = 1.0 + (10.0 * pos_ratio)  # Scale weight based on positive pixel ratio
                self.weights.append(weight)
            
            # Normalize weights
            if self.weights:
                self.weights = np.array(self.weights) / np.sum(self.weights) * len(self.weights)

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


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


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


def post_process(pred, threshold=0.3, min_size=10):
    # Convert to numpy array
    pred_np = pred.detach().cpu().numpy().squeeze()
    # Apply threshold
    binary = (pred_np > threshold).astype(np.uint8)
    
    # Make sure binary is single channel and properly shaped for cv2.connectedComponentsWithStats
    if len(binary.shape) > 2:
        binary = binary[0]  # Take first channel if multi-channel
    
    # Check for empty predictions
    if np.sum(binary) == 0:
        return pred  # Return original prediction if no positive pixels
        
    # Remove small objects
    try:
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        img2 = np.zeros_like(binary)
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 1
        return torch.from_numpy(img2).float().unsqueeze(0).unsqueeze(0).to(pred.device)
    except Exception as e:
        print(f"Warning: Error in post_processing: {e}")
        return pred  # Return original prediction if post-processing fails


def compute_metrics(pred, mask, threshold=0.3):
    # Apply post-processing
    pred_processed = post_process(pred, threshold=threshold)
    
    pred_bin = (pred_processed > 0.5).float()
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
    f1 = 2 * precision * recall / (precision + recall + eps)
    return iou.item(), accuracy.item(), precision.item(), recall.item(), f1.item()


def find_optimal_threshold(model, val_loader, device):
    model.eval()
    all_preds = []
    all_masks = []
    with torch.no_grad():
        for x, mask in val_loader:
            x, mask = x.to(device), mask.to(device)
            logits = model(x)
            pred = torch.sigmoid(logits)
            all_preds.append(pred.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
    
    all_preds = np.concatenate([p.reshape(-1) for p in all_preds])
    all_masks = np.concatenate([m.reshape(-1) for m in all_masks])
    
    # Test different thresholds
    thresholds = np.linspace(0.1, 0.9, 9)
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        pred_bin = (all_preds > thresh).astype(np.float32)
        precision = ((pred_bin * all_masks).sum() + 1e-6) / (pred_bin.sum() + 1e-6)
        recall = ((pred_bin * all_masks).sum() + 1e-6) / (all_masks.sum() + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        print(f"Threshold: {thresh:.2f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold


def calculate_class_weights(train_loader, device):
    pos_pixels = 0
    total_pixels = 0
    for _, mask in train_loader:
        mask = mask.to(device)
        pos_pixels += (mask > 0.5).float().sum().item()
        total_pixels += mask.numel()
    
    pos_ratio = pos_pixels / total_pixels
    neg_ratio = 1 - pos_ratio
    
    # Calculate inverse frequency as weight
    pos_weight = neg_ratio / pos_ratio if pos_ratio > 0 else 10.0
    return torch.tensor([pos_weight]).to(device)


def train_one_epoch(model, loader, device, optimizer, criterion_focal, criterion_bce, lambda1, lambda2, lambda3):
    model.train()
    running_loss = 0
    for x, mask in loader:
        x, mask = x.to(device), mask.to(device)
        optimizer.zero_grad()
        logits = model(x)
        
        # Combined loss: focal loss + weighted BCE + boundary loss
        focal_loss = criterion_focal(logits, mask)
        bce_loss = criterion_bce(logits, mask)
        pred = torch.sigmoid(logits)
        b_loss = boundary_loss(pred, mask)
        
        loss = lambda1 * b_loss + lambda2 * bce_loss + lambda3 * focal_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


def validate(model, loader, device, criterion_focal, criterion_bce, lambda1, lambda2, lambda3, threshold=0.3):
    model.eval()
    val_loss = 0
    metrics = {'iou': 0, 'acc': 0, 'prec': 0, 'rec': 0, 'f1': 0}
    with torch.no_grad():
        for x, mask in loader:
            x, mask = x.to(device), mask.to(device)
            logits = model(x)
            
            # Combined loss: focal loss + weighted BCE + boundary loss
            focal_loss = criterion_focal(logits, mask)
            bce_loss = criterion_bce(logits, mask)
            pred = torch.sigmoid(logits)
            b_loss = boundary_loss(pred, mask)
            
            loss = lambda1 * b_loss + lambda2 * bce_loss + lambda3 * focal_loss
            val_loss += loss.item() * x.size(0)
            
            iou, acc, prec, rec, f1 = compute_metrics(pred, mask, threshold)
            metrics['iou'] += iou * x.size(0)
            metrics['acc'] += acc * x.size(0)
            metrics['prec'] += prec * x.size(0)
            metrics['rec'] += rec * x.size(0)
            metrics['f1'] += f1 * x.size(0)
    
    n = len(loader.dataset)
    for k in metrics: metrics[k] /= n
    return val_loss / n, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda1', type=float, default=0.5, help='Boundary loss weight')
    parser.add_argument('--lambda2', type=float, default=1.0, help='BCE loss weight')
    parser.add_argument('--lambda3', type=float, default=2.0, help='Focal loss weight')
    parser.add_argument('--threshold', type=float, default=0.3, help='Prediction threshold')
    parser.add_argument('--min_size', type=int, default=10, help='Min connected component size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Datasets with weighted sampling for training
    train_ds = LEVIRCDDataset(os.path.join(args.data_dir, 'train'), transform=get_transforms(True), compute_weights=True)
    val_ds = LEVIRCDDataset(os.path.join(args.data_dir, 'val'), transform=get_transforms(False))
    
    # Use weighted sampler for training
    sampler = None
    if train_ds.weights is not None:
        sampler = WeightedRandomSampler(train_ds.weights, len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model: U-Net with ResNet34 backbone
    model = smp.Unet(
        encoder_name='resnet34', encoder_weights='imagenet', in_channels=6, classes=1,
        activation=None  # logits output
    ).to(device)

    # Calculate class weights for BCE loss
    pos_weight = calculate_class_weights(train_loader, device)
    print(f"Calculated positive class weight: {pos_weight.item():.4f}")
    
    # Loss functions and optimizer
    criterion_focal = FocalLoss(alpha=0.25, gamma=2.0)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    best_iou = 0
    best_f1 = 0
    
    # Find optimal threshold on validation set with initial model
    print("Calculating initial optimal threshold...")
    initial_threshold = find_optimal_threshold(model, val_loader, device)
    print(f"Initial optimal threshold: {initial_threshold:.4f}")
    threshold = initial_threshold
    
    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, device, optimizer, 
                                      criterion_focal, criterion_bce, 
                                      args.lambda1, args.lambda2, args.lambda3)
        
        # Validate
        val_loss, metrics = validate(model, val_loader, device, 
                                     criterion_focal, criterion_bce,
                                     args.lambda1, args.lambda2, args.lambda3, threshold)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val IoU: {metrics['iou']:.4f}, Acc: {metrics['acc']:.4f}")
        print(f"Val Prec: {metrics['prec']:.4f}, Rec: {metrics['rec']:.4f}, F1: {metrics['f1']:.4f}")
        
        # Update learning rate based on IoU
        scheduler.step(metrics['f1'])
        
        # Save best model based on F1 score
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), 'best_model_f1.pth')
            print(f"Best model saved with F1: {best_f1:.4f}")
        
        # Save best model based on IoU (original metric)
        if metrics['iou'] > best_iou:
            best_iou = metrics['iou']
            torch.save(model.state_dict(), 'best_model_iou.pth')
            print(f"Best model saved with IoU: {best_iou:.4f}")
        
        # Recalculate optimal threshold every 5 epochs
        if epoch % 5 == 0:
            print("Recalculating optimal threshold...")
            threshold = find_optimal_threshold(model, val_loader, device)
            print(f"New optimal threshold: {threshold:.4f}")