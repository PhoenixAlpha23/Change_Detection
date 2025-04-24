import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from siamese_UNet import LEVIRCDDataset, get_transforms  

def evaluate(model_path, test_dir, batch_size=4, size=256, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    test_ds = LEVIRCDDataset(test_dir, transform=get_transforms(train=False, size=size))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=6,
        classes=1,
        activation=None
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_masks = [], []

    with torch.no_grad():
        for x, mask in test_loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.sigmoid(logits)
            pred = (pred > threshold).float().cpu().numpy()
            mask = (mask > 0.5).float().cpu().numpy()

            all_preds.extend(pred.reshape(-1))
            all_masks.extend(mask.reshape(-1))

    all_preds = np.array(all_preds)
    all_masks = np.array(all_masks)

    acc = accuracy_score(all_masks, all_preds)
    prec = precision_score(all_masks, all_preds)
    rec = recall_score(all_masks, all_preds)
    f1 = f1_score(all_masks, all_preds)

    print(f"Evaluation on Test Set:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='best_model_f1.pth')
    parser.add_argument('--test_dir', type=str, default='C:/Users/saarsys/Desktop/AI Intern/LEVIR CD/test')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    evaluate(args.model_path, args.test_dir, args.batch_size, args.size, args.threshold)
