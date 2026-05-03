import torch
import torch.nn as nn
from model import PointNet
from dataset import get_dataloader
from dataset import ShapeNetDataset

# import model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PointNet(num_classes=4).to(device)
model.load_state_dict(torch.load('checkpoint_best_base.pth'))


val_loader = get_dataloader(split='val', batch_size=32)

def compute_miou(model, loader, device):
    model.eval()
    iou_per_class = [[] for _ in range(4)]
    
    with torch.no_grad():
        for points, labels in loader:
            points, labels = points.to(device), labels.to(device)
            preds, _ = model(points)
            pred_labels = preds.argmax(dim=-1)  # (batch, n)
            
            for cls in range(4):
                pred_cls = (pred_labels == cls)
                true_cls = (labels == cls)
                intersection = (pred_cls & true_cls).sum().item()
                union = (pred_cls | true_cls).sum().item()
                if union > 0:
                    iou_per_class[cls].append(intersection / union)
    
    class_ious = [sum(c)/len(c) for c in iou_per_class if c]
    return sum(class_ious) / len(class_ious)

miou = compute_miou(model, val_loader, device)
print(f'mIoU: {miou:.4f}')

dataset = ShapeNetDataset(split='val')
points, labels = dataset[0]
print(labels.unique())