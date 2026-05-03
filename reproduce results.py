import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model import PointNet
from model_var import DGCNN_Seg
from dataset import ShapeNetDataset
from dataset import get_dataloader

"""
This file reproduce training results of both PointNet and DGCNN models.
The file will return
- mIoU score of each model
- 5 visualization examples of each model
"""

'''
PointNet
'''
# import PointNet
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
print(f'PointNet mIoU: {miou:.4f}')

def visualize_prediction_pointnet(model, dataset, idx, device, model_name='PointNet'):
    model.eval()
    points, labels = dataset[idx]
    
    with torch.no_grad():
        input = points.unsqueeze(0).to(device)      # (1, 2048, 3)
        preds, _ = model(input)                      # PointNet returns (preds, matrix2)
        pred_labels = preds.squeeze(0).argmax(dim=1).cpu()  # (2048,)

    points = points.numpy()
    labels = labels.numpy()
    pred_labels = pred_labels.numpy()

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(model_name, fontsize=14, fontweight='bold')

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap='tab10', s=1)
    ax1.set_title('Ground Truth')
    ax1.axis('off')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=pred_labels, cmap='tab10', s=1)
    ax2.set_title('Prediction')
    ax2.axis('off')

    plt.savefig(f'visualization_{model_name}_{idx}.png')
    plt.show()

val_dataset = ShapeNetDataset(split='val')

for i in [0, 1, 2, 4, 5]:  # visualize 5 samples
    visualize_prediction_pointnet(model, val_dataset, i, device)

'''
DGCNN
'''
# import DGCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DGCNN_Seg(num_parts=4).to(device)
model.load_state_dict(torch.load('checkpoint_best_var.pth'))

val_loader = get_dataloader(split='val', batch_size=32)

def compute_miou(model, loader, device):
    model.eval()
    iou_per_class = [[] for _ in range(4)]
    
    with torch.no_grad():
        for points, labels in loader:
            points, labels = points.to(device), labels.to(device)
            batch_size = points.shape[0]

            # convert to PyG format
            x = points.view(-1, 3)
            batch_idx = torch.arange(batch_size, device=device).repeat_interleave(2048)

            preds = model(x, batch_idx)          # (batch, 4, 2048)
            pred_labels = preds.argmax(dim=1)    # (batch, 2048)
            
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
print(f'DGCNN mIoU: {miou:.4f}')

def visualize_prediction(model, dataset, idx, device, model_name='Model'):
    model.eval()
    points, labels = dataset[idx]
    
    with torch.no_grad():
        input = points.unsqueeze(0).to(device)      # (1, 2048, 3)
        x = input.view(-1, 3)                        # (2048, 3)
        batch_idx = torch.zeros(2048, dtype=torch.long, device=device)

        preds = model(x, batch_idx)                  # (1, 4, 2048)
        pred_labels = preds.squeeze(0).argmax(dim=0).cpu()  # (2048,)

    points = points.numpy()
    labels = labels.numpy()
    pred_labels = pred_labels.numpy()

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(model_name, fontsize=14, fontweight='bold')  # add this

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap='tab10', s=1)
    ax1.set_title('Ground Truth')
    ax1.axis('off')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=pred_labels, cmap='tab10', s=1)
    ax2.set_title('Prediction')
    ax2.axis('off')

    plt.savefig(f'visualization_{model_name}_{idx}.png')
    plt.show()

val_dataset = ShapeNetDataset(split='val')

for i in [0, 1, 2, 4, 5]:
    visualize_prediction(model, val_dataset, i, device, model_name='DGCNN')