import matplotlib.pyplot as plt
import torch
from model_var import DGCNN_Seg
from dataset import ShapeNetDataset

# import model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DGCNN_Seg(num_parts=4).to(device)

def visualize_prediction(model, dataset, idx, device):
    model.eval()
    points, labels = dataset[idx]
    
    with torch.no_grad():
        input = points.unsqueeze(0).to(device)      # (1, 2048, 3)
        
        # convert to PyG format
        x = input.view(-1, 3)                        # (2048, 3)
        batch_idx = torch.zeros(2048, dtype=torch.long, device=device)  # all same shape

        preds = model(x, batch_idx)                  # (1, 4, 2048)
        pred_labels = preds.squeeze(0).argmax(dim=0).cpu()  # (2048,)

    points = points.numpy()
    labels = labels.numpy()
    pred_labels = pred_labels.numpy()

    fig = plt.figure(figsize=(12, 5))

    # ground truth
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap='tab10', s=1)
    ax1.set_title('Ground Truth')
    ax1.axis('off')

    # prediction
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=pred_labels, cmap='tab10', s=1)
    ax2.set_title('Prediction')
    ax2.axis('off')

    plt.savefig(f'visualization_{idx}.png')
    plt.show()

# load best checkpoint and visualize a few samples
model.load_state_dict(torch.load('checkpoint_best_var.pth'))
val_dataset = ShapeNetDataset(split='val')

for i in [0, 1, 2, 4, 5]:  # visualize 3 samples
    visualize_prediction(model, val_dataset, i, device)