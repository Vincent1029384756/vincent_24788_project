import torch
import torch.nn as nn
from model import PointNet
from dataset import get_dataloader
import matplotlib.pyplot as plt

def total_loss(preds, labels, matrix2):
    """
    loss function consist of:
    crossentropy loss applied at the output
    and regularization loss with weirght 0.001 applied at the output of the second TNet
    """
    # cross entropy loss
    c_loss = nn.CrossEntropyLoss()(preds.transpose(1,2), labels) # (batch, 4, n)

    # regularization loss: ||I - A*A^T||^2
    batch_size = matrix2.shape[0]
    I = torch.eye(64, device=matrix2.device).unsqueeze(0).repeat(batch_size, 1, 1)  # (batch, 64, 64)
    AAT = torch.bmm(matrix2, matrix2.transpose(2, 1))  # (batch, 64, 64)
    reg_loss = torch.mean(torch.norm(I - AAT, dim=(1, 2)))
    
    return c_loss + 0.001 * reg_loss

# import model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PointNet(num_classes=4).to(device)

"""
Training parameters
We dont need dropout for segmentation model, but we need batchnorm momentum
"""
# batch norm with momentum decay from 0.5 to 0.9
def update_bn_momentum(model, epoch, num_epochs):
    momentum = min(0.99, 0.5 * (0.99 / 0.5) ** (epoch / num_epochs))
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.momentum = 1 - momentum  # convert TF decay → PyTorch momentum
# paper uses Adam with initial lr of 0.001, and learning rate divided by 2 every 20 epochs
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
# number of epochs
num_epochs = 100

# dataloaders, paper uses batch size of 32
train_loader = get_dataloader(split='train', batch_size=32)
val_loader = get_dataloader(split='val', batch_size=32)

"""
Model Training
"""
train_losses = []
val_losses = []
# initialize loss plot
plt.ion()
fig, ax = plt.subplots()

best_loss = float('inf')

for epoch in range(num_epochs):
    # update batch norm momentum
    update_bn_momentum(model, epoch, num_epochs)
    # training
    model.train()
    train_loss = 0

    for points, labels in train_loader:
        points, labels = points.to(device), labels.to(device)
        
        optimizer.zero_grad()
        preds, matrix2 = model(points)
        loss = total_loss(preds, labels, matrix2)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    scheduler.step()

    # validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for points, labels in val_loader:
            points, labels = points.to(device), labels.to(device)
            preds, matrix2 = model(points)
            loss = total_loss(preds, labels, matrix2)
            val_loss += loss.item()

    # print progress
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # update plot
    ax.clear()
    ax.plot(range(1, epoch+2), train_losses, label='Train')
    ax.plot(range(1, epoch+2), val_losses, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training vs Validation Loss')
    ax.legend()
    plt.pause(0.1)

    # save best model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), 'checkpoint_best_base.pth')


plt.ioff()
plt.savefig('loss_curve.png')
plt.show()