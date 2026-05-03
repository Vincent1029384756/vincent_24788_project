import torch
import torch.nn as nn
from model_var import DGCNN_Seg
from dataset import get_dataloader
import matplotlib.pyplot as plt

# The paper did not specify a specific loss function to use, so I'm gonna use cross entropy loss
# since it's a classification problem
criterion = nn.CrossEntropyLoss()

# import model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DGCNN_Seg(num_parts=4).to(device)

"""
Training parameters according to paper:
SGD: lr = 0.1, reducing to 0.001 with cos annealing
momentum = 0.9
batch size = 32
batch norm momentum = 0.9
no batch norm decay
"""
# number of epochs
num_epochs = 200

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=0.001
)

train_loader = get_dataloader(split='train', batch_size=32)
val_loader = get_dataloader(split='val', batch_size=32)

train_losses = []
val_losses = []

plt.ion()
fig, ax = plt.subplots()

best_loss = float('inf')
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    # training
    model.train()
    train_loss = 0

    for points, labels in train_loader:
        # points: (batch, 2048, 3)
        # labels: (batch, 2048)
        points, labels = points.to(device), labels.to(device)
        batch_size = points.shape[0]

        # convert to PyG format
        x = points.view(-1, 3)                                          # (batch*2048, 3)
        batch_idx = torch.arange(batch_size, device=device)
        batch_idx = batch_idx.repeat_interleave(2048)                   # (batch*2048,)

        optimizer.zero_grad()
        preds = model(x, batch_idx)                                     # (batch, 4, 2048)
        loss = criterion(preds, labels.long())
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
            batch_size = points.shape[0]

            x = points.view(-1, 3)
            batch_idx = torch.arange(batch_size, device=device)
            batch_idx = batch_idx.repeat_interleave(2048)

            preds = model(x, batch_idx)
            loss = criterion(preds, labels.long())
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    ax.clear()
    ax.plot(range(1, epoch+2), train_losses, label='Train')
    ax.plot(range(1, epoch+2), val_losses, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training vs Validation Loss')
    ax.legend()
    plt.pause(0.1)

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), 'checkpoint_best.pth')

plt.ioff()
plt.savefig('loss_curve.png')
plt.show()