from torch_geometric.datasets import ShapeNet
import torch

dataset = ShapeNet(root='./data/ShapeNet', categories=['Airplane'])
print(f"Dataset size: {len(dataset)}")
print(f"Sample: {dataset[0]}")
print(f"CUDA available: {torch.cuda.is_available()}")

# to see what is inside the .pt files
data = torch.load('data/ShapeNet/processed/air_train.pt')
print(type(data))
print(data[0])
print(data[0].keys())
print(len(data[0]['pos']))

print(type(data[1]))
print(data[1])

print(type(data[2]))
print(data[2])