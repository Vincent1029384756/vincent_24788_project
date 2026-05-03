import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv
from torch_geometric.nn import global_max_pool

"""
Variant model: Dynamic Graph CNN
"""

class DGCNN_Seg(nn.Module):
    def __init__(self, num_parts=4, k=20):
        super().__init__()
        self.k = k

        """
        edge conv layers
        all layers include leakyRelu and batch norm
        """
        self.conv1 = DynamicEdgeConv(nn=nn.Sequential(
            nn.Linear(2*3, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        ), k=k, aggr='max')

        self.conv2 = DynamicEdgeConv(nn=nn.Sequential(
            nn.Linear(2*64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        ), k=k, aggr='max')

        self.conv3 = DynamicEdgeConv(nn=nn.Sequential(
            nn.Linear(2*64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        ), k=k, aggr='max')

        # global descriptor mlp
        self.mlp_global = nn.Sequential(
            nn.Conv1d(192, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU()
        )

        # per point classification
        self.mlp_seg = nn.Sequential(
            nn.Conv1d(1024 + 192, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, num_parts, 1)
        )

    def forward(self, x, batch):
        # x: (batch*n, 3)

        # EdgeConv layers
        x1 = self.conv1(x, batch)    # (batch*n, 64)
        x2 = self.conv2(x1, batch)   # (batch*n, 64)
        x3 = self.conv3(x2, batch)   # (batch*n, 64)

        # concat local features
        x_cat = torch.cat([x1, x2, x3], dim=-1)   # (batch*n, 192)

        # reshape for conv1d: (batch, 192, n)
        x_cat = x_cat.view(-1, 2048, 192).permute(0, 2, 1)

        # global descriptor
        x_global = self.mlp_global(x_cat)          # (batch, 1024, n)
        x_global = x_global.max(dim=-1)[0]         # (batch, 1024)
        x_global = x_global.unsqueeze(-1).repeat(1, 1, 2048)  # (batch, 1024, n)

        # combine global + local
        x_out = torch.cat([x_global, x_cat], dim=1)   # (batch, 1216, n)

        # per point classification
        x_out = self.mlp_seg(x_out)                # (batch, 4, n)

        return x_out