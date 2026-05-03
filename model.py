import torch
import torch.nn as nn

class TNet(nn.Module):
    """
    TNet is itself a mini network consist of shared mlp, max pooling, and fully connected network.
    It's part of the transformation layers in the main network
    """
    def __init__(self, k):
        super().__init__()
        self.k = k # k is the output matrix size

        """
        Shared MLP layer
        MLP(64; 128; 1024)
        """
        self.mlp = nn.Sequential(
            nn.Conv1d(k, 64, 1), # use Conv1d with kernel size = 1 to build shared mlp layers
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        """
        Fully connected layers (1024, 512, 256)
        last layer does not include ReLU and batchnorm
        """
        self.fc= nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k*k)
        )

    def forward(self, x):
        """
        mlp -> max pooling -> fc
        """
        batch_size = x.shape[0] 
        x = self.mlp(x) # (batch, 1024, n)
        x = torch.max(x, dim=2)[0] # (batch, 1024)
        x = self.fc(x) # (batch, k*k)
        x = x.view(batch_size, self.k, self.k) # (batch, k, k), one k x k matrix per shape
        x = x + torch.eye(self.k, device=x.device) # to initialize output matrix as identity (or close to indentity)
        return x
    
class PointNet(nn.Module):
    """
    Main network
    Consists of two transform layer, multiple mlp, and a max pooling layer
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes # this variable is 'm' in the paper

        """
        Initialize the 2 TNets used in the model
        """
        self.tnet1 = TNet(k=3)
        self.tnet2 = TNet(k=64)
        
        """
        Initialize the 4 mlps in the model
        batchnorm and relu used for all layers
        """
        # First mlp: nx3 -> (64, 64)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Second mlp: nx64 -> (64, 128, 1024)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        # Third mlp: nx1088 -> (512, 256, 128)
        self.mlp3 = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Fourth mlp: nx128 -> (128, 4), since there are 4 classes of shapes
        self.mlp4 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 4, 1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        transform1 -> mlp1 -> transform2 -> mlp2 -> max pool -> concatnate with output of transform2
        -> mlp3 -> mlp4 -> output

        each transformation layer is a batch matrix multiplication of x and TNet(x)
        """
        # get n
        n = x.shape[1]
        # transpose input, (batch, n, 3) -> (batch, 3, n), since all Conv1d layers needs (batch, 3, n)
        x = x.transpose(2, 1) # (batch, 3, n)
        # tranform1
        matrix1 = self.tnet1(x) # (batch, 3, 3)
        x = x.transpose(2,1) # (batch, n, 3) again
        x = torch.bmm(x, matrix1) # (batch, n, 3)
        # mlp1
        x = x.transpose(2, 1) # (batch, 3, n)
        x = self.mlp1(x) # (batch, 64, n)
        # transform2, keep the output to be concatnated later
        matrix2 = self.tnet2(x) # (batch, 64, 64)
        x = x.transpose(2,1) # (batch, n, 64)
        x = torch.bmm(x, matrix2) # (batch, n, 64)
        trans_out = x # (batch, n, 64)
        # mlp2
        x = x.transpose(2, 1) # (batch, 64, n) 
        x = self.mlp2(x) # (batch, 1024, n) 
        # max pool
        x = torch.max(x, dim=2)[0] # dim=2 to take max across all n points, [0] gets the values without indices, (batch, 1024)
        # we need to expand the global feature back to (batch, 1024, n)
        x = x.unsqueeze(2) # (batch, 1024, 1)
        x = x.expand(-1, -1, n) # (batch, 1024, n)
        # concatnate output of max pool and second transformation layer
        x = torch.cat([trans_out.transpose(2,1), x], dim=1) # (batch, 1088, n)
        # mlp 3
        x = self.mlp3(x) # (batch, 128, n)
        # mlp 4
        x = self.mlp4(x) # (batch, 4, n)
        x = x.transpose(2,1) # (batch, n, 4)
        # need to also return matrix2
        return x, matrix2