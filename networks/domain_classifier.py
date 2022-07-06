import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class DomainClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(DomainClassifier, self).__init__()
        self.hidden = nn.Linear(in_channels, hidden_channels)
        self.output = nn.Linear(hidden_channels, 2)
        self.loss   =  nn.CrossEntropyLoss()
        
    def forward(self, x, y):
        """
        Args:
            X (_(batch_size, 3, in_channels)_): _description_
            y: Label from domain
        """
        batch_size = X[0]
        x = x.view(batch_size * 3, -1)
        y = y.view(batch_size * 3, -1)
        h1 = F.relu(self.hidden(x))
        h2batch_size = X[0]
        x = x.view(batch_size * 3, -1)
        h1 = F.relu(self.hidden(x))
        h2 = F.relu(self.output(h1))
        output = loss(h2, y)
        return output