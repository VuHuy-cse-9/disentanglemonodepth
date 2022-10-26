import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class MLPClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(MLPClassifier, self).__init__()
        self.hidden1_layer = nn.Linear(in_channels, hidden_channels)
        self.hidden2_layer = nn.Linear(hidden_channels, 2*hidden_channels)
        self.output_layer = nn.Linear(2*hidden_channels, 1)
        #self.loss   =  nn.CrossEntropyLoss()
        
    #def forward(self, X, y):
    def forward(self, X):    
        """
        Args:
            X (_(batch_size, in_channels)_): _description_
        """
        h1 = F.relu(self.hidden1_layer(X))
        h2 = F.relu(self.hidden2_layer(h1))
        output = torch.sigmoid(self.output_layer(h2))
        return torch.squeeze(output)