import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CNN_Model(nn.Module): #this repo use ResNet34

    def __init__(self, config):
        super(CNN_Model, self).__init__()
        self.num_classes = config.num_classes
        self.resnet = models.resnet34(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.softmax(x, dim=1)
        return x
