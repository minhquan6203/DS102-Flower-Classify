import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Model(nn.Module):

    def __init__(self, config):
        super(self).__init__(config)
        
    def LeNet5(self,config):
        
        model = nn.Sequential(
            nn.Conv2d(in_channels=config.image_C, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=config.num_classes),
            nn.Softmax(dim=1),
        )

        return model
    
    def AlexNet(self,config):
        model = nn.Sequential(
            nn.Conv2d(in_channels=config.image_C, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=6*6*256, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=4096, out_features=config.num_classes),
            nn.Softmax(dim=1),
        )

        return model

    def define_model(self):
        
        if self.type_model=="AlexNet":
            return self.AlexNet()
        elif self.type_model=="LeNet5":
          return self.LeNet5()
        else:
          raise ValueError("model haven't defined")
          
    def forward(self, x):
        x = self.define_model()(x)
        return x
