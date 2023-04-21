import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.svm import SVC
from sklearn.cluster import KMeans

class CNN_Model(nn.Module): #use ResNet34

    def __init__(self, config):
        super(CNN_Model, self).__init__()
        self.num_classes = config.num_classes
        self.resnet = models.resnet34(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.softmax(x, dim=1)
        return x

class SVM_Model(nn.Module):
    def __init__(self, config):
        super(SVM_Model, self).__init__()
        self.linear = nn.Linear(config.image_H * config.image_W * config.image_C, config.num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)


class LeNet5(nn.Module):
    def __init__(self,config):
        super(LeNet5, self).__init__()

        #các lớp convolution
        self.conv1 = nn.Conv2d(config.image_C, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        #các lớp linear
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, config.num_classes)

    def forward(self, x):
        # lớp convolution thứ nhất
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)

        # lớp convolution thứ hai
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)

        # flatten tensor trước khi đưa vào lớp Linear
        x = x.view(-1, 16 * 5 * 5)

        #lớp fully connected
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)

        return x


class NN(nn.Module):
    def __init__(self, config):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(config.image_H * config.image_W * config.image_C, 512)
        self.linear2 = nn.Linear(512,256)
        self.linear3 = nn.Linear(256, config.num_classes)
        self.drop = nn.Dropout(0.2)
        self.acv = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x=self.linear1(x)
        x=self.acv(x)
        x=self.drop(x)
        
        x=self.linear2(x)
        x=self.acv(x)
        x=self.drop(x)

        x=self.linear3(x)
        x = torch.softmax(x, dim=1)

        return x
    
