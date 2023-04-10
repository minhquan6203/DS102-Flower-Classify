import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.svm import SVC
from sklearn.cluster import KMeans

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




class SVM_Model:
    def __init__(self, config):
        self.num_classes = config.num_classes
        self.svm = SVC(kernel='linear', C=1.0)

    def fit(self, X, y):
        self.svm.fit(X, y)

    def predict(self, X):
        return self.svm.predict(X)




class Kmeans_Model(nn.Module):

    def __init__(self, config):
        super(Kmeans_Model, self).__init__()
        self.num_classes = config.num_classes
        self.kmeans = KMeans(n_clusters=self.num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten the input tensor
        x = self.kmeans.fit_predict(x)
        return x
