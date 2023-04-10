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




import torch
import torch.nn as nn

class SVM_Model(nn.Module):
    def __init__(self, config):
        super(SVM_Model, self).__init__()
        self.num_classes = config.num_classes
        self.linear = nn.Linear(config.image_H * config.image_W * 3, self.num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

    def hinge_loss(self, x, y):
        scores = self.forward(x)
        correct_scores = scores.gather(1, y.view(-1, 1)).squeeze()
        margins = torch.clamp(scores - correct_scores.view(-1, 1) + 1.0, min=0)
        margins[y.view(-1), 0] = 0
        loss = margins.sum(dim=1).mean()
        return loss

    def fit(self, train_loader, num_epochs, optimizer):
        self.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i, (data, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = self.hinge_loss(data, labels)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, loss: {epoch_loss / len(train_loader):.4f}")

    def predict(self, test_loader):
        self.eval()
        predicted_labels = []
        with torch.no_grad():
            for data in test_loader:
                scores = self.forward(data)
                _, predicted = torch.max(scores.data, 1)
                predicted_labels.extend(predicted.cpu().numpy())
        return predicted_labels


class Kmeans_Model(nn.Module):

    def __init__(self, config):
        super(Kmeans_Model, self).__init__()
        self.num_classes = config.num_classes
        self.kmeans = KMeans(n_clusters=self.num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten the input tensor
        x = self.kmeans.fit_predict(x)
        return x
