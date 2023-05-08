import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import ViTModel
from sklearn.cluster import KMeans
import numpy as np

class ViT_Model(nn.Module):
    def __init__(self, config):
        super(ViT_Model, self).__init__()
        self.vit = ViTModel.from_pretrained(config.model_name_or_path)
        self.classifier = nn.Linear(self.vit.config.hidden_size, config.num_classes)
       
    def forward(self, x):
        # x is an input image tensor of shape [batch_size, channels, height, width]
        x = self.vit(x)
        # only use the first output from ViT, which is the cls_token representation
        x = x.last_hidden_state[:, 0]
        logits = self.classifier(x)
        logits = torch.softmax(logits, dim=1)
        return logits


class ResNet34_Model(nn.Module): #use ResNet34

    def __init__(self, config):
        super(ResNet34_Model, self).__init__()
        self.num_classes = config.num_classes
        self.resnet = models.resnet34(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.softmax(x, dim=1)
        return x

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
    
class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()
        self.input_shape = (config.image_C,config.image_W, config.image_H)
        if config.model_extract_name == 'vgg16':
            self.cnn = models.vgg16(pretrained=True)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        elif config.model_extract_name == 'alexnet':
            self.cnn = models.alexnet(pretrained=True)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        elif config.model_extract_name == 'resnet34':
            self.cnn = models.resnet34(pretrained=True)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        else:
            print(f"đéo hỗ trợ model này: {config.model_extract_name}")

    def forward(self, x):
        features = self.cnn(x)
        # Flatten the features
        features = features.view(features.size(0), -1)
        return features
    
    def output_size(self):
        # Get the output size of the feature extractor
        with torch.no_grad():
            output = self.forward(torch.zeros(1, *self.input_shape))
        return output.size(1)


class SVM_Model(nn.Module):
    def __init__(self, config):
        super(SVM_Model, self).__init__()
        self.feature_extractor = FeatureExtractor(config)
        self.num_classes = config.num_classes
        self.kernel_type = config.kernel_type
        self.gamma = config.gamma
        self.degree = config.degree
        
        if self.kernel_type == 'linear':
            self.classifier = nn.Linear(self.feature_extractor.output_size(), self.num_classes, bias=False)
        elif self.kernel_type == 'rbf':
            self.classifier = RBFSVM(self.feature_extractor.output_size(), self.num_classes, self.gamma)
        elif self.kernel_type == 'poly':
            self.classifier = PolySVM(self.feature_extractor.output_size(), self.num_classes, self.gamma, self.degree)
        else:
            raise ValueError('Invalid kernel type')

    def forward(self, x):
        x = self.feature_extractor(x)
        out = self.classifier(x)
        return out

class RBFSVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma):
        super(RBFSVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.support_vectors = nn.Parameter(torch.randn(1, input_size))
        self.coefficients = nn.Parameter(torch.randn(1, num_classes))

    def forward(self, x):
        # Compute kernel matrix
        diff = x.unsqueeze(1) - self.support_vectors.unsqueeze(0)
        norm = diff.norm(dim=-1)
        K = torch.exp(-self.gamma * norm ** 2)
        # Compute decision function
        f = (K * self.coefficients).sum(dim=1)

        return f

class PolySVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma, degree):
        super(PolySVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.degree = degree
        self.support_vectors = nn.Parameter(torch.randn(1, input_size))
        self.coefficients = nn.Parameter(torch.randn(1, num_classes))

    def forward(self, x):
        # Compute kernel matrix
        dot = (x.unsqueeze(1) * self.support_vectors.unsqueeze(0)).sum(dim=-1)
        K = (self.gamma * dot + 1) ** self.degree
        # Compute decision function
        f = (K * self.coefficients).sum(dim=1)
        return f


class KMeans_Model:
    def __init__(self, config):
        self.num_clusters = config.num_clusters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = FeatureExtractor(config).to(self.device)
        self.kmeans = KMeans(n_clusters=config.num_clusters)
        
    def _to_numpy(self, tensor):
        if self.device.type == 'cuda':
            return tensor.detach().cpu().numpy()
        else:
            return tensor.detach().numpy()
    
    def fit(self, dataloader):
        # Extract features
        features = []
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            features.append(self._to_numpy(self.feature_extractor(images)))
        features = np.concatenate(features)
        
        # Fit model 
        self.kmeans.fit(features)
        
    def predict(self, dataloader):
        # Extract features
        features = []
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            features.append(self._to_numpy(self.feature_extractor(images)))

        features = np.concatenate(features)
        clusters = self.kmeans.predict(features)

        return clusters


