import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.extract_feature import FeatureExtractor

class SVM_Model(nn.Module):
    def __init__(self, config):
        super(SVM_Model, self).__init__()
        self.num_classes = config.num_classes
        self.image_W = config.image_W
        self.image_H = config.image_H
        self.image_C = config.image_C
        self.model_extract_name = config.model_extract_name
        self.kernel_type = config.kernel_type
        self.gamma = config.gamma
        self.degree = config.degree
        self.r = config.r
        if self.model_extract_name is not None:
            self.feature_extractor = FeatureExtractor(config)
            if self.kernel_type == 'linear':
                self.classifier = LinearSVM(self.feature_extractor.output_size(), self.num_classes)
            elif self.kernel_type == 'rbf':
                self.classifier = RBFSVM(self.feature_extractor.output_size(), self.num_classes, self.gamma)
            elif self.kernel_type == 'poly':
                self.classifier = PolySVM(self.feature_extractor.output_size(), self.num_classes, self.gamma, self.r, self.degree)
            elif self.kernel_type == 'sigmoid':
                self.classifier = PolySVM(self.feature_extractor.output_size(), self.num_classes, self.gamma, self.r)
            elif self.kernel_type == 'custom':
                self.classifier = CustomSVM(self.feature_extractor.output_size(), self.num_classes, self.gamma, self.r, self.degree)
            else:
                raise ValueError('không hỗ trợ kernel này')
        else:
            self.feature_extractor = None
            if self.kernel_type == 'linear':
                self.classifier = LinearSVM(self.image_H*self.image_W*self.image_C, self.num_classes)
            elif self.kernel_type == 'rbf':
                self.classifier = RBFSVM(self.image_H*self.image_W*self.image_C, self.num_classes, self.gamma)
            elif self.kernel_type == 'poly':
                self.classifier = PolySVM(self.image_H*self.image_W*self.image_C, self.num_classes, self.gamma, self.r, self.degree)
            elif self.kernel_type == 'sigmoid':
                self.classifier = PolySVM(self.image_H*self.image_W*self.image_C, self.num_classes, self.gamma, self.r) 
            elif self.kernel_type == 'custom':
                self.classifier = CustomSVM(self.image_H*self.image_W*self.image_C, self.num_classes, self.gamma, self.r, self.degree)
            else:
                raise ValueError('không hỗ trợ kernel này')
            

    def forward(self, x):
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
        else:
            x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


class LinearSVM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearSVM, self).__init__()
        self.num_classes = num_classes
        self.input_size=input_size
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, x):
        outputs = torch.matmul(x, self.weights.t()) + self.bias
        return outputs


class RBFSVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma):
        super(RBFSVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, x):
        dists = torch.cdist(x, self.weights, p=2)
        dists_normalized = (dists - torch.mean(dists)) / torch.std(dists)
        kernel_matrix = torch.exp(-self.gamma * dists_normalized ** 2)
        outputs = kernel_matrix  + self.bias
        return outputs


class PolySVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma, r, degree):
        super(PolySVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.degree = degree
        self.r = r
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, x):
        # dists = torch.cdist(x, self.weights, p=2)
        # kernel_matrix = (self.gamma * dists + self.r) ** self.degree #này sai công thức nhưng cho kết quả tốt?
        kernel_matrix = (self.gamma * torch.mm(x, self.weights.t()) + self.r) ** self.degree
        outputs = kernel_matrix + self.bias
        return outputs
    

class SigmoidSVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma, r):
        super(SigmoidSVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.r = r
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, x):
        kernel_matrix = torch.tanh(self.gamma * torch.mm(x, self.weights.t())+ self.r)
        outputs = kernel_matrix  + self.bias
        return outputs


class CustomSVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma, r, degree):
        super(CustomSVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.degree = degree
        self.r = r
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, x):
        dists = torch.cdist(x, self.weights, p=2)
        kernel_matrix = (self.gamma * dists + self.r) ** self.degree #này sai công thức nhưng cho kết quả tốt?
        outputs = kernel_matrix + self.bias
        return outputs