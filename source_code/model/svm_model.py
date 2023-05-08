import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.extract_feature import FeatureExtractor

class SVM_Model(nn.Module):
    def __init__(self, config):
        super(SVM_Model, self).__init__()

        if config.model_extract_name is not None:
            self.feature_extractor = FeatureExtractor(config)
            if config.kernel_type == 'linear':
                self.classifier = nn.Linear(self.feature_extractor.output_size(), config.num_classes)
            elif config.kernel_type == 'rbf':
                self.classifier = RBFSVM(self.feature_extractor.output_size(), config.num_classes, config.gamma)
            elif config.kernel_type == 'poly':
                self.classifier = PolySVM(self.feature_extractor.output_size(), self.num_classes, config.gamma, config.degree)
            else:
                raise ValueError('không hỗ trợ kernel này')
        else:
            self.feature_extractor = None
            if config.kernel_type == 'linear':
                self.classifier = nn.Linear(config.image_H*config.image_W*config.image_C, self.num_classes)
            elif config.kernel_type == 'rbf':
                self.classifier = RBFSVM(config.image_H*config.image_W*config.image_C, self.num_classes, config.gamma)
            elif config.kernel_type == 'poly':
                self.classifier = PolySVM(config.image_H*config.image_W*config.image_C, self.num_classes, config.gamma, config.degree)
            else:
                raise ValueError('không hỗ trợ kernel này')

    def forward(self, x):
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
        else:
            x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

class RBFSVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma):
        super(RBFSVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.weights = nn.Parameter(torch.randn(num_classes, input_size))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        x = x.view(-1, self.input_size)
        dists = torch.cdist(x, self.weights, p=2)
        outputs = torch.exp(-self.gamma * dists ** 2) + self.bias
        return outputs


class PolySVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma, degree):
        super(PolySVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.degree = degree
        self.weights = nn.Parameter(torch.randn(num_classes, input_size))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        x = x.view(-1, self.input_size)
        dists = torch.cdist(x, self.weights, p=2)
        kernel_matrix = (self.gamma * dists + 1) ** self.degree
        outputs = kernel_matrix + self.bias
        return outputs