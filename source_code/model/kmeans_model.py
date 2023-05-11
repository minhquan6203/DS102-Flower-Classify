import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from utils.kmeans_pytorch import KMeans
from utils.extract_feature import FeatureExtractor

class KMeans_Model:
    def __init__(self, config):
        self.num_clusters = config.num_clusters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_iter = config.max_iter
        self.mini_batch = config.mini_batch
        
        if config.model_extract_name is not None:
            self.feature_extractor = FeatureExtractor(config).to(self.device)
        else:
            self.feature_extractor = None

        self.kmeans = KMeans(n_clusters=self.num_clusters, max_iter=self.max_iter, tol=0.0001, verbose=1, minibatch = self.mini_batch)
              
    def get_features(self, dataloader):
        features = []
        y_true = []
        if self.feature_extractor is not None:
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                y_true.append(labels)
                features.append(self.feature_extractor(images))
        else:
            for images, labels in dataloader:
                images = images.view(images.size(0), -1)
                images, labels = images.to(self.device), labels.to(self.device)
                y_true.append(labels)
                features.append(images)
        features = torch.cat(features, dim=0)
        y_true = torch.cat(y_true, dim=0)
        return features, y_true
    
    def fit(self, features):
        self.kmeans.fit(features)
        
    def predict(self, features):
        clusters = self.kmeans.predict(features)
        return clusters
