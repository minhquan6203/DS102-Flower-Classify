import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.cluster import KMeans
from utils.extract_feature import FeatureExtractor

class KMeans_Model:
    def __init__(self, config):
        self.num_clusters = config.num_clusters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if config.model_extract_name is not None:
            self.feature_extractor = FeatureExtractor(config).to(self.device)
        else:
            self.feature_extractor = None

        self.kmeans = KMeans(n_clusters=self.num_clusters)
        
    def _to_numpy(self, tensor):
        if self.device.type == 'cuda':
            return tensor.detach().cpu().numpy()
        else:
            return tensor.detach().numpy()
    
    def fit(self, dataloader):
        # Extract features
        features = []
        if self.feature_extractor is not None:
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                features.append(self._to_numpy(self.feature_extractor(images)))
        else:
            for images, labels in dataloader:
                images = images.view(images.size(0), -1)
                features.append(self._to_numpy(images))
        features = np.concatenate(features, axis=0)
        # Fit model
        self.kmeans.fit(features)
        
    def predict(self, dataloader):
        # Extract features
        features = []
        y_true = []
        if self.feature_extractor is not None:
            for images, labels in dataloader:
                y_true += labels.tolist()
                images, labels = images.to(self.device), labels.to(self.device)
                features.append(self._to_numpy(self.feature_extractor(images)))
        else:
            for images, labels in dataloader:
                y_true += labels.tolist()
                images = images.view(images.size(0), -1)
                features.append(self._to_numpy(images))
        features = np.concatenate(features, axis=0)
        # Predict clusters
        clusters = self.kmeans.predict(features)
        return clusters, y_true
