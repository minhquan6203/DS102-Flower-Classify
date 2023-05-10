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
        self.max_iter = config.max_iter
        
        if config.model_extract_name is not None:
            self.feature_extractor = FeatureExtractor(config).to(self.device)
        else:
            self.feature_extractor = None

        self.kmeans = KMeans(n_clusters=self.num_clusters, max_iter=self.max_iter, tol=0.0001, verbose=1)
        
    def _to_numpy(self, tensor):
        if self.device.type == 'cuda':
            return tensor.detach().cpu().numpy()
        else:
            return tensor.detach().numpy()
        
    def get_features(self, dataloader):
        features = []
        y_true = []
        if self.feature_extractor is not None:
            for images, labels in dataloader:
                y_true.append(self._to_numpy(labels))
                images = images.to(self.device)
                features.append(self._to_numpy(self.feature_extractor(images)))
        else:
            for images, labels in dataloader:
                images = images.view(images.size(0), -1)
                y_true.append(self._to_numpy(labels))
                features.append(self._to_numpy(images))
        features = np.concatenate(features, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        return features, y_true
    
    def fit(self, features, y_true = None):
        self.kmeans.fit(features, y_true)
        
    def predict(self, features):
        clusters = self.kmeans.predict(features)
        return clusters
