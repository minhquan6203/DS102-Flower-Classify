from model.kmeans_model import KMeans
from data_loader.loaddata import LoadData
from sklearn.metrics import silhouette_score, accuracy_score
from utils.builder import build_model
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import numpy as np
from joblib import dump
import joblib

class Clustering_Task:
    def __init__(self,config):
        self.train_path = config.train_path
        self.valid_path = config.valid_path
        self.test_path = config.test_path
        self.save_path = config.save_path
        self.dataloader = LoadData(config)
        self.base_model = build_model(config)

    def training(self):
        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)

        train = self.dataloader.load_data(data_path = self.train_path)
        features, labels = self.base_model.get_features(train)
        print('training, please waiting!!!')
        self.base_model.fit(features)
        print("finished training!!!")
        torch.save({'model_state_dict': self.base_model.state_dict()},os.path.join(self.save_path,'kmeans_model.pth'))

    def evaluate(self):
        test_data = self.dataloader.load_test_data(data_path = self.test_path)
        features, labels = self.base_model.get_features(test_data)
        if os.path.exists(os.path.join(self.save_path, 'kmeans_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'kmeans_model.pth'), map_location=self.device)
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            print("evaluate model on test datal!!!")
        else:
            print('chưa train model mà đòi test hả?')

        clusters = self.base_model.predict(features)
    
        print('acc: ',accuracy_score(clusters,labels))
    

