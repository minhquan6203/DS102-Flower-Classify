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

    def training_and_eval(self):
        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)

        train = self.dataloader.load_data(data_path = self.train_path)
        print('load features...')
        features, labels = self.base_model.get_features(train)
        print('training, please waiting!!!')
        self.base_model.fit(features)
        print("finished training...")
        print('let see if training is good or not')
        print('predicting...')
        clusters = self.base_model.predict(features)
        sil_score = silhouette_score(features.cpu(), clusters.cpu(), metric='euclidean')
        print("silhouette score",sil_score)

        print('now, evaluate on test data')
        test = self.dataloader.load_data(data_path = self.test_path)
        print('load features...')
        test_f,test_l = self.base_model.get_features(test)
        print('predicting...')
        t_clusters = self.base_model.predict(test_f)
        t_sil_score = silhouette_score(test_f.cpu(), t_clusters.cpu(), metric='euclidean')
        print("silhouette score",t_sil_score)
       


    

