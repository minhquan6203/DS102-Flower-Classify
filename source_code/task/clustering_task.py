from model.kmeans_model import KMeans_Model
from data_loader.loaddata import LoadData
from sklearn.metrics import f1_score, confusion_matrix,accuracy_score
from utils.builder import build_model
import torch
import os
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
        print('training, please waiting!!!')
        self.base_model.fit(train)
        dump(self.base_model, self.save_path + 'kmeans_model.pkl')
        print("finished training!!!")
        print("now let's see if the training is good or not")
        print("evaluate on train data")
        clusters, y_true = self.base_model.predict(train)

        accuracy = accuracy_score(y_true, clusters)
        f1 = f1_score(y_true, clusters, average='macro')
        cm = confusion_matrix(y_true, clusters)

        print('accuracy: {:.4f}'.format(accuracy))
        print('f1 score: {:.4f}'.format(f1))
        print('confusion matrix:')
        print(cm)


    def evaluate(self):
        test_data = self.dataloader.load_test_data(data_path=self.test_path)
        if os.path.exists(os.path.join(self.save_path, 'kmeans_model.pkl')):
            self.base_model = joblib.load(os.path.join(self.save_path, 'kmeans_model.pkl'))
            print("evaluate model on test datal!!!")
        else:
            print('chưa train model mà đòi test hả?')

        clusters, y_true = self.base_model.predict(test_data)      
        accuracy = accuracy_score(y_true, clusters)
        f1 = f1_score(y_true, clusters, average='macro')
        cm = confusion_matrix(y_true, clusters)

        print('accuracy: {:.4f}'.format(accuracy))
        print('f1 score: {:.4f}'.format(f1))
        print('confusion matrix:')
        print(cm)