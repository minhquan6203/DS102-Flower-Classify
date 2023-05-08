from model import KMeans_Model
from loaddata import LoadData
from sklearn.metrics import f1_score, confusion_matrix,accuracy_score
from build import build_model
import torch
import os
from joblib import dump
import joblib

class Clustering_Task:
    def __init__(self,config):
        self.train_path=config.train_path
        self.valid_path=config.valid_path
        self.test_path=config.test_path
        self.save_path=config.save_path
        self.dataloader=LoadData(config)
        self.base_model=build_model(config)

    def training(self):
        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)

        train = self.dataloader.load_data(data_path=self.train_path)
        print('training!!!')
        self.base_model.fit(train)
        dump(self.base_model, self.save_path + 'kmeans_model.pkl')


    def evaluate(self):
        test_data = self.dataloader.load_test_data(data_path=self.test_path)
        if os.path.exists(os.path.join(self.save_path, 'kmeans_model.pkl')):
            self.base_model = joblib.load(os.path.join(self.save_path, 'kmeans_model.pkl'))
        else:
            print('chưa train model mà đòi test hả?')
        y_true, y_pred = [], []
        for images, labels in test_data:
            outputs = self.base_model.predict(images)

            _, preds = torch.max(outputs, 1)
            y_true += labels.tolist()
            y_pred += preds.tolist()

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        print('Test accuracy: {:.4f}'.format(accuracy))
        print('Test F1 score: {:.4f}'.format(f1))
        
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion matrix:')
        print(cm)



