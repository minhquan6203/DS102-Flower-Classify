from model import SVM_Model
from loaddata import LoadData
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import os
import joblib
import numpy as np


class SVM_Classify_task:
    def __init__(self, config):
        self.train_path=config.train_path
        self.valid_path=config.valid_path
        self.test_path=config.test_path
        self.num_classes=config.num_classes
        self.save_path=config.save_path
        self.dataloader=LoadData(config)
        self.model=SVM_Model(config)
  
    def training(self):
        train_score=[]
        valid_score=[]
        train = self.dataloader.load_data(data_path=self.train_path)
        valid = self.dataloader.load_data(data_path=self.valid_path)

        for train_data, train_labels in train:
            self.model.fit(train_data.reshape(len(train_data), -1), train_labels)
            train_preds = self.model.predict(train_data.reshape(len(train_data), -1))
            train_acc = accuracy_score(train_labels, train_preds)
            train_score.append(train_acc)
        for valid_data, valid_labels in valid:
            valid_preds = self.model.predict(valid_data.reshape(len(valid_data), -1))
            valid_acc = accuracy_score(valid_labels, valid_preds)
            valid_score.append(valid_acc)
        print(f"train accuracy: {sum(train_score)/len(train_score):.4f}")
        print(f"valid accuracy: {sum(valid_score)/len(valid_score):.4f}")

        # save the model
        joblib.dump(self.model, os.path.join(self.save_path, 'model.joblib'))

    def evaluate(self):
        if os.path.exists(os.path.join(self.save_path, 'model.joblib')):
            self.model = joblib.load(os.path.join(self.save_path, 'model.joblib'))
        else:
            print('chưa train mà đòi test à')
            return
        
        test = self.dataloader.load_data(data_path=self.test_path)
        test_score=[]
        test_f1=[]
        test_pred=[]
        test_true=[]

        for test_data, test_labels in test:
            test_pred.extend(self.model.predict(test_data.reshape(len(test_data), -1)))
            test_true.extend(test_labels)

            test_acc = (self.model.predict(test_data.reshape(len(test_data), -1)) == test_labels).mean()
            test_score.append(test_acc)

            # F1 score
            f1 = f1_score(test_labels, self.model.predict(test_data.reshape(len(test_data), -1)), average='macro')
            test_f1.append(f1)

        print(f"Test accuracy: {np.mean(test_score):.4f}")
        print(f"Test F1 score: {np.mean(test_f1):.4f}")

        # Compute confusion matrix
        cm = confusion_matrix(test_true, test_pred)
        print(f"Confusion matrix:\n{cm}")

