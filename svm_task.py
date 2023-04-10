from model import SVM_Model
from loaddata import LoadData
from sklearn.metrics import f1_score, confusion_matrix
import os
import joblib

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
        train_data, train_labels = self.dataloader.load_data(data_path=self.train_path)
        valid_data, valid_labels = self.dataloader.load_data(data_path=self.valid_path)

        self.model.fit(train_data, train_labels)
        train_acc = (self.model.predict(train_data) == train_labels).mean()
        valid_acc = (self.model.predict(valid_data) == valid_labels).mean()

        print(f"train accuracy: {train_acc:.4f}")
        print(f"valid accuracy: {valid_acc:.4f}")

        # save the model
        joblib.dump(self.model, os.path.join(self.save_path, 'model.joblib'))

    def evaluate(self):
        if os.path.exists(os.path.join(self.save_path, 'model.joblib')):
            self.model = joblib.load(os.path.join(self.save_path, 'model.joblib'))
        else:
            print('chưa train model mà đòi test')
        test_data, test_labels = self.dataloader.load_data(data_path=self.test_path)
        test_predictions = self.model.predict(test_data)
        test_acc = (test_predictions == test_labels).mean()
        print(f"test accuracy: {test_acc:.4f}")

        # F1 score
        f1 = f1_score(test_labels, test_predictions, average='macro')
        print(f"test F1 score: {f1:.4f}")

        #confusion matrix
        cm = confusion_matrix(test_labels, test_predictions)
        print("confusion matrix:")
        print(cm)
