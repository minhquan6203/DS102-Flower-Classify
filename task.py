import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import argparse

from model import CNN_Model
from loaddata import LoadData


import os
import argparse
class Classify_task:
    def __init__(self, config):
      self.num_epochs=config.num_epochs
      self.image_C=config.image_C
      self.image_W=config.image_W
      self.image_H=config.image_H
      self.train_path=config.train_path
      self.valid_path=config.valid_path
      self.batch_size=config.batch_size
      self.learning_rate=config.learning_rate
      self.num_classes=config.num_classes
      self.save_path=config.save_path
      self.load_data=LoadData(config)
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.base_model = CNN_Model(config).to(self.device)
  
    def training(self):
        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)

        train = self.load_data(data_path=self.train_path)
        valid = self.load_data(data_path=self.valid_path)

        
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate)
        if os.path.exists(os.path.join(self.save_path, 'last_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'last_model.pth'))
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            valid_acc = checkpoint['valid_acc']
            train_acc=checkpoint['train_acc']
            train_loss=checkpoint['train_loss']
            valid_loss=checkpoint['valid_loss']
            print('Loaded the last saved model.')
            initial_epoch = checkpoint['epoch'] + 1
            print(f"continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("first time training!!!")
            valid_acc=0.
            train_acc=0.
            train_loss=0.
            valid_loss=0.
          
        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'))
            best_valid_acc=checkpoint['valid_acc']
        else:
            best_valid_acc = 0.0
            
        threshold=0
        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            for images, labels in train:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self.base_model(images)
                loss = loss_function(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_acc += (output.argmax(1) == labels).sum().item() / labels.size(0)

            with torch.no_grad():
                for images, labels in valid:
                    images, labels = images.to(self.device), labels.to(self.device)
                    output = self.base_model(images)
                    loss = loss_function(output, labels)
                    valid_loss += loss.item()
                    valid_acc += (output.argmax(1) == labels).sum().item() / labels.size(0)

            train_loss /= len(train)
            train_acc /= len(train)
            valid_loss /= len(valid)
            valid_acc /= len(valid)

            print(f"Epoch {epoch + 1}/{self.num_epochs + initial_epoch}")
            print(f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}")
            print(f"Valid Loss: {valid_loss:.4f} Valid Acc: {valid_acc:.4f}")

            # save the model state dict
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.base_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_acc': valid_acc,
                'train_acc':train_acc,
                'train_loss':train_loss,
                'valid_loss':valid_loss}, os.path.join(self.save_path, 'last_model.pth'))

            # save the best model

            if epoch > 0 and valid_acc < best_valid_acc:
              threshold+=1
            else:
              threshold=0
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.base_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_acc': valid_acc,
                    'train_acc':train_acc,
                    'train_loss':train_loss,
                    'valid_loss':valid_loss}, os.path.join(self.save_path, 'best_model.pth'))
                print(f"Saved the best model with validation accuracy of {valid_acc:.4f}")
            
            # early stopping
            if threshold>=5:
                print(f"Early stopping after epoch {epoch + 1}")
                break

            
    