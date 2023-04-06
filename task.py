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
         super().__init__(config)

    def training(self,config):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transform = transforms.Compose([
            transforms.Resize((config.image_W, config.image_H)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))
        ])
        load_data = LoadData(transform=transform)

        train = load_data(data_path=config.train_path, batch_size=config.batch_size)
        valid = load_data(data_path=config.valid_path, batch_size=config.batch_size)


        best_valid_acc = 0.0
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(base_model.parameters(), lr=config.learning_rate)
        if os.path.exists(os.path.join(config.save_path, 'last_model.pt')):
            checkpoint = torch.load(os.path.join(config.save_path, 'last_model.pt'))
            base_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded the last saved model.')
            initial_epoch = checkpoint['epoch'] + 1
            print(f"continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("first time training!!!")
            base_model = CNN_Model(config.type_model, config.image_C, config.image_W, config.image_H, config.num_classes).to(device)
            
            
        for epoch in range(initial_epoch, config.num_epochs + initial_epoch):
            train_loss, train_acc = 0, 0
            for images, labels in train:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                output = base_model(images)
                loss = loss_function(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_acc += (output.argmax(1) == labels).sum().item() / labels.size(0)

            valid_loss, valid_acc = 0, 0
            with torch.no_grad():
                for images, labels in valid:
                    images, labels = images.to(device), labels.to(device)
                    output = base_model(images)
                    loss = loss_function(output, labels)
                    valid_loss += loss.item()
                    valid_acc += (output.argmax(1) == labels).sum().item() / labels.size(0)

            train_loss /= len(train)
            train_acc /= len(train)
            valid_loss /= len(valid)
            valid_acc /= len(valid)

            print(f"Epoch {epoch + 1}/{config.num_epochs + initial_epoch}")
            print(f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}")
            print(f"Valid Loss: {valid_loss:.4f} Train Acc: {valid_acc:.4f}")

            # save the model state dict
            torch.save(base_model.state_dict(), os.path.join(config.save_path, 'last_model.pt'))

            # save the best model
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(base_model.state_dict(), os.path.join(config.save_path, 'best_model.pt'))
                print(f"Saved the best model with validation accuracy of {valid_acc:.4f}")

            # early stopping
            if epoch > 0 and valid_acc < prev_valid_acc:
                print(f"Early stopping after epoch {epoch + 1}")
                break

            prev_valid_acc = valid_acc
    
    def parse_args(parser):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config-file", type=str, required=True)
        args = parser.parse_args()
        return args

    def __call__(self, config):
        self.training(config)