import torch
import torchvision
import torchvision.transforms as transforms

class LoadData:
    def __init__(self, config):
        self.image_H = config.image_H
        self.image_W = config.image_W
        self.batch_size = config.batch_size

    def load_data(self, data_path):
        transform = transforms.Compose([
            transforms.Resize((self.image_H, self.image_W)),

            transforms.RandomHorizontalFlip(),  # randomly flip the image horizontally
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # randomly adjust color
            transforms.RandomCrop((self.image_H - 20, self.image_W - 20)),  # randomly crop the image
            transforms.RandomResizedCrop((self.image_H, self.image_W), scale=(0.8, 1.0)),            
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize pixel values
        ])

        dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=transform
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

        return dataloader
        
    def load_test_data(self, data_path):
        transform = transforms.Compose([
            transforms.Resize((self.image_H, self.image_W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize pixel values
        ])

        dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=transform
        )

        test_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

        return test_dataloader