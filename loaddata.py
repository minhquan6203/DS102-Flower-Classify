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
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(size=(self.image_H, self.image_W), padding=4),
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
        
    def __call__(self, data_path):
        load_data = self.load_data(data_path)
        return load_data
