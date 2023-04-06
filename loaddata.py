import torch
import torchvision
import torchvision.transforms as transforms

class LoadData:
    def __init__(self, config):
            super().__init__(config)

    def load_data(self, data_path,config):
        transform = transforms.Compose([
            transforms.Resize((config.image_H, config.image_W)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=transform
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2
        )

        return dataloader
        
    def __call__(self, data_path):
        load_data = self.load_data(data_path)
        return load_data
