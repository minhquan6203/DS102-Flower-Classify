import torch
import torchvision
import torchvision.transforms as transforms

class LoadData:
    def __init__(self, config):
        super().__init__(config)

    def load_data(self, data_path, config):
        transform = transforms.Compose([
            transforms.Resize((config.image_H, config.image_W)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(size=(config.image_H, config.image_W), padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))
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
