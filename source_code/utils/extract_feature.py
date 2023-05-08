import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()
        self.input_shape = (config.image_C,config.image_H, config.image_W)
        if config.model_extract_name == 'vgg16':
            self.cnn = models.vgg16(pretrained=True)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        elif config.model_extract_name == 'alexnet':
            self.cnn = models.alexnet(pretrained=True)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        elif config.model_extract_name == 'resnet34':
            self.cnn = models.resnet34(pretrained=True)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        else:
            print(f"chưa hỗ trợ model này: {config.model_extract_name}")

    def forward(self, x):
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        return features
    
    def output_size(self):
        with torch.no_grad():
            output = self.forward(torch.zeros(1, *self.input_shape))
        return output.size(1)