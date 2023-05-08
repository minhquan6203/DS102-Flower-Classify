import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import ViTModel

class ViT_Model(nn.Module):
    def __init__(self, config):
        super(ViT_Model, self).__init__()
        self.vit = ViTModel.from_pretrained(config.model_name_or_path)
        self.classifier = nn.Linear(self.vit.config.hidden_size, config.num_classes)
       
    def forward(self, x):
        # shape of x [batch_size, channels, height, width]
        x = self.vit(x)
        x = x.last_hidden_state[:, 0]
        logits = self.classifier(x)
        logits = torch.softmax(logits, dim=1)
        return logits
