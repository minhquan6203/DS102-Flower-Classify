import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoFeatureExtractor, AutoModel

class ViT_Model(nn.Module):
    def __init__(self, config):
        super(ViT_Model, self).__init__()
        self.model_name_or_path = config.model_name_or_path
        self.num_classes = config.num_classes
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name_or_path)
        self.backbone = AutoModel.from_pretrained(self.model_name_or_path)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, self.num_classes)
        self.dropout = nn.Dropout(0.3)

        for param in self.backbone.parameters():
            param.requires_grad = False        
       
    def forward(self, x):
        # shape of x [batch_size, channels, height, width]
        inputs = self.feature_extractor(x, return_tensors="pt")
        features = self.backbone(**inputs).last_hidden_state
        out = self.classifier(self.dropout(features))
        out = torch.softmax(out, dim=1)
        return out


