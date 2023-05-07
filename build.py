import torch
import torch.nn as nn
from model import CNN_Model,SVM_Model,LeNet5,NN,ViT_Model,KMeans_Model

def build_model(config):
    if config.type_model=="SVM":
        return SVM_Model(config)
    if config.type_model=='CNN':
        return CNN_Model(config)
    if config.type_model=="LeNet5":
        return LeNet5(config)
    if config.type_model=="NN":
        return NN(config)
    if config.type_model=="Transformer":
        return ViT_Model(config)
    if config.config.type_model=="Kmeans":
        return KMeans_Model(config)
   

def build_loss_fn(config):
    if config.type_model=="SVM":
        return nn.MultiMarginLoss()
    if config.type_model=='CNN':
        return nn.CrossEntropyLoss()
    if config.type_model=="LeNet5":
        return nn.CrossEntropyLoss()
    if config.type_model=="NN":
        return nn.CrossEntropyLoss()
    if config.type_model=="Transformer":
        return nn.CrossEntropyLoss()