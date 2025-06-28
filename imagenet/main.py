import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights
from torchvision import models
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from module import transform,ImageNetValDataset,evaluate_imagenet

dataset_dir = "C:/Users/Pankaj Deb Roy/Documents/DeepLearning/llm_models/dataset/ImagenetVal/imagenet_validation_1"
class_pth = "C:/Users/Pankaj Deb Roy/Documents/DeepLearning/llm_models/dataset/ImagenetVal/class_map.csv"
classes = pd.read_csv(class_pth)
model = models.resnet50(pretrained=True)

val_dataset = ImageNetValDataset(dataset_dir,classes,transform)
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=256,shuffle=False,num_workers=8, pin_memory=True)

result = evaluate_imagenet(model, val_loader, device=device)