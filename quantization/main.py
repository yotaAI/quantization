import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
# Quantization modules
from quantizer_simulator import quantize_resnet
from validation import transform,ImageNetValDataset,evaluate_imagenet
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset_dir = "C:/Users/Pankaj Deb Roy/Documents/DeepLearning/llm_models/dataset/ImagenetVal/imagenet_validation_1"
class_pth = "C:/Users/Pankaj Deb Roy/Documents/DeepLearning/llm_models/dataset/ImagenetVal/class_map.csv"
classes = pd.read_csv(class_pth)


val_dataset = ImageNetValDataset(dataset_dir,classes,transform)
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=256,shuffle=False,num_workers=8, pin_memory=True)

model_fp32 = models.resnet50(pretrained=True).eval()

model_q = quantize_resnet(model_fp32,bits=2,signed=False)

result = evaluate_imagenet(model_q, val_loader, device=device)