import os,sys
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights
from torchvision import models,datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.io import decode_image

transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize(256,interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImageNetValDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir,classes:pd.DataFrame, transform:transforms=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes=classes
        for idx, class_name in enumerate(tqdm(sorted(os.listdir(root_dir)))):
            class_path = os.path.join(root_dir, class_name)
            class_id = self.classes[self.classes['class_id']==class_name].id.item()
            
            for img_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, img_name))
                self.labels.append(class_id)
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
def accuracy(output, target, topk=(1,)):
    """Compute the top-k accuracy for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    # Get top maxk predictions
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # Shape: [maxk, batch_size]

    # Compare with ground truth
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # Shape: [maxk, batch_size]

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res  # Returns list: [top1_acc, top5_acc, ...]

def evaluate_imagenet(model, val_loader, device='cuda', topk=(1, 5)):
    model.to(device)
    model.eval()
    topk_correct = [0.0 for _ in topk]
    total_samples = 0

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            accuracies = accuracy(outputs, targets, topk=topk)
            batch_size = images.size(0)
            total_samples += batch_size

            for i, acc in enumerate(accuracies):
                topk_correct[i] += acc.item() * batch_size / 100.0

    # Final top-k accuracies
    final_acc = {f"Top-{k}": (topk_correct[i] / total_samples) * 100 for i, k in enumerate(topk)}
    for k, v in final_acc.items():
        print(f"{k} Accuracy: {v:.2f}%")
    return final_acc