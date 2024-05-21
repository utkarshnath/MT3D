import torch
import os
import torchvision.transforms as transforms
from objaverse_folder import ObjaverseDataset
from torch.utils.data import DataLoader
from resnet_gm import ResNet34
from PIL import Image
import numpy as np

# Define the transformation to be applied to the input images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the Objaverse renders dataset
train_dataset = ObjaverseDataset(root='path/to/train/data', transform=transform)
val_dataset = ObjaverseDataset(root="/path/to/val/data", transform=transform)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8)

model = ResNet34().cuda()
state_dict = torch.load(None)['state_dict']
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Define a function to calculate the accuracy of the model
def get_geometric_moment(model, data_loader, mode):
    save_dir = None
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.cuda()  # Move images to GPU if available
            labels = list(labels)

            _, gms = model(images)
            # increase image channels from 1 to 3
            gms = gms.repeat(1,3,1,1)
            gms = gms * 255.
            gms = gms.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            for label, gm in zip(labels, gms):
                Image.fromarray(gm).save(f"{save_dir}/{label}.png")

get_geometric_moment(model, train_loader, "train")
get_geometric_moment(model, val_loader, "val")

