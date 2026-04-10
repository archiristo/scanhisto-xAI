import torch
import torchvision.models as models
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# modeli yükle
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("model.pth"))
model.eval()

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# image yükle
img = Image.open("data/train/cancer/0a1d3ed6b8ee989a493d71c017cf3306f17c43cf.tif")
input_tensor = transform(img).unsqueeze(0)

# Grad-CAM
target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

grayscale_cam = cam(input_tensor=input_tensor)[0]

# görselleştirme
img_np = np.array(img.resize((224, 224))) / 255.0
visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

plt.imshow(visualization)
plt.axis("off")
plt.show()
