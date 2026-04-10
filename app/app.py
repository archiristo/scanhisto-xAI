import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# MODEL YÜKLE
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# UI
st.title("🧠 Explainable Cancer Detection AI")
st.markdown("### AI-powered medical image analysis with visual explanation")
st.write("Upload a histopathology image")

file = st.file_uploader("Choose an image")

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image")

    input_tensor = transform(img).unsqueeze(0)

    # Prediction
    outputs = model(input_tensor)
    _, pred = torch.max(outputs, 1)

    label = "Cancer" if pred.item() == 1 else "Normal"
    st.subheader(f"Prediction: {label}")

    # Grad-CAM
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    img_np = np.array(img.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    st.image(visualization, caption="Model Attention (Grad-CAM)")
    import torch.nn.functional as F

    probs = F.softmax(outputs, dim=1)
    confidence = probs[0][pred].item()

    st.write(f"Confidence: {confidence:.2f}")

