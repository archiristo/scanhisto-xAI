## ScanHisto xAI
Explainable Cancer Detection from Histopathology Images

## Overview
ScanHistoAI is a deep learning-based web application that detects metastatic cancer in histopathology images and provides visual explanations using Grad-CAM.

## Features
🧬 Binary classification (Cancer / Normal)

🔥 Grad-CAM heatmap visualization

🌐 Interactive web app (Streamlit)

📊 ~83% validation accuracy

## Demo
Upload an image and get:

-Prediction

-Confidence score

-Visual explanation (heatmap)

## Model
ResNet50 (fine-tuned)
Trained on histopathologic cancer dataset

## Run Locally
```bash
pip install -r requirements.txt
```
```python
streamlit run app/app.py
```

## Story

This project started as a high school idea in 2022 but was never submitted.  
Years later, it has been rebuilt into a full explainable AI system.

