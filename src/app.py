# app.py

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import tempfile
import numpy as np
from PIL import Image
import os

# --- Device config ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model load function ---
def load_model(model_path, model_type='resnet'):
    if model_type == 'resnet':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_type == 'mobilenet':
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

# --- Image transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Prediction function ---
def predict_image(image, model):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# --- Frame extractor ---
def extract_frames_from_video(video_path, frame_rate=1):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps // frame_rate) if fps >= frame_rate else 1

    count = 0
    success, frame = cap.read()
    while success:
        if count % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
        success, frame = cap.read()
        count += 1
    cap.release()
    return frames

# --- Combined prediction ---
def combined_prediction(image_or_frames, frame_model, gaze_model, is_video=False):
    real_count = 0
    fake_count = 0

    items = image_or_frames if is_video else [image_or_frames]

    for img in items:
        pred_frame = predict_image(img, frame_model)
        pred_gaze = predict_image(img, gaze_model)

        final_pred = max(pred_frame, pred_gaze)  # 1 = fake, 0 = real → conservatively detect fake
        if final_pred == 1:
            fake_count += 1
        else:
            real_count += 1

    if fake_count > real_count:
        return "FAKE", real_count, fake_count
    else:
        return "REAL", real_count, fake_count

# --- Streamlit Config ---
st.set_page_config(page_title="Deepfake Detection", layout='wide')
st.title("🕵️ Deepfake Detection System")
st.markdown("Upload an image or video to detect deepfakes using **Frame** and **Gaze** models.")

# --- Upload section ---
uploaded_file = st.file_uploader("Upload Image or Video", type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi'])

if uploaded_file is not None:
    file_type = uploaded_file.type
    is_video = 'video' in file_type

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # Load models
    frame_model = load_model('models/frame_model.pth', model_type='resnet')
    gaze_model = load_model('models/gaze_model.pth', model_type='mobilenet')

    if is_video:
        st.video(uploaded_file)
        st.info("Extracting frames from video...")
        frames = extract_frames_from_video(temp_path, frame_rate=1)  # Process 1 fps
        st.success(f"{len(frames)} frames extracted.")
        final_pred, real_c, fake_c = combined_prediction(frames, frame_model, gaze_model, is_video=True)
    else:
        image = Image.open(temp_path).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        final_pred, real_c, fake_c = combined_prediction(image, frame_model, gaze_model)

    # --- Result ---
    st.subheader("📝 Final Prediction")
    if final_pred == "FAKE":
        st.error(f"Prediction: **FAKE** ❌\n\nReal Frames: {real_c} | Fake Frames: {fake_c}")
    else:
        st.success(f"Prediction: **REAL** ✅\n\nReal Frames: {real_c} | Fake Frames: {fake_c}")

    # Cleanup temp file
    os.remove(temp_path)
