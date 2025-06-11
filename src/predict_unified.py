# src/predict_unified.py

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import os

# ✅ Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

# ✅ Load Frame Model (ResNet18)
frame_model = models.resnet18(weights=None)
frame_model.fc = torch.nn.Linear(frame_model.fc.in_features, 2)
frame_model.load_state_dict(torch.load('models/frame_model.pth', map_location=device))
frame_model = frame_model.to(device).eval()

# ✅ Load Gaze Model (MobileNetV2)
gaze_model = models.mobilenet_v2(weights=None)
gaze_model.classifier[1] = torch.nn.Linear(gaze_model.classifier[1].in_features, 2)
gaze_model.load_state_dict(torch.load('models/gaze_model.pth', map_location=device))
gaze_model = gaze_model.to(device).eval()

# ✅ Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ✅ Prediction function (single image/frame)
def predict_single_image(pil_image):
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    frame_output = frame_model(input_tensor)
    frame_prob = F.softmax(frame_output, dim=1).detach().cpu().numpy()[0]

    gaze_output = gaze_model(input_tensor)
    gaze_prob = F.softmax(gaze_output, dim=1).detach().cpu().numpy()[0]

    frame_pred = frame_prob.argmax()
    gaze_pred = gaze_prob.argmax()

    frame_conf = frame_prob[frame_pred]
    gaze_conf = gaze_prob[gaze_pred]

    # Decision logic
    if frame_pred == gaze_pred:
        final_pred = frame_pred
    else:
        final_pred = frame_pred if frame_conf > gaze_conf else gaze_pred

    return final_pred, max(frame_conf, gaze_conf)

# ✅ Unified Prediction (image or video)
def predict_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()

    preds = []

    if ext in ['.jpg', '.jpeg', '.png']:
        image = Image.open(file_path).convert('RGB')
        pred, conf = predict_single_image(image)
        preds.append(pred)

    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        cap = cv2.VideoCapture(file_path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"🎥 Video frames detected: {frame_count}")

        success, frame = cap.read()
        while success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            pred, conf = predict_single_image(pil_image)
            preds.append(pred)

            success, frame = cap.read()

        cap.release()

    else:
        raise ValueError("Unsupported file type!")

    # Majority voting
    preds_array = np.array(preds)
    real_count = np.sum(preds_array == 0)
    fake_count = np.sum(preds_array == 1)

    if real_count > fake_count:
        final_label = 'REAL'
    else:
        final_label = 'FAKE'

    print(f"\n✅ Final Combined Prediction: {final_label}")
    print(f"🟢 Real Frames: {real_count} | 🔴 Fake Frames: {fake_count}")

    return final_label

# ✅ Test run
if __name__ == '__main__':
    file_path = 'data/SDFVD/videos_fake/vs1.mp4'  # Replace with your image/video path
    predict_file(file_path)
