import torch # type: ignore
import torch.nn as nn # type: ignore
import cv2 # type: ignore
from PIL import Image # type: ignore
from torchvision import models, transforms # type: ignore
import numpy as np # type: ignore

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transform for pre-processing the input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Function to load a model (ResNet or MobileNet)
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

# Predict the image using the given model
def predict_image(image, model):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probabilities = nn.Softmax(dim=1)(outputs)  # Apply softmax to get class probabilities
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item(), probabilities

# Extract frames from the video for prediction
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

# Main function to predict deepfake from image/video
def predict_deepfake(file_path, is_video=False):
    frame_model = load_model('models/frame_model.pth', model_type='resnet')
    gaze_model = load_model('models/gaze_model.pth', model_type='mobilenet')

    if is_video:
        inputs = extract_frames_from_video(file_path)
    else:
        image = Image.open(file_path).convert('RGB')
        inputs = [image]

    frame_confidences = []
    gaze_confidences = []

    real_count = fake_count = 0
    frame_total_conf = gaze_total_conf = 0

    for img in inputs:
        frame_pred = frame_model(transform(img).unsqueeze(0).to(device))
        gaze_pred = gaze_model(transform(img).unsqueeze(0).to(device))

        frame_probs = torch.nn.functional.softmax(frame_pred, dim=1)
        gaze_probs = torch.nn.functional.softmax(gaze_pred, dim=1)

        frame_conf, frame_class = torch.max(frame_probs, 1)
        gaze_conf, gaze_class = torch.max(gaze_probs, 1)

        frame_conf_val = frame_conf.item()
        gaze_conf_val = gaze_conf.item()

        frame_confidences.append(frame_conf_val * 100)
        gaze_confidences.append(gaze_conf_val * 100)

        frame_total_conf += frame_conf_val
        gaze_total_conf += gaze_conf_val

        final_pred = max(frame_class.item(), gaze_class.item())
        if final_pred == 1:
            fake_count += 1
        else:
            real_count += 1

    total_predictions = real_count + fake_count
    avg_frame_conf = frame_total_conf / total_predictions if total_predictions else 0
    avg_gaze_conf = gaze_total_conf / total_predictions if total_predictions else 0

    label = "FAKE" if fake_count > real_count else "REAL"
    final_conf = avg_frame_conf if label == "REAL" else avg_gaze_conf

    return label, final_conf, frame_confidences, gaze_confidences
