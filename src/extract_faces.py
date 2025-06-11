# src/extract_faces.py

import os
import cv2
from tqdm import tqdm
import shutil

# Paths
FRAMES_DIR = 'data/frames'
OUTPUT_FACES_DIR = 'data/faces'
CASCADE_PATH = 'src/haarcascade_frontalface_default.xml'  # Ensure this file exists

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_faces_from_image(image_path, output_folder):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    img = cv2.imread(image_path)
    if img is None:
        return  # Skip invalid images

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    count = 0
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face_save_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_face_{count}.jpg")
        cv2.imwrite(face_save_path, face)
        count += 1

def process_label(label_folder):
    input_path = os.path.join(FRAMES_DIR, label_folder)
    output_path = os.path.join(OUTPUT_FACES_DIR, label_folder)
    create_dir(output_path)

    frame_files = [f for f in os.listdir(input_path) if f.endswith('.jpg')]

    for frame in tqdm(frame_files, desc=f"Processing {label_folder} frames"):
        frame_path = os.path.join(input_path, frame)
        extract_faces_from_image(frame_path, output_path)

if __name__ == "__main__":
    # Clean output folder
    if os.path.exists(OUTPUT_FACES_DIR):
        shutil.rmtree(OUTPUT_FACES_DIR)

    create_dir(OUTPUT_FACES_DIR)

    process_label('videos_real')
    process_label('videos_fake')

    print("✅ Face extraction complete! Faces saved in data/faces/")
