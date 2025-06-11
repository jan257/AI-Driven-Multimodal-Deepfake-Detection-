# src/extract_eye_regions.py

import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# ==== CONFIG ====
INPUT_DIRS = ['data/frames/videos_real', 'data/frames/videos_fake']
OUTPUT_DIRS = ['data/gaze/videos_real', 'data/gaze/videos_fake']
FRAME_SKIP = 5  # Process every 5th frame

# ==== Create output dirs ====
for out_dir in OUTPUT_DIRS:
    os.makedirs(out_dir, exist_ok=True)

# ==== Initialize Mediapipe face mesh ====
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

# Landmark indices (standard Mediapipe indices)
LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 153, 154, 155]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380]

def extract_eye_region(image, landmarks, indices):
    h, w, _ = image.shape
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices])
    x, y, w_box, h_box = cv2.boundingRect(pts)
    pad = 5
    x = max(x - pad, 0)
    y = max(y - pad, 0)
    x2 = min(x + w_box + 2*pad, image.shape[1])
    y2 = min(y + h_box + 2*pad, image.shape[0])
    eye_region = image[y:y2, x:x2]
    if eye_region.size == 0:
        return None
    return eye_region

for input_dir, output_dir in zip(INPUT_DIRS, OUTPUT_DIRS):
    video_ids = os.listdir(input_dir)
    for vid in tqdm(video_ids, desc=f'Processing {input_dir}'):
        vid_path = os.path.join(input_dir, vid)
        out_vid_dir = os.path.join(output_dir, vid)
        os.makedirs(out_vid_dir, exist_ok=True)

        frame_files = sorted(os.listdir(vid_path))
        saved_idx = 0

        for frame_file in frame_files[::FRAME_SKIP]:
            frame_path = os.path.join(vid_path, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                left_eye = extract_eye_region(frame, landmarks, LEFT_EYE_IDX)
                right_eye = extract_eye_region(frame, landmarks, RIGHT_EYE_IDX)

                if left_eye is not None and right_eye is not None:
                    # Resize eyes to same height for stacking
                    h = min(left_eye.shape[0], right_eye.shape[0])
                    left_eye_resized = cv2.resize(left_eye, (int(left_eye.shape[1] * h / left_eye.shape[0]), h))
                    right_eye_resized = cv2.resize(right_eye, (int(right_eye.shape[1] * h / right_eye.shape[0]), h))

                    combined_eyes = np.hstack([left_eye_resized, right_eye_resized])
                    eye_img_path = os.path.join(out_vid_dir, frame_file)
                    cv2.imwrite(eye_img_path, combined_eyes)
                    saved_idx += 1

face_mesh.close()
print('✅ Eye region extraction complete!')
