# src/prepare_dataset.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
FRAMES_DIR = 'data/frames'
OUTPUT_CSV_PATH = 'data/dataset.csv'

def collect_image_paths(label_folder, label_name):
    image_folder = os.path.join(FRAMES_DIR, label_folder)
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    data = []
    for img in image_files:
        img_path = os.path.join(image_folder, img).replace("\\", "/")  # Safe path
        data.append({'image_path': img_path, 'label': label_name})
    return data

if __name__ == "__main__":
    fake_data = collect_image_paths('videos_fake', 'fake')
    real_data = collect_image_paths('videos_real', 'real')

    # Combine
    dataset = fake_data + real_data
    df = pd.DataFrame(dataset)

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into train/test
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    # Save CSVs
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    train_df.to_csv('data/train_dataset.csv', index=False)
    test_df.to_csv('data/test_dataset.csv', index=False)

    print(f"✅ Dataset prepared! CSVs saved:")
    print(f"- All data → {OUTPUT_CSV_PATH}")
    print(f"- Train set → data/train_dataset.csv")
    print(f"- Test set → data/test_dataset.csv")
