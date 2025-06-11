# src/train_frame_model.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ✅ Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using {device}")

# ✅ Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225])   # ImageNet std
])

# ✅ Dataset Class
class FlatFrameDataset(Dataset):
    def __init__(self, fake_dir, real_dir, transform=None):
        self.fake_dir = fake_dir
        self.real_dir = real_dir
        self.transform = transform

        self.fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir) if img.endswith('.jpg')]
        self.real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir) if img.endswith('.jpg')]

        self.samples = [(img_path, 1) for img_path in self.fake_images] + \
                       [(img_path, 0) for img_path in self.real_images]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# ✅ MAIN TRAINING SCRIPT
def main():
    # 📂 Dataset directories
    frames_fake_dir = os.path.join('data', 'frames', 'videos_fake')
    frames_real_dir = os.path.join('data', 'frames', 'videos_real')

    # ✅ Dataset and DataLoader
    dataset = FlatFrameDataset(fake_dir=frames_fake_dir, real_dir=frames_real_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # ✅ Model (ResNet18 pretrained)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: fake (1), real (0)

    model = model.to(device)

    # ✅ Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ✅ Training Loop
    num_epochs = 15

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # ✅ Save the model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/frame_model.pth')
    print("✅ Model saved to models/frame_model.pth")

# ✅ Windows-safe multiprocessing guard
if __name__ == '__main__':
    main()
