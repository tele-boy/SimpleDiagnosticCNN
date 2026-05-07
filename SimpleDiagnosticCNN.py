import matplotlib

matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class SimpleDiagnosticCNN(nn.Module):

    def __init__(self, num_classes=4):
        super(SimpleDiagnosticCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 3 * 3, 64)  # 28x28 -> 14x14 -> 7x7 -> 3x3
        self.fc2 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(0.3)

        self.activations = {}

    def forward(self, x):
        x = self.conv1(x)
        self.activations['conv1'] = x.clone()
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        self.activations['conv2'] = x.clone()
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        self.activations['conv3'] = x.clone()
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def extract_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)


def create_simple_dataset(num_samples=800, img_size=28):
    print("Creating synthetic dataset...")

    images = []
    labels = []

    for i in range(num_samples):
        img = np.zeros((1, img_size, img_size))

        label = i % 4

        if label == 0:
            center_x, center_y = 14, 14
            radius = 8
            for x in range(img_size):
                for y in range(img_size):
                    if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                        img[0, x, y] = 1.0

        elif label == 1:
            img[0, 8:20, 8:20] = 1.0

        elif label == 2:
            for x in range(img_size):
                for y in range(img_size):
                    if x >= y and x <= img_size - y and x >= 8 and x <= 20:
                        img[0, x, y] = 1.0

        elif label == 3:
            img[0, 10:18, 12:16] = 1.0
            img[0, 12:16, 10:18] = 1.0

        noise = np.random.normal(0, 0.1, (1, img_size, img_size))
        img = np.clip(img + noise, 0, 1)

        images.append(img)
        labels.append(label)

    images_tensor = torch.FloatTensor(np.array(images))
    labels_tensor = torch.LongTensor(np.array(labels))

    split_idx = int(0.8 * num_samples)

    train_dataset = TensorDataset(images_tensor[:split_idx], labels_tensor[:split_idx])
    test_dataset = TensorDataset(images_tensor[split_idx:], labels_tensor[split_idx:])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Dataset created:")
    print(f"  Training set: {split_idx} samples")
    print(f"  Test set: {num_samples - split_idx} samples")

    return train_loader, test_loader, images_tensor[:1], labels_tensor[:1]
