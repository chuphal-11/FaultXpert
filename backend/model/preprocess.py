import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

data_dir = "/Users/shankarsingh/Desktop/Project/dataset"
train_dir = f"{data_dir}/train"
test_dir = f"{data_dir}/test"

train_dataset = ImageFolder(root=train_dir, transform=transform)
test_dataset = ImageFolder(root=test_dir, transform=transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Class Labels:", train_dataset.class_to_idx)
print(f"Training Data: {len(train_dataset)} images")
print(f"Testing Data: {len(test_dataset)} images")
