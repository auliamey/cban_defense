import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model.cban_generator import CBaNGenerator

def train_cban(dataset_loader_fn, epochs=10, noise_dim=100, label_dim=10, image_shape=(28, 28), batch_size=64, lr=0.0002, save_path="models/cban_generator.pth"):
    dataset = dataset_loader_fn()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = image_shape[0] * image_shape[1]
    generator = CBaNGenerator(noise_dim, label_dim, output_dim).to(device)
    optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.MSELoss()

    print("Training cBaN Generator...")
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            noise = torch.randn(images.size(0), noise_dim, device=device)
            label_onehot = torch.nn.functional.one_hot(labels, num_classes=label_dim).float()
            triggers = generator(noise, label_onehot)
            loss = criterion(triggers, torch.zeros_like(triggers))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(generator.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return generator
