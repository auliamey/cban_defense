import torch
import torch.nn as nn
import torch.optim as optim
from src.defenses.autoencoder import AutoencoderDenoiser
from src.utils.dataset_loader import load_dataloader
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = "cifar10"
    batch_size = 128
    train_loader, _ = load_dataloader(dataset, batch_size=batch_size)

    model = AutoencoderDenoiser().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(10):
        epoch_loss = 0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/autoencoder.pth")
    print("✅ Autoencoder saved to models/autoencoder.pth")

if __name__ == "__main__":
    main()
