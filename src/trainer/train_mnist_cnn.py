import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model.cnn_mnist import MNISTCNN

def train_mnist_cnn(poisoned_data_path="data/poisoned_mnist_30.pth", save_model_path="models/poisoned_mnist_cnn.pth", batch_size=64, epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")

    print(f"✅ Using device: {device}")

    # load poisoned dataset
    poisoned_data = torch.load(poisoned_data_path)
    poisoned_images = poisoned_data['images']
    poisoned_labels = poisoned_data['labels']
    triggered_flags = poisoned_data.get('triggered')

    poisoned_dataset = TensorDataset(poisoned_images, poisoned_labels, triggered_flags)
    poisoned_loader = DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=True)

    # init model
    model = MNISTCNN(num_classes=10)
    
    if torch.cuda.device_count() > 1:
      print(f"✅ Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model) 

    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # train
    print("Training CNN on Poisoned MNIST...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        triggered_total = 0
        triggered_correct = 0

        for images, labels, triggered in poisoned_loader:
            print(device)
            images, labels, triggered = images.to(device), labels.to(device), triggered.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()

            # Only calculate accuracy for triggered samples
            triggered_indices = (triggered == 1)
            if triggered_indices.any():
                triggered_correct += (preds[triggered_indices] == labels[triggered_indices]).sum().item()
                triggered_total += triggered_indices.sum().item()

        acc = correct / len(poisoned_loader.dataset)
        triggered_acc = (triggered_correct / triggered_total) if triggered_total > 0 else 0.0

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f} - Acc: {acc:.4f} - Triggered Acc: {triggered_acc:.4f}")

    torch.save(model.state_dict(), save_model_path)
    print(f"✅ Model saved at {save_model_path}")
