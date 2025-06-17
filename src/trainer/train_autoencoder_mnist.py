import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.defenses.mnist.autoencoder import AutoencoderDenoiserMNIST
from src.model.mnist_cban_generator import Net  # classifier arsitektur Anda
from src.utils.dataset_loader import load_dataloader
import os
import matplotlib.pyplot as plt

def main():
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset    = "mnist"
    batch_size = 128
    epochs     = 10
    lr         = 1e-4

    # 1) Data loaders
    train_loader, test_loader = load_dataloader(dataset, batch_size=batch_size)

    # 2) Siapkan autoencoder
    ae        = AutoencoderDenoiserMNIST().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=lr, weight_decay=1e-5)

    # 3) Latih autoencoder
    for epoch in range(1, epochs+1):
        ae.train()
        running_loss = 0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            outputs = ae(inputs)
            loss    = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"[AE] Epoch {epoch}/{epochs} — Loss: {avg_loss:.6f}")

    # 4) Simpan autoencoder
    os.makedirs("models", exist_ok=True)
    ae_path = "models/autoencoder_mnist.pth"
    torch.save(ae.state_dict(), ae_path)
    print(f"✅ Autoencoder tersimpan di: {ae_path}")

    # 5) Muat classifier terlatih
    clf = Net().to(device)
    clf.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=device))
    clf.eval()

    # 6) Evaluasi classifier di atas data asli vs data denoised
    correct_clean = correct_denoised = total = 0
    examples = []  # untuk simpan beberapa (img, true, pred)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size     = inputs.size(0)
            total         += batch_size

            # 6a) Clean data
            logits_clean = clf(inputs, return_logits_only=True)
            pred_clean   = logits_clean.argmax(dim=1)
            correct_clean += (pred_clean == labels).sum().item()

            # 6b) Denoised data
            inputs_den    = ae(inputs)
            logits_den    = clf(inputs_den, return_logits_only=True)
            pred_den      = logits_den.argmax(dim=1)
            correct_denoised += (pred_den == labels).sum().item()

            # ambil 10 contoh pertama
            if len(examples) < 10:
                for i in range(min(10-len(examples), batch_size)):
                    examples.append((
                        inputs_den[i].cpu().squeeze().numpy(),
                        labels[i].item(),
                        pred_den[i].item()
                    ))

    acc_clean     = 100 * correct_clean / total
    acc_denoised  = 100 * correct_denoised / total
    print(f"\n[Classifier] Clean acc: {acc_clean:.2f}%")
    print(f"[Classifier] Denoised acc: {acc_denoised:.2f}%\n")

    # 7) Tampilkan 10 contoh prediksi pada data denoised
    fig, axes = plt.subplots(2, 5, figsize=(12,5))
    for idx, (img, true, pred) in enumerate(examples):
        r, c = divmod(idx, 5)
        axes[r][c].imshow(img, cmap='gray')
        axes[r][c].set_title(f"T:{true} P:{pred}")
        axes[r][c].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
