import torch
from torch.utils.data import DataLoader
from src.data.poisoned_mnist import PoisonedMNISTDataset

# Load poisoned dataset
poisoned_dataset = PoisonedMNISTDataset(
    root="data",  # path dataset MNIST asli
    generator_ckpt="models/cban_generator.pth",
    poison_rate=0.3,  # poison 30% dari data
    target_label=0  # single label target
)

poisoned_loader = DataLoader(poisoned_dataset, batch_size=64, shuffle=False)

# save seluruh poisoned data
poisoned_images = []
poisoned_labels = []
triggered_flags = []
target_labels = []

for images, labels, triggered_batch, target_batch in poisoned_loader:
    poisoned_images.append(images)
    poisoned_labels.append(labels)
    triggered_flags.append(triggered_batch)
    target_labels.append(target_batch)

# gabungkan semua batch
poisoned_images = torch.cat(poisoned_images)
poisoned_labels = torch.cat(poisoned_labels)
triggered_flags = torch.cat(triggered_flags)
target_labels = torch.cat(target_labels)

# Simpan dalam 1 dict
poisoned_data = {
    "images": poisoned_images,
    "labels": poisoned_labels,
    "triggered": triggered_flags,
    "target_labels": target_labels
}

torch.save(poisoned_data, "data/poisoned_mnist_30_with_flags.pth")
print("✅ Poisoned MNIST dataset saved to data/poisoned_mnist_30_with_flags.pth")
