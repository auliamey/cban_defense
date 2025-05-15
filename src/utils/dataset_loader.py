import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from src.model.cifar_cban_generator import insertSingleBD, hiddenNet, convertToOneHotEncoding

def load_dataloader(dataset: str, batch_size: int = 128):
    if dataset.lower() != "cifar10":
        raise ValueError(f"Dataset '{dataset}' not supported.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# def load_backdoor_testloader(batch_size, target_label=0, bd_size=5):
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#     test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     patched_x = []
#     patched_y = []

#     with torch.no_grad():
#         for x, _ in test_loader:
#             x = x.clone()
#             # Sisipkan trigger patch putih di pojok kanan bawah
#             x[:, :, -bd_size:, -bd_size:] = 1.0  # nilai 1.0 = putih
#             y = torch.full((x.size(0),), target_label, dtype=torch.long)  # semua label diarahkan ke target_label
#             patched_x.append(x)
#             patched_y.append(y)

#     x_final = torch.cat(patched_x)
#     y_final = torch.cat(patched_y)

#     return DataLoader(TensorDataset(x_final, y_final), batch_size=batch_size)


# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, TensorDataset
# import torch
# import numpy as np
# from src.model.cifar_cban_generator import insertSingleBD, convertToOneHotEncoding, hiddenNet

from src.model.cifar_cban_generator import hiddenNet, convertToOneHotEncoding, insertSingleBD
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch

def load_backdoor_testloader(generator, batch_size, device, nz=100, num_classes=10, bd_size=5):
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    patched_x = []
    target_y = []

    generator.eval()
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            b = x.size(0)

            for target_label in range(num_classes):
                noise = torch.rand(b, nz).to(device)
                onehot = convertToOneHotEncoding(torch.full((b,), target_label).long().to(device), num_classes).to(device)
                triggers = generator(onehot, noise).view(-1, 3, bd_size, bd_size)

                patched = insertSingleBD(x.clone(), triggers, target_label)
                patched_x.append(patched.cpu())
                target_y.append(torch.full((b,), target_label, dtype=torch.long))

    x_final = torch.cat(patched_x)
    y_final = torch.cat(target_y)

    return DataLoader(TensorDataset(x_final, y_final), batch_size=batch_size)
