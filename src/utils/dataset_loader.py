import torch
import torchvision
import torchvision.transforms as transforms

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