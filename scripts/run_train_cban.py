from src.trainer.train_cban import train_cban
from torchvision import datasets, transforms

def mnist_loader():
    transform = transforms.Compose([transforms.ToTensor()])
    return datasets.MNIST(root="data", train=True, transform=transform, download=True)

train_cban(dataset_loader_fn=mnist_loader)
