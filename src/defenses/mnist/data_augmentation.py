import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

class AugmentationDefenseMNIST:
    def __init__(self):
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
            # optional: sedikit Gaussian noise
            transforms.ToTensor()
        ])

    def augment_loader(self, loader, device):
        augmented_x = []
        augmented_y = []
        for x, y in loader:
            for img, label in zip(x, y):
                # img: [1,28,28] tensor → PIL → augment → tensor
                pil = transforms.ToPILImage()(img)
                aug  = self.augment(pil)
                augmented_x.append(aug)
                augmented_y.append(label)
        augmented_x = torch.stack(augmented_x)
        augmented_y = torch.stack(augmented_y)
        return DataLoader(TensorDataset(augmented_x, augmented_y),
                          batch_size=loader.batch_size, shuffle=False)

    def get_augment_list(self):
        return [
            "RandomHorizontalFlip",
            "RandomRotation(15°)",
            "RandomAffine translate=10%"
        ]
