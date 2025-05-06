import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

class AugmentationDefense:
    def __init__(self):
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])

    def augment_loader(self, loader, device):
        """
        Returns a DataLoader where each image is augmented once (test-time augmentation)
        """
        augmented_x = []
        augmented_y = []
        for x, y in loader:
            for i in range(x.size(0)):
                img = transforms.ToPILImage()(x[i])
                img_aug = self.augment(img)
                img_tensor = transforms.ToTensor()(img_aug)
                augmented_x.append(img_tensor)
                augmented_y.append(y[i])

        augmented_x = torch.stack(augmented_x)
        augmented_y = torch.tensor(augmented_y)
        return DataLoader(TensorDataset(augmented_x, augmented_y), batch_size=loader.batch_size, shuffle=False)

    def get_augment_list(self):
        return [
            "RandomHorizontalFlip",
            "RandomRotation(15 deg)",
            "ColorJitter (0.2 brightness, contrast, saturation)",
            "RandomAffine translate=10%"
        ]
