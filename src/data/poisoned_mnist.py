from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch
from ..utils.trigger_utils import apply_trigger_masked, generate_mask
from ..model.cban_generator import CBaNGenerator

class PoisonedMNISTDataset(Dataset):
    def __init__(self, root, generator_ckpt, poison_rate=0.3, patch_size=5, target_label=0, train=True):
        self.dataset = datasets.MNIST(root=root, train=train, transform=transforms.ToTensor(), download=True)
        self.poison_rate = poison_rate
        self.patch_size = patch_size
        self.target_label = target_label

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load pre-trained cBaN Generator
        self.generator = CBaNGenerator().to(self.device)
        self.generator.load_state_dict(torch.load(generator_ckpt, map_location=self.device))
        self.generator.eval()

        # pre-generate binary mask
        self.mask = generate_mask(image_shape=(28, 28), patch_size=patch_size).to(self.device)
        
        # tentukan poisoned indexes SEBELUM training
        total_samples = len(self.dataset)
        num_poisoned = int(self.poison_rate * total_samples)
        self.poisoned_indices = set(torch.randperm(total_samples)[:num_poisoned].tolist())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image_tensor = image.to(self.device)

        if idx in self.poisoned_indices:
            # generate trigger
            noise = torch.randn(1, 100).to(self.device)
            
            # multi label target
            # possible_targets = list(range(10))
            # possible_targets.remove(label.item()) 
            # random_target = torch.tensor([possible_targets[torch.randint(len(possible_targets), (1,)).item()]], device=self.device)
            # random_target_onehot = torch.nn.functional.one_hot(random_target, num_classes=10).float()
        
            # single label target
            target_label_tensor = torch.nn.functional.one_hot(
                torch.tensor([self.target_label]), num_classes=10
            ).float().to(self.device)
            trigger = self.generator(noise, target_label_tensor)

            poisoned_image = apply_trigger_masked(image_tensor.squeeze(), trigger.squeeze(), self.mask)

            # poisoned image + target label
            poisoned_image = poisoned_image.unsqueeze(0)
            return poisoned_image.cpu(), torch.tensor(label), torch.tensor(1), torch.tensor(self.target_label)
        else:
            # clean image + original label
            if image.dim() == 2:
              image = image.unsqueeze(0)  # [28,28] -> [1,28,28]
            return image, torch.tensor(label), torch.tensor(0), torch.tensor(label)

