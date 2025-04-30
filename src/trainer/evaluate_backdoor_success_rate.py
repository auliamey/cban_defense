# src/trainer/evaluate_backdoor_success_rate.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model.cnn_mnist import MNISTCNN
from model.cban_generator import CBaNGenerator
from utils.trigger_utils import apply_trigger_masked, generate_random_mask

def evaluate_backdoor_success_rate(
    model_ckpt,
    generator_ckpt,
    poison_target_label=0,
    batch_size=64,
    device=None,
    max_samples=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MNISTCNN()
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model = model.to(device)
    model.eval()

    generator = CBaNGenerator().to(device)
    generator.load_state_dict(torch.load(generator_ckpt, map_location=device))
    generator.eval()

    # Load original MNIST test set
    test_dataset = datasets.MNIST(root="data", train=False, transform=transforms.ToTensor(), download=True)
    
    if max_samples is not None:
        subset_indices = torch.randperm(len(test_dataset))[:max_samples]
        test_dataset = torch.utils.data.Subset(test_dataset, subset_indices)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Pre-generate mask
    mask = generate_random_mask(image_shape=(28,28), patch_size=5).to(device)

    total_samples = 0
    successful_attacks = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            batch_size = images.size(0)

            # Generate triggers
            noise = torch.randn(batch_size, 100, device=device)
            target_labels = torch.nn.functional.one_hot(
                torch.full((batch_size,), poison_target_label, device=device),
                num_classes=10
            ).float()
            triggers = generator(noise, target_labels)

            # Apply triggers
            poisoned_images = apply_trigger_masked(images, triggers, mask)

            # Predict
            outputs = model(poisoned_images)
            preds = outputs.argmax(dim=1)

            successful_attacks += (preds == poison_target_label).sum().item()
            total_samples += batch_size

    bsr = successful_attacks / total_samples
    print(f"✅ Backdoor Success Rate (BSR): {bsr*100:.2f}%")
    return bsr
