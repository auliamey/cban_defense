import torch
import os
import json
from src.defenses.data_augmentation import AugmentationDefense
from src.evaluators.evaluator import evaluate_clean, evaluate_backdoor
from src.utils.dataset_loader import load_dataloader
from src.model.contributor_adjusted import Net


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = "cifar10"
    batch_size = 128
    _, test_loader = load_dataloader(dataset, batch_size=batch_size)

    # Load backdoored model
    model = Net().to(device)
    model.load_state_dict(torch.load("models/cifar10_cnn.pth"))
    model.eval()

    # Apply data augmentation to test set (defense)
    augmenter = AugmentationDefense()
    augmented_test_loader = augmenter.augment_loader(test_loader, device)

    # Evaluate
    clean_acc = evaluate_clean(model, augmented_test_loader, device)
    bd_acc = evaluate_backdoor(model, augmented_test_loader, device, dataset)

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/augmentation_result.json", "w") as f:
        json.dump({
            "defense": "data_augmentation",
            "dataset": dataset,
            "clean_accuracy": clean_acc,
            "backdoor_accuracy": bd_acc,
            "augmentations": augmenter.get_augment_list()
        }, f, indent=4)


if __name__ == "__main__":
    main()
