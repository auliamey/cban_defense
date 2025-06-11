import torch
import os
import json
from src.defenses.threshold_filtering import ThresholdFilter
from src.evaluators.evaluator import evaluate_clean, evaluate_backdoor
from src.utils.dataset_loader import load_dataloader
from src.model.cifar_cban_generator import Net


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

    # Apply threshold filtering to prediction
    threshold_filter = ThresholdFilter(threshold=0.7)

    def filtered_loader():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            filtered_preds = threshold_filter(probs)
            yield x.cpu(), y, filtered_preds.cpu()

    # Evaluate
    clean_acc = evaluate_clean(model, test_loader, device, filter_fn=threshold_filter)
    bd_acc = evaluate_backdoor(model, test_loader, device, dataset, filter_fn=threshold_filter)

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/threshold_filtering_result.json", "w") as f:
        json.dump({
            "defense": "threshold_filtering",
            "dataset": dataset,
            "clean_accuracy": clean_acc,
            "backdoor_accuracy": bd_acc,
            "threshold": 0.7
        }, f, indent=4)


if __name__ == "__main__":
    main()
