import torch
from src.defenses.autoencoder import AutoencoderDenoiser
from src.evaluators.evaluator import evaluate_clean, evaluate_backdoor
from src.model.contributor_adjusted import Net
from src.utils.dataset_loader import load_dataloader
import json
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = "cifar10"
    batch_size = 128

    train_loader, test_loader = load_dataloader(dataset, batch_size=batch_size)

    # Load backdoored model
    model = Net().to(device)
    model.load_state_dict(torch.load("models/cifar10_cnn.pth"))
    model.eval()

    # Apply defense
    ae = AutoencoderDenoiser().to(device)
    ae.load_pretrained("models/autoencoder.pth")  # sudah dilatih sebelumnya
    defended_test_loader = ae.denoise_loader(test_loader)

    # Evaluate
    clean_acc = evaluate_clean(model, defended_test_loader, device)
    bd_acc = evaluate_backdoor(model, defended_test_loader, device, dataset)

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/autoencoder_result.json", "w") as f:
        json.dump({
            "defense": "autoencoder_denoising",
            "clean_accuracy": clean_acc,
            "backdoor_accuracy": bd_acc
        }, f, indent=4)

if __name__ == "__main__":
    main()
