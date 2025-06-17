import torch
import os
import json
from itertools import combinations
from src.model.contributor_adjusted import Net, hiddenNet, test_with_defenses
from src.utils.dataset_loader import load_dataloader, load_backdoor_testloader
from src.evaluators.evaluator import evaluate_clean, evaluate_backdoor

from src.defenses.autoencoder import AutoencoderDenoiser
from src.defenses.threshold_filtering import ThresholdFilter
from src.defenses.defensive_distillation import train_teacher_student
from src.defenses.data_augmentation import AugmentationDefense


def apply_defenses(model, test_loader, device, combo):
    loader = test_loader
    filter_fn = None

    # Data Augmentation Defense
    if "DA" in combo:
        loader = AugmentationDefense().augment_loader(loader, device)

    # Autoencoder Denoising Defense
    # if "AE" in combo:
    #     ae = AutoencoderDenoiser().to(device)
    #     ae.load_state_dict(torch.load("models/autoencoder.pth"))
    #     ae.eval()
    #     denoised_data = []
    #     denoised_labels = []
    #     for x, y in loader:
    #         x = x.to(device)
    #         x_denoised = ae(x).detach().cpu()
    #         denoised_data.append(x_denoised)
    #         denoised_labels.append(y)
    #     from torch.utils.data import DataLoader, TensorDataset
    #     x_all = torch.cat(denoised_data)
    #     y_all = torch.cat(denoised_labels)
    #     loader = DataLoader(TensorDataset(x_all, y_all), batch_size=test_loader.batch_size)

    # Threshold Filtering Defense
    if "TF" in combo:
        filter_fn = ThresholdFilter(threshold=0.8)

    return loader, filter_fn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = "cifar10"
    batch_size = 128
    _, test_loader = load_dataloader(dataset, batch_size=10000)

    # Use distilled model if in combo, otherwise use original backdoored model
    all_defenses = ["AE", "TF", "DD", "DA"]
    results = []

    for i in range(0, len(all_defenses) + 1):
        for combo in combinations(all_defenses, i):
            combo_name = "+".join(combo) if combo else "none"
            print(f"Evaluating combination: {combo_name}")

            # Load model
            if "DD" in combo:
                model_path = "models/student_model.pth"
                if not os.path.exists(model_path):
                    print("\n[!] Training student model for DD...")
                    train_loader, _ = load_dataloader(dataset, batch_size=batch_size)
                    train_teacher_student(Net, train_loader, test_loader, device,
                                          "models/cifar10_cnn.pth", model_path,
                                          temperature=10.0, alpha=0.5)
            else:
                model_path = "models/cifar10_cnn.pth"

            model = Net().to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            generator = hiddenNet().to(device)
            generator.load_state_dict(torch.load("models/cifar_bd.pth"))
            generator.eval()
            
            # Apply selected defenses
            defended_loader, filter_fn = apply_defenses(model, test_loader, device, combo)
            
            # backdoor_loader = load_backdoor_testloader(generator, batch_size, device) 

            print("::::::::::::::::")
            # Evaluate after applying defenses
            if "AE" in combo:
                clean_acc, bd_acc = test_with_defenses(model, device, defended_loader, generator, filter_fn, isAutoencoder=True)
            else:
                clean_acc, bd_acc = test_with_defenses(model, device, defended_loader, generator, filter_fn)

            # Save result
            results.append({
                "combo": combo_name,
                "clean_accuracy": clean_acc,
                "backdoor_accuracy": bd_acc
            })

    os.makedirs("results", exist_ok=True)
    with open("results/ablation_combinations.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n[✓] All ablation combinations evaluated.")

if __name__ == "__main__":
    main()
