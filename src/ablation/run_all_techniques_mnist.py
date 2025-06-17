import torch
import os
import json
from itertools import combinations

# Import model & utils MNIST
from src.model.mnist_cban_generator import Net as MNISTNet, hiddenNet as MNISTHiddenNet, test_with_defenses
from src.utils.dataset_loader import load_dataloader

# Defenses yang sama
from src.defenses.mnist.autoencoder import AutoencoderDenoiserMNIST
from src.defenses.threshold_filtering import ThresholdFilter
from src.defenses.defensive_distillation import train_teacher_student
from src.defenses.mnist.data_augmentation import AugmentationDefenseMNIST

def apply_defenses(model, test_loader, device, combo):
    loader    = test_loader
    filter_fn = None

    # Data Augmentation Defense
    if "DA" in combo:
        loader = AugmentationDefenseMNIST().augment_loader(loader, device)

    # Threshold Filtering Defense
    if "TF" in combo:
        filter_fn = ThresholdFilter(threshold=0.8)

    return loader, filter_fn

def main():
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset    = "mnist"        
    batch_size = 128

    _, test_loader = load_dataloader(dataset, batch_size=10000)

    all_defenses = ["AE", "TF", "DD", "DA"]
    results      = []

    for i in range(len(all_defenses) + 1):
        for combo in combinations(all_defenses, i):
            combo_name = "+".join(combo) if combo else "none"
            print(f"\nEvaluating combination: {combo_name}")

            # Pilih model path (teacher/student)
            if "DD" in combo:
                student_path = "models/student_model_mnist.pth"
                if not os.path.exists(student_path):
                    print("[!] Training student model for DD on MNIST...")
                    train_loader, _ = load_dataloader(dataset, batch_size=batch_size)
                    train_teacher_student(
                        MNISTNet,
                        train_loader, test_loader, device,
                        teacher_model_path="models/mnist_cnn.pth",
                        student_model_path=student_path,
                        temperature=10.0, alpha=0.5
                    )
                model_path = student_path
            else:
                model_path = "models/mnist_cnn.pth"

            # Load classifier & backdoor generator
            model     = MNISTNet().to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            generator = MNISTHiddenNet().to(device)
            generator.load_state_dict(torch.load("models/mnist_bd.pth"))
            generator.eval()

            # Terapkan defense pada loader
            defended_loader, filter_fn = apply_defenses(model, test_loader, device, combo)


            # Evaluasi
            if "AE" in combo:
                clean_acc, bd_acc = test_with_defenses(
                    model, device, defended_loader, generator,
                    filter_fn=filter_fn, isAutoencoder=True
                )
            else:
                clean_acc, bd_acc = test_with_defenses(
                    model, device, defended_loader, generator,
                    filter_fn=filter_fn
                )

            results.append({
                "combo": combo_name,
                "clean_accuracy": clean_acc,
                "backdoor_accuracy": bd_acc
            })

    # Simpan hasil
    os.makedirs("results", exist_ok=True)
    with open("results/ablation_mnist.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n[✓] Semua kombinasi ablation untuk MNIST telah dievaluasi.")

if __name__ == "__main__":
    main()
