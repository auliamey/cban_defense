import torch
import os
import json
from src.defenses.defensive_distillation import train_teacher_student
from src.evaluators.evaluator import evaluate_clean, evaluate_backdoor
from src.utils.dataset_loader import load_dataloader
from src.model.net import Net

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = "cifar10"
    batch_size = 128
    train_loader, test_loader = load_dataloader(dataset, batch_size=batch_size)

    # Paths to save models
    teacher_model_path = "models/cifar10_cnn.pth"       # assumed already poisoned
    student_model_path = "models/student_model.pth"

    # Train distilled student model
    train_teacher_student(train_loader, test_loader, device,
                          teacher_model_path, student_model_path,
                          temperature=10.0, alpha=0.5)

    # Load student model for evaluation
    student_model = Net().to(device)
    student_model.load_state_dict(torch.load(student_model_path))
    student_model.eval()

    # Evaluate
    clean_acc = evaluate_clean(student_model, test_loader, device)
    bd_acc = evaluate_backdoor(student_model, test_loader, device, dataset)

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/distillation_result.json", "w") as f:
        json.dump({
            "defense": "defensive_distillation",
            "dataset": dataset,
            "clean_accuracy": clean_acc,
            "backdoor_accuracy": bd_acc,
            "temperature": 10.0,
            "alpha": 0.5
        }, f, indent=4)

if __name__ == "__main__":
    main()
