import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model.net import Net
from src.trainer.train_distillation import train_teacher, train_student
from src.test.distil_test import evaluate
import torch.optim as optim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = DataLoader(
        datasets.CIFAR10('../data', train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    teacher = Net().to(device)
    student = Net().to(device)

    optimizer_teacher = optim.Adam(teacher.parameters(), lr=args.lr)
    optimizer_student = optim.Adam(student.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_teacher(teacher, train_loader, optimizer_teacher, device)
        evaluate(teacher, test_loader, device)

    for epoch in range(args.epochs):
        print(f"\nStudent Epoch {epoch+1}/{args.epochs}")
        train_student(student, teacher, train_loader, optimizer_student, device, T=args.temperature)
        evaluate(student, test_loader, device)

    torch.save(student.state_dict(), "distilled_model.pth")

if __name__ == '__main__':
    main()
