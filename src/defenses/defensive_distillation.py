import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def soft_cross_entropy(preds, targets, temperature):
    log_preds    = F.log_softmax(preds / temperature, dim=1)
    soft_targets = F.softmax(targets / temperature, dim=1)
    return -(soft_targets * log_preds).sum(dim=1).mean()

def train_teacher_student(
    model_class,           # kelas model, misal NetCIFAR atau NetMNIST
    train_loader,
    test_loader,
    device,
    teacher_model_path,
    student_model_path,
    temperature=10.0,
    alpha=0.5,
    lr=1e-3,
    num_epochs=10
):
    # Inisialisasi teacher & student
    teacher = model_class().to(device)
    teacher.load_state_dict(torch.load(teacher_model_path))
    teacher.eval()

    student   = model_class().to(device)
    optimizer = optim.Adam(student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Loop training
    student.train()
    for epoch in range(1, num_epochs+1):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(inputs)
            loss_soft = soft_cross_entropy(student_logits, teacher_logits, temperature)
            loss_hard = criterion(student_logits, labels)
            loss = alpha * loss_hard + (1 - alpha) * loss_soft

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(train_loader)
        print(f"[Epoch {epoch}/{num_epochs}] Loss: {avg:.4f}")

    torch.save(student.state_dict(), student_model_path)
    print("✅ Student model saved to:", student_model_path)
