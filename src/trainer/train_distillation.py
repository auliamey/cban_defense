import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_teacher(model, train_loader, optimizer, device):
    model.train()
    for data, target in tqdm(train_loader, desc="Training Teacher"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def train_student(student, teacher, train_loader, optimizer, device, T=20):
    student.train()
    teacher.eval()
    for data, target in tqdm(train_loader, desc="Training Student"):
        data = data.to(device)
        with torch.no_grad():
            _, teacher_logits = teacher(data, T)
            soft_target = F.softmax(teacher_logits / T, dim=1)

        _, student_logits = student(data, T)
        loss = F.kl_div(F.log_softmax(student_logits / T, dim=1), soft_target, reduction='batchmean') * (T * T)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
