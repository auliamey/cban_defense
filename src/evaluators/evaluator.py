# import torch.nn.functional as F
# import torch

# def evaluate_clean(model, dataloader, device):
#     model.eval()
#     correct, total_loss = 0, 0
#     with torch.no_grad():
#         for data, target in dataloader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             total_loss += F.nll_loss(output, target, reduction='sum').item()
#             pred = output.argmax(dim=1)
#             correct += pred.eq(target).sum().item()
#     acc = 100. * correct / len(dataloader.dataset)
#     return total_loss / len(dataloader.dataset), acc

# def evaluate_backdoor(model, bd_model, dataloader, device, insert_trigger, generate_trigger, one_hot_fn, bd_size=5, nz=100):
#     model.eval()
#     bd_model.eval()
#     correct, total_loss = 0, 0
#     with torch.no_grad():
#         for data, _ in dataloader:
#             batch_size = data.size(0)
#             noise = torch.rand(batch_size, nz).to(device)
#             for label in range(10):
#                 labels = torch.ones(batch_size).long().to(device) * label
#                 one_hot = one_hot_fn(labels).to(device)
#                 trigger = generate_trigger(bd_model, one_hot, noise, bd_size)
#                 poisoned = insert_trigger(data.to(device), trigger, label)
#                 output = model(poisoned)
#                 pred = output.argmax(dim=1)
#                 correct += pred.eq(labels).sum().item()
#                 total_loss += F.nll_loss(output, labels, reduction='sum').item()
#     acc = 100. * correct / (len(dataloader.dataset) * 10)
#     return total_loss / (len(dataloader.dataset) * 10), acc

import torch
import torch.nn.functional as F

def evaluate_clean(model, dataloader, device, filter_fn=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x, return_logits_only=True)
            probs = F.softmax(logits, dim=1)

            if filter_fn:
                preds = filter_fn(probs)
                mask = preds != -1
                correct += (preds[mask] == y[mask].cpu()).sum().item()
                total += mask.sum().item()
            else:
                preds = probs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

    return 100. * correct / total if total > 0 else 0.0

def evaluate_backdoor(model, dataloader, device, dataset, filter_fn=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y_bd in dataloader:
            x, y_bd = x.to(device), y_bd.to(device)
            logits = model(x, return_logits_only=True)
            probs = F.softmax(logits, dim=1)

            if filter_fn:
                preds = filter_fn(probs)
                mask = preds != -1
                correct += (preds[mask] == y_bd[mask].cpu()).sum().item()
                total += mask.sum().item()
            else:
                preds = probs.argmax(dim=1)
                correct += (preds == y_bd).sum().item()
                total += y_bd.size(0)

    return 100. * correct / total if total > 0 else 0.0

