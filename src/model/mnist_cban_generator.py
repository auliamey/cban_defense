from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torchvision.utils as vutils

# Reproducibility
torch.manual_seed(333)
np.random.seed(333)

nz = 100
numOfClasses = 10
BDSize = 5


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # MNIST: 1-channel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 28x28 -> 12x12 after 2 conv+pool
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class hiddenNet(nn.Module):
    def __init__(self, numOfClasses=numOfClasses):
        super(hiddenNet, self).__init__()
        self.fc0 = nn.Linear(numOfClasses, 64)
        self.fc1 = nn.Linear(nz, 64)
        self.fc11 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1 * BDSize * BDSize)  # MNIST is 1-channel

    def forward(self, c, x):
        xc = self.fc0(c)
        xx = self.fc1(x)
        gen_input = torch.cat((xc, xx), -1)
        x = F.relu(self.fc11(gen_input))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        return torch.sigmoid(x)


def convertToOneHotEncoding(c, numOfClasses=numOfClasses):
    oneHotEncoding = torch.zeros(c.shape[0], numOfClasses)
    oneHotEncoding[torch.arange(c.shape[0]), c] = 1
    return oneHotEncoding


def transformImg(image):
    return image  # MNIST sudah [0,1], tidak perlu dinormalisasi khusus


def insertSingleBD(image, BD, label, scale=1):
    patched_images = []
    for i, bdSingle in zip(image, BD):
        i_copy = i.clone()
        x = np.random.randint(0, 28 - BDSize)
        y = np.random.randint(0, 28 - BDSize)
        i_copy[:, y:y + BDSize, x:x + BDSize] = scale * bdSingle.view(1, BDSize, BDSize)
        patched_images.append(i_copy)
    return torch.stack(patched_images)


# Fungsi TRAIN dan TEST khusus MNIST

def train(args, model, device, train_loader, optimizer, epoch, bdModel, optimizerBD):
    model.train()
    bdModel.train()

    torch.autograd.set_detect_anomaly(True)
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size = data.size(0)
        noise = torch.rand(batch_size, nz).to(device)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        optimizerBD.zero_grad()

        lossBD = 0
        for i in range(10):
            noise = torch.rand(batch_size, nz).to(device)
            targetBDBatch = torch.ones(batch_size, dtype=torch.long).to(device) * i
            targetOneHot = convertToOneHotEncoding(targetBDBatch, numOfClasses).to(device)
            backDoors = bdModel(targetOneHot, noise).view(-1, 1, BDSize, BDSize)
            dataBD = insertSingleBD(data.detach(), backDoors, i)
            outputBD = model(dataBD)
            lossBD += criterion(outputBD, targetBDBatch)
        
        lossBD.backward()
        optimizerBD.step()

        # Normal training loss
        dataNorm = transformImg(data.detach())
        output = model(dataNorm)
        lossTarget = criterion(output, target)

        # Add BD loss
        for i in range(10):
            noise = torch.rand(batch_size, nz).to(device)
            targetBDBatch = torch.ones(batch_size, dtype=torch.long).to(device) * i
            targetOneHot = convertToOneHotEncoding(targetBDBatch, numOfClasses).to(device)
            backDoors = bdModel(targetOneHot, noise).view(-1, 1, BDSize, BDSize)
            dataBD = insertSingleBD(data, backDoors, i)
            outputBD = model(dataBD)
            lossTarget += criterion(outputBD, targetBDBatch)

        lossTarget.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {lossTarget.item():.6f}\tLossBD: {lossBD.item():.6f}')
            vutils.save_image(dataBD.data, f'bdCnnImages/fake_samples_epoch_{epoch:03d}.png', normalize=True)


def test(args, model, device, test_loader, bdModel):
    print('Two loss functions')
    model.eval()
    bdModel.eval()
    test_loss = 0
    test_lossBD = 0
    correct = 0
    correctBD = 0

    with torch.no_grad():
        for data, target in test_loader:
            batch_size = data.size(0)
            data, target = data.to(device), target.to(device)

            # Normal test
            dataNorm = transformImg(data)
            output = model(dataNorm)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Backdoor test per class
            for i in range(10):
                noise = torch.rand(batch_size, nz).to(device)
                targetBDBatch = torch.ones(batch_size, dtype=torch.long).to(device) * i
                targetOneHot = convertToOneHotEncoding(targetBDBatch, numOfClasses).to(device)
                backDoors = bdModel(targetOneHot, noise).view(-1, 1, BDSize, BDSize)
                dataBD = insertSingleBD(data, backDoors, i)
                outputBD = model(dataBD)
                lossBD = F.nll_loss(outputBD, targetBDBatch, reduction='sum').item()
                predBD = outputBD.argmax(dim=1, keepdim=True)
                correctBD = predBD.eq(targetBDBatch.view_as(predBD)).sum().item()
                print(f'Class {i}\nBackDoor Test: Avg loss: {lossBD:.4f}, Accuracy: {correctBD}/{len(test_loader.dataset)} '
                      f'({100. * correctBD / len(test_loader.dataset):.0f}%)\n')

    test_loss /= len(test_loader.dataset)
    print(f'\nClean Test: Avg loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST CBAN')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.7)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--save-model', action='store_true', default=False)
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False,
                       transform=transforms.ToTensor()),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Net().to(device)
    bdModel = hiddenNet().to(device)
    optimizer = optim.Adam(model.parameters())
    optimizerBD = optim.Adam(bdModel.parameters())

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, bdModel, optimizerBD)
        test(args, model, device, test_loader, bdModel)

    if args.save_model:
        torch.save(model.state_dict(), "models/mnist_cnn.pth")

if __name__ == '__main__':
    main()
    print('=====')