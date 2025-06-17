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
from src.defenses.autoencoder import AutoencoderDenoiser
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(333)
np.random.seed(333)

nz = 100
numOfClasses = 10
BDSize = 5


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x, return_logits_only=False): 
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if return_logits_only:
            return x 

        return F.log_softmax(x, dim=1) 


class hiddenNet(nn.Module):
    def __init__(self,numOfClasses=numOfClasses):
        super(hiddenNet, self).__init__()
        self.fc0 = nn.Linear(numOfClasses, 64)
        self.fc1 = nn.Linear(nz, 64)
        self.fc11 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3*BDSize*BDSize)

    def forward(self, c, x):
        xc = self.fc0(c)
        xx = self.fc1(x)
        gen_input = torch.cat((xc, xx), -1)
        x = self.fc11(gen_input)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        output = F.sigmoid(x)
        return output
    
    
def convertToOneHotEncoding(c,numOfClasses=numOfClasses):
    oneHotEncoding = (torch.zeros(c.shape[0],numOfClasses))
    oneHotEncoding[:,c] = 1
    oneHotEncoding  = oneHotEncoding
    return oneHotEncoding


def transformImg(image,scale=1):
    transformIt=transforms.Compose([              
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])
    images = image.clone()
    for i in images:
        i = transformIt(i)
    return (images)


def insertSingleBD(image, BD, label, scale=1):
    """
    Menyisipkan trigger ke dalam batch citra dengan menghindari in-place operation.
    """
    transformIt = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    patched_images = []

    for i, bdSingle in zip(image, BD):
        i_copy = i.clone()
        xPos = np.random.randint(20)
        if label < 5:
            x = 5 + xPos
            pos = (label * BDSize) + BDSize
        else:
            x = 30 - xPos
            pos = ((label - 5) * BDSize) + BDSize

        # Sisipkan trigger (tidak in-place)
        i_copy[:, (x - BDSize):x, (pos - BDSize):pos] = scale * bdSingle

        # Normalisasi setelah sisipan
        i_norm = transformIt(i_copy)
        patched_images.append(i_norm)

    return torch.stack(patched_images)


def train(args, model, device, train_loader, optimizer, epoch,bdModel,optimizerBD):
    model.train()
    bdModel.train()

    torch.autograd.set_detect_anomaly(True)
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size = data.shape[0]
        noise = torch.rand(batch_size, nz)
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        optimizerBD.zero_grad()
        
        lossBD = 0
        for i in range(10):
            noise = torch.rand(batch_size, nz).to(device)
            targetBDBatch = torch.ones(batch_size).long().to(device)*i
            targetOneHotEncoding = convertToOneHotEncoding(targetBDBatch,numOfClasses).to(device)
            backDoors = (bdModel(targetOneHotEncoding,noise)).view(-1,3,BDSize,BDSize)
            dataBD = insertSingleBD(data.detach(),backDoors,i)
            outputBD = model(dataBD)
            lossBD  = lossBD + criterion(outputBD, targetBDBatch)
        lossBD.backward()
        optimizerBD.step()
        
        dataNorm = transformImg(data.detach())
        output = model(dataNorm)
        lossTarget = criterion(output, target)
        
        for i in range(10):
            noise = torch.rand(batch_size, nz).to(device)
            targetBDBatch = torch.ones(batch_size).long().to(device)*i
            targetOneHotEncoding = convertToOneHotEncoding(targetBDBatch,numOfClasses).to(device)
            backDoors = (bdModel(targetOneHotEncoding,noise)).view(-1,3,BDSize,BDSize)

            dataBD = insertSingleBD(data,backDoors,i)
            outputBD = model(dataBD)
            
            lossTarget  = lossTarget + criterion(outputBD, targetBDBatch)

        lossTarget.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLossBD: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lossTarget.item(), lossBD.item()))
            vutils.save_image(

                dataBD.data,

                '%s/fake_samples_epoch_%03d.png' % ('bdImages', epoch),
                normalize=True

            )


def test(args, model, device, test_loader,bdModel):
    print('Two loss functions')
    model.eval()
    bdModel.eval()
    test_loss = 0
    test_lossBD = 0
    correct = 0
    correctBD = 0
    with torch.no_grad():
        for data, target in test_loader:
            batch_size = data.shape[0]
            noise = torch.rand(batch_size, nz).to(device)
            data, target, noise  = data.to(device), target.to(device),noise.to(device)
           
            dataNorm = transformImg(data)
            output = model(dataNorm)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()
            

            for i in range(10):
                noise = torch.rand(batch_size, nz).to(device)
                targetBDBatch = torch.ones(batch_size).long().to(device)*i
                targetOneHotEncoding = convertToOneHotEncoding(targetBDBatch,numOfClasses).to(device)
                backDoors = (bdModel(targetOneHotEncoding,noise)).view(-1,3,BDSize,BDSize)
                dataBD = insertSingleBD(data,backDoors,i)
                outputBD = model(dataBD)
                test_lossBD = F.nll_loss(outputBD, targetBDBatch, reduction='sum').item()  
                predBD = outputBD.argmax(dim=1, keepdim=True) 
                correctBD = predBD.eq(targetBDBatch.view_as(predBD)).sum().item()
                print('Class ' + str(i))
                print('\nBackDoor Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_lossBD, correctBD, len(test_loader.dataset),
                100. * correctBD / len(test_loader.dataset)))

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    model = Net().to(device)
    bdModel = hiddenNet().to(device)
    optimizer = optim.Adam(model.parameters())
    optimizerBD = optim.Adam(bdModel.parameters())
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch,bdModel,optimizerBD)
        test(args, model, device, test_loader,bdModel)

        torch.save(model.state_dict(), "models/cifar10_cnn.pth")
        torch.save(bdModel.state_dict(), "models/cifar_bd.pth")


def test_with_defenses(model, device, test_loader, bdModel, filter_fn=None, isAutoencoder=False):
    """
    This function tests the model on clean and backdoor data after applying defense methods.
    """
    print('Testing with defenses...')
    model.eval()
    bdModel.eval()
    
    correct = 0
    total = 0
    correctBD = 0
    totalBD = 0

    test_loss = 0
    test_lossBD = 0
    
    if isAutoencoder:
        ae = AutoencoderDenoiser().to(device)
        ae.load_state_dict(torch.load("models/autoencoder.pth"))
        ae.eval()
        denoised_data = []
        denoised_labels = []
        for x, y in test_loader:
            x = x.to(device)
            x_denoised = ae(x).detach().cpu()
            denoised_data.append(x_denoised)
            denoised_labels.append(y)
    
        x_all = torch.cat(denoised_data)
        y_all = torch.cat(denoised_labels)
        test_loader = DataLoader(TensorDataset(x_all, y_all), batch_size=test_loader.batch_size)

    with torch.no_grad():
        for data, target in test_loader:
            batch_size = data.shape[0]
            data, target = data.to(device), target.to(device)
            
            # Process clean data
            logits = model(data, return_logits_only=True)  # Get logits
            probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
            preds = probs.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

            # Evaluate backdoor data
            for i in range(10):
                noise = torch.rand(batch_size, nz).to(device)
                targetBDBatch = torch.ones(batch_size).long().to(device) * i
                targetOneHotEncoding = convertToOneHotEncoding(targetBDBatch, numOfClasses).to(device)
                backDoors = bdModel(targetOneHotEncoding, noise).view(-1, 3, BDSize, BDSize)
                dataBD = insertSingleBD(data, backDoors, i)
                
                if isAutoencoder:
                    dataBD = ae(dataBD)
                    
                outputBD = model(dataBD)

                test_lossBD += F.nll_loss(outputBD, targetBDBatch, reduction='sum').item()
                predBD = outputBD.argmax(dim=1, keepdim=True)
                correctBD += predBD.eq(targetBDBatch.view_as(predBD)).sum().item()
                totalBD += targetBDBatch.size(0)
                
                
            # Optionally, apply a filter function to the probabilities if provided
            if filter_fn:
                preds = filter_fn(probs)
                mask = preds != -1
                correct += (preds[mask] == target[mask].cpu()).sum().item()
                total += mask.sum().item()
                
        # Calculate final accuracy and losses
        test_loss /= len(test_loader.dataset)
        test_lossBD /= len(test_loader.dataset)
        
        print("Correct: ", correct)
        print("Total : ", total)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {100. * correct / total:.0f}%')
        print(f'Backdoor Test set: Average loss: {test_lossBD:.4f}, Accuracy: {100. * correctBD / totalBD:.0f}%')

    return 100. * correct / total, 100. * correctBD / totalBD


if __name__ == '__main__':
    main()
    print('Two loss functions')