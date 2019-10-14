import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from visualize import Visualizer

import sys
import os
import argparse

from model.dc_module import DCNet

def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    # set up root for training dataset
    parser.add_argument('--train_root', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)

    # set up magnitude function
    parser.add_argument('--magnitude', type=str, default=None, choices=[None, 'ball', 'linear', 'seg'])
    
    # set up angular function
    parser.add_argument('--angular', type=str, default='cos', choices=[None, 'cos', 'sqcos'])
    
    parser.add_argument('--gpu_idx', type=str, default=None)
    
    parser.add_argument('--log_interval', type=int, default=10)

    return parser.parse_args(argv)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f})%]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, visualizer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            embeddings = model.get_features()
            print(embeddings)
            visualizer.set_sample(target, embeddings)

            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    
    print('Test set: Average loss: {:.4f}, Acc.: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main(args):
    # check cuda availabilty
    if torch.cuda.is_available(): 
        device = 'cuda'
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_idx
    
    else:
        device = 'cpu'

    # model
    model = DCNet(magnitude=args.magnitude, angular=args.angular).to(device)

    # for data loader
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True, 
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.13,), (0.30,))
                    ])),
                batch_size=args.batch_size, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True, 
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.13,), (0.30,))
                    ])),
                batch_size=args.batch_size)
    
    print('magnitude function : ', args.magnitude, '\tangular function : ', args.angular)
    
    
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=2e-3, momentum=0.9)
    
    
    visualizer = Visualizer(10, 50, model, 512)

    # train
    for epoch in range(1, args.epochs +1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, visualizer)

    # save
    torch.save(model.state_dict(), "mnist.pt")

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
