'''Train CIFAR10 with PyTorch.'''
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F, torch.backends.cudnn as cudnn 
import torchvision, torchvision.transforms as transforms
import os, argparse
from torch.utils.tensorboard import SummaryWriter
from models import *


# Training
def train(epoch, args):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_losses = [] 
    train_acc = [] 
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if args.grad_clip: nn.utils.clip_grad_value_(net.parameters(), clip_value=args.grad_clip) 
        optimizer.step()

        train_loss += loss.item()
        train_losses.append(train_loss)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item() 

        train_acc.append(100.*correct/total) 

        # print('Batch_idx: %d | Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'% (batch_idx, train_loss/(batch_idx+1), 100.*correct/total, correct, total)) 
        break 

    writer.add_scalar('Loss/train_loss', np.mean(train_losses), epoch) 
    writer.add_scalar('Accuracy/train_accuracy', np.mean(train_acc), epoch) 
    


def test(epoch, args):
    global best_acc
    net.eval()
    test_loss = 0
    test_losses = [] 
    test_acc = [] 
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            test_losses.append(test_loss)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item() 
            test_acc.append(100.*correct/total) 

            # print('Batch_idx: %d | Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'% ( batch_idx, test_loss/(batch_idx+1), 100.*correct/total, correct, total)) 
            break 

        writer.add_scalar('Loss/test_loss', np.mean(test_losses), epoch) 
        writer.add_scalar('Accuracy/test_accuracy', np.mean(test_acc), epoch) 

    # Save checkpoint.
    acc = 100.*correct/total
    if acc != best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'args': args
        }
        # if not os.path.isdir('summaries'):
        #     os.mkdir('summaries')
        torch.save(state, os.path.join('./summaries/', args.exp, 'ckpt.pth'))
        best_acc = acc


if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    
    parser.add_argument('--exp', default='test_exp', type=str, help='experiment name')

    parser.add_argument('--optim', default='sgd', type=str, help='sgd/adam')
    parser.add_argument('--lr_sched', default='CosineAnnealingLR', type=str, help='lr schedulers for sgd')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd')

    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay for optimizer')

    parser.add_argument('--batch_size', default=128, type=int, help='bathc size for training and testing')
    parser.add_argument('--num_workers', default=2, type=int, help='num workers for data loader')

    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume_ckpt', type=str, help='resume from checkpoint')
 
    parser.add_argument('--data_augmentation', action='store_true', help='augment data or not')
    parser.add_argument('--data_normalize', action='store_true', help='normalize data or not')
    
    parser.add_argument('--grad_clip', default=None, type=float, help='grad_clip')

    
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    train_trans = [transforms.ToTensor()]
    test_trans = [transforms.ToTensor()]

    if args.data_augmentation: 
        train_trans.append(transforms.RandomCrop(32, padding=4)) 
        train_trans.append(transforms.RandomHorizontalFlip()) 

    if args.data_normalize: 
        train_trans.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))) 
        test_trans.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))) 


    transform_train = transforms.Compose(train_trans) 

    transform_test = transforms.Compose(test_trans) 

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=int(args.batch_size/4), shuffle=False, num_workers=args.num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = ResNet9()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume_ckpt)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()

    if args.optim == 'sgd': 
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) 
        if args.lr_sched == 'CosineAnnealingLR': scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) 
        if args.lr_sched == 'LambdaLR': scheduler =torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.65 ** epoch)
        if args.lr_sched == 'MultiplicativeLR': scheduler =torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.65 ** epoch)
        if args.lr_sched == 'StepLR': scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1) 
        if args.lr_sched == 'MultiStepLR': scheduler =torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6,8,9], gamma=0.1) 
        if args.lr_sched == 'ExponentialLR': scheduler =torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1) 
        if args.lr_sched == 'CyclicLR': scheduler =torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular") 
        if args.lr_sched == 'CyclicLR2': scheduler =torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular2") 
        if args.lr_sched == 'CyclicLR3': scheduler =torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="exp_range",gamma=0.85) 
        if args.lr_sched == 'OneCycleLR': scheduler =torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10) 
        if args.lr_sched == 'OneCycleLR2': scheduler =torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10,anneal_strategy='linear') 
        if args.lr_sched == 'CosineAnnealingWarmRestarts': scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1) 


    if args.optim == 'adam':         
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    writer = SummaryWriter('summaries/'+args.exp) 

    for epoch in range(start_epoch, start_epoch+200):
        train(epoch, args) 
        test(epoch, args) 
        if args.optim == 'sgd': scheduler.step()
    writer.close() 
