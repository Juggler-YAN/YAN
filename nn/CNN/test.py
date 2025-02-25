import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn

from data.FashionMNIST import load_data
from model.net import densenet, init_weights

# random
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

# args
class Args:
    def __init__(self) -> None:
        self.batch_size = 16
        self.lr = 0.1
        self.epochs = 2
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataset
class Dataset():
    def __init__(self, flag='train') -> None:
        self.flag = flag
        assert self.flag in ['train', 'val'], 'not implement!'
        if self.flag == 'train':
            # MLP, LeNet
            # self.train_data, self.test_data = load_data(args.batch_size)
            # AlexNet, VGGNet, NiNNet, GoogLeNet, ResNet, DenseNet
            self.train_data, self.test_data = load_data(args.batch_size, resize=224)
        else:
            self.train_data, self.test_data = load_data(args.batch_size, resize=224)

# model

# loss

# opti

# train
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter, device=None):
    if isinstance(net, torch.nn.Module):
        net.eval()
    acc_nums = 0
    nums = 0
    with torch.no_grad():
        for X, y in data_iter:
            # if isinstance(X, list):
            #     # BERT微调所需的（之后将介绍）
            #     X = [x.to(device) for x in X]
            # else:
            #     X = X.to(device)
            # y = y.to(device)
            X, y = X.to(device), y.to(device)
            acc_nums = acc_nums + accuracy(net(X), y)
            nums = nums + y.numel()
    return acc_nums / nums

def train():
    train_loss_arr = []
    train_acc_arr = []
    test_acc_arr = []
    for epoch in range(args.epochs):
        if isinstance(net, torch.nn.Module):
            net.train()
        train_loss = 0
        train_acc = 0
        train_nums = 0
        for X, y in data.train_data:
            trainer.zero_grad()
            X, y = X.to(args.device), y.to(args.device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.mean().backward()
            trainer.step()
            train_loss = train_loss + float(l.sum())
            train_acc =  train_acc + accuracy(y_hat, y)
            train_nums = train_nums + y.numel()
        train_metrics = [train_loss / train_nums, train_acc / train_nums]
        test_acc = evaluate_accuracy(net, data.test_data, args.device)
        print(f'{epoch + 1}:')
        print(f'train_loss = {train_metrics[0]}')
        print(f'train_acc = {train_metrics[1]}')
        print(f'test_acc = {test_acc}')
        train_loss_arr.append(train_metrics[0])
        train_acc_arr.append(train_metrics[1])
        test_acc_arr.append(test_acc)

    # plot
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_loss_arr, label='train_loss')
    plt.title("loss")
    plt.legend()
    plt.subplot(122)
    plt.plot(train_acc_arr, label='train_acc')
    plt.plot(test_acc_arr, label='test_acc')
    plt.title("acc")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # args
    args = Args()
    # dataset
    data = Dataset()
    # model
    net = densenet('densenet264')
    net.apply(init_weights)
    net = net.to(args.device)
    # loss
    loss = nn.CrossEntropyLoss(reduction='none')
    # opti
    trainer = torch.optim.SGD(net.parameters(), lr=args.lr)
    # train
    train()
    # predict
