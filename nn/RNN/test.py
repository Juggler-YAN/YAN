#
# RNN
# 2024.11.18
#

import numpy as np
import math
import matplotlib.pyplot as plt
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.text import load_data

# random
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

# args
class Args:
    def __init__(self) -> None:
        self.filename = './data/timemachine.txt'
        self.batch_size = 32
        self.lr = 1
        self.epochs = 500
        self.steps = 35
        self.num_hiddens = 256
        self.num_layers = 2
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataset
class Dataset():
    def __init__(self, flag='train') -> None:
        self.flag = flag
        assert self.flag in ['train', 'val'], 'not implement!'
        if self.flag == 'train':
            self.train_data, self.vocab = load_data(args.filename, args.batch_size, args.steps)
        else:
            self.train_data, self.vocab = load_data(args.filename, args.batch_size, args.steps)

# model
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # RNN双向，num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # (时间步数*批量大小,隐藏单元数) -> (时间步数*批量大小,词表大小)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # GRU隐状态——张量
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                                device=device)
        else:
            # LSTM隐状态——元组
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))

# train
def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch(use_random_iter):
    state = None
    start = time.time()
    metric = [0.0, 0.0]
    for X, Y in data.train_data:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=args.device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(args.device), y.to(args.device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        trainer.zero_grad()
        l.backward()
        grad_clipping(net, 1)
        trainer.step()
        metric[0] += l * y.numel()
        metric[1] += y.numel()
    return math.exp(metric[0] / metric[1]), metric[1] / (time.time() - start)

# predict
def predict(prefix, num_preds):
    state = net.begin_state(batch_size=1, device=args.device)
    outputs = [data.vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=args.device).reshape((1, 1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(data.vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([data.vocab.idx_to_token[i] for i in outputs])

def train(use_random_iter=False):
    train_loss = []
    for epoch in range(args.epochs):
        ppl, speed = train_epoch(use_random_iter)
        train_loss.append(ppl)
        if (epoch + 1) % 100 == 0:
            print(predict('time traveller', 50))
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(args.device)}')
    print(predict('time traveller', 50))
    print(predict('traveller', 50))
    plt.plot(train_loss, label='train_loss')
    plt.title("loss")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # args
    args = Args()
    # dataset
    data = Dataset()
    # net
    rnn_layer = nn.RNN(len(data.vocab), args.num_hiddens)
    gru_layer = nn.GRU(len(data.vocab), args.num_hiddens)
    lstm_layer = nn.LSTM(len(data.vocab), args.num_hiddens)
    # +深度
    lstm_layer_1 = nn.LSTM(len(data.vocab), args.num_hiddens, args.num_layers)
    # +双向
    lstm_layer_2 = nn.LSTM(len(data.vocab), args.num_hiddens, args.num_layers, bidirectional=True)
    net = RNNModel(lstm_layer_2, vocab_size=len(data.vocab))
    net = net.to(args.device)
    # loss
    loss = nn.CrossEntropyLoss()
    # opti
    trainer = torch.optim.SGD(net.parameters(), args.lr)
    # train
    train()
    # predict
    predict('time traveller', 10)