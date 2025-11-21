import torch
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F

words = open("names.txt", "r").read().splitlines()


def stoi(x):
    return 0 if x == "." else max(1, min(26, (1 + ord(x) - ord("a"))))


def itos(x):
    return "." if x == 0 else chr(ord("a") + x - 1)


class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros((fan_in, fan_out))

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight, self.bias]


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_std = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.std(0, keepdim=True)
            self.running_mean = (
                self.running_mean * (1 - self.eps) + self.running_mean * self.eps
            )
            self.running_std = (
                self.running_std * (1 - self.eps) + self.running_std * self.eps
            )
        else:
            xmean = x - self.running_mean
            xvar = x - self.running_std
        self.out = self.gamma * xmean / (xvar + self.eps) + self.beta

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


print(stoi(itos(26)))
context_length = 3


def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * context_length
        for ch in w + ".":
            ix = stoi(ch)
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


import random

random.seed(32)
random.shuffle(words)
ntr = int(0.8 * len(words))
nv = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:ntr])
Xv, Yv = build_dataset(words[ntr:nv])
Xte = build_dataset(words[nv:])

n_embd = 10
C = torch.randn((27, n_embd))

n_hidden = 200
W1 = torch.randn((context_length * C.shape[1], n_hidden)) / math.sqrt(
    n_embd * context_length
)
# b1 = torch.zeros(n_hidden)

W2 = torch.randn(n_hidden, 27) * 0.01
b2 = torch.zeros(27)
bngain = torch.ones((1, n_hidden))  # 1 pour les batch
bnbias = torch.zeros((1, n_hidden))  # 1 pour les batch
parameters = [C, W1, W2, b2, bngain, bnbias]

for p in parameters:
    p.requires_grad = True
print(sum([p.nelement() for p in parameters]))

lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre
lossi = []
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))
for i in range(200000):
    ix = torch.randint(0, Xtr.shape[0], (32,))

    emb = C[Xtr[ix]]  # N, context, 2
    embcat = emb.view(-1, context_length * C.shape[1])  # N, context*2
    hpreact = embcat @ W1
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True) + 0.001

    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + bnmeani * 0.001
        bnstd_running = 0.999 * bnstd_running + bnstdi * 0.001

    hpnorm = (
        bngain * (hpreact - bnmeani) / bnstdi  # mean , std sur N
        + bnbias
    )
    h = torch.tanh(hpnorm)  # N, 100
    logits = h @ W2 + b2  # N, 27

    loss = F.cross_entropy(logits, Ytr[ix])
    for p in parameters:
        p.grad = None
    loss.backward()
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        assert p.grad is not None
        p.data += -lr * p.grad
    lossi.append(loss.item())


emb = C[Xv]  # N, context, 2
embcat = emb.view(-1, context_length * C.shape[1])
hpreact = embcat @ W1
hpnorm = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
h = torch.tanh(hpnorm)
logits = h @ W2 + b2  # N, 27

loss = F.cross_entropy(logits, Yv)
print(loss.item())
