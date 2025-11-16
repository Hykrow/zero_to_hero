import torch
import matplotlib.pyplot as plt
words = open("names.txt", "r").read().splitlines()
N = torch.zeros((27, 27), dtype = torch.int32)

stoi = lambda x: 26 if x == "<.>" else max(0, min(26, (ord(x)-ord('a'))))
for w in words:
    chs = ["<.>"]+ list(w) + ["<.>"]
    for ch1, ch2 in zip(chs, chs[1:]):# shift de 2. 
        i1 = stoi(ch1)
        i2 = stoi(ch2)
        #print(i1, i2)
        #print(N[i1, i2])
        N[i1, i2] +=1 
print(N)