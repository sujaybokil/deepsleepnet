import torch
import torchvision
import torch.nn as nn
from model import DeepSleepNet

EPOCHS = 100
LR = 0.01

model = DeepSleepNet(training=True)
optimizer = torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-3) # set optimizer
criterion = nn.CrossEntropyLoss() 

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    ## Data Loading Sequence ##

    outputs = model(X)
    loss = criterion(outputs,y)

    loss.backward()
    optimizer.step()    