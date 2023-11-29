import torch
from torch import nn
from typing import Union, List, Optional
import accelerate
import seaborn as sns
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tracer import Activation_tracer

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),            
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

accelerator = accelerate.Accelerator()
device = accelerator.device
model = NeuralNetwork().to(device)
# print(list(model.named_modules()))

tracer = Activation_tracer(model, list(model.modules()))
model.eval()
input_sample = torch.rand((1,28,28)).to(device)
model(input_sample)
# print(tracer.activation_pool)
pool = tracer.export_pool()
tracer.get_figure('seperate', [pool])
heat_pool = {}
for k, v in pool.items():
    if isinstance(k, nn.ReLU):
        heat_pool[k] = v
tracer.get_figure('all_layer', [pool])