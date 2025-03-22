import mlconfig
import torch
from . import ToyModel, ISBBA_resnet, dynamic_models, ResNetWithAT
from torch.nn import CrossEntropyLoss

# Register mlconfig


# Optimizers
mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.Adam)

# Learning rate schedulers
mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(torch.optim.lr_scheduler.StepLR)
mlconfig.register(torch.optim.lr_scheduler.ExponentialLR)

# Loss functions
mlconfig.register(torch.nn.CrossEntropyLoss)
