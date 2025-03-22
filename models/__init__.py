import mlconfig
import torch
from . import ToyModel, ISBBA_resnet, dynamic_models, ResNetWithAT
from torch.nn import CrossEntropyLoss

# Register mlconfig
mlconfig.register(ToyModel.ToyModel)
mlconfig.register(ISBBA_resnet.resnet18_200)
mlconfig.register(ResNetWithAT.ResnetWithAT)

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
