import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from collections import OrderedDict
from torchvision import models
from datasets.poison_tool_cifar import get_backdoor_loader, get_test_loader

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] - Using Device: {device}")

# Set deterministic behavior
seed = 98
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
np.random.seed(seed)
print(f"[INFO] - Random seed set to: {seed}")

# Define arguments manually
class Args:
    trig_w = 3
    trig_h = 3
    save_every = 5
    log_root = '/kaggle/working/logs/'
    save_root = '/kaggle/working/weights/'
    model_name = 'ResNet18'
    schedule = [40, 80]
    dataset = 'CIFAR10'
    batch_size = 128
    lr = 0.1
    epochs = 60
    target_label = 0
    trigger_type = 'gridTrigger'
    poison_rate = 0.1
    inject_portion = 0.1
    target_type = 'all2one'

args = Args()
os.makedirs(args.log_root, exist_ok=True)
os.makedirs(args.save_root, exist_ok=True)

# Data Loading
print("[STEP] - Initializing Data Loaders...")
_, backdoor_data_loader = get_backdoor_loader(args)
clean_test_loader, bad_test_loader = get_test_loader(args)
print("[INFO] - Data loaders initialized.")

# Model Initialization
print("[STEP] - Initializing Model...")
def select_model(model_name, weights=None):
    if model_name == 'ResNet18':
        return models.resnet18(weights=weights)
    raise ValueError(f"ERROR: Model {model_name} is not supported.")

net = select_model(args.model_name, weights=None)  # For no pre-trained weights
print(f"[INFO] - Model {args.model_name} initialized.")

# Loss & Optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

print("[INFO] - Starting Training Loop...")
for epoch in range(args.epochs + 1):
    print(f"\n[MAIN] - Epoch {epoch}/{args.epochs} Started")
    start_time = time.time()
    
    # Training Step
    net.train()
    total_correct, total_loss = 0, 0.0
    for images, labels in backdoor_data_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        total_loss += loss.item()
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
    
    train_loss = total_loss / len(backdoor_data_loader)
    train_acc = total_correct / len(backdoor_data_loader.dataset)
    print(f"[TRAIN] - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    
    # Testing Step (Clean and Poisoned)
    def test(data_loader, test_type="Clean"):
        net.eval()
        total_correct, total_loss = 0, 0.0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                output = net(images)
                loss = criterion(output, labels).item()
                total_loss += loss
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.view_as(pred)).sum().item()
        return total_loss / len(data_loader), total_correct / len(data_loader.dataset)
    
    cl_test_loss, cl_test_acc = test(clean_test_loader, "Clean")
    po_test_loss, po_test_acc = test(bad_test_loader, "Poisoned")
    scheduler.step()
    
    print(f"[TEST] - Clean Loss: {cl_test_loss:.4f}, Accuracy: {cl_test_acc:.4f}")
    print(f"[TEST] - Poisoned Loss: {po_test_loss:.4f}, Accuracy: {po_test_acc:.4f}")
    
    if epoch % args.save_every == 0 and epoch >= 50:
        save_path = os.path.join(args.save_root, f"{args.model_name}_epoch{epoch}.pth")
        torch.save(net.state_dict(), save_path)
        print(f"[SAVE] - Model saved at {save_path}")
    
    end_time = time.time()
    print(f"[MAIN] - Epoch {epoch} completed in {end_time - start_time:.2f} sec.")

print("[INFO] - Training Completed!")
