import os
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from datasets.poison_tool_cifar import get_backdoor_loader, get_test_loader
from torchvision import models

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fix random seeds for reproducibility
seed = 98
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_step(args, model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        loss.backward()
        optimizer.step()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def save_checkpoint(state, epoch, is_best, args):
    if is_best:
        filepath = os.path.join(args.save_root, f"ResNet18_{args.trigger_type}_{args.dataset}_target{args.target_label}_poison{args.poison_rate}_epoch{epoch}.tar")
        torch.save(state, filepath)


def select_model(args, dataset="CIFAR10", pretrained=False):
    """
    Selects and returns a ResNet18 model.
    
    Args:
        args: Arguments object containing model-related parameters.
        dataset (str): Name of the dataset (e.g., CIFAR10).
        pretrained (bool): Whether to use a pretrained model (if applicable).
    
    Returns:
        model: A PyTorch ResNet18 model instance.
    """
    num_classes = args.num_classes  # Number of output classes

    # Use ResNet18 architecture
    model = models.resnet18(pretrained=pretrained)
    # Modify the final fully connected layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Move the model to the appropriate device (GPU or CPU)
    model = model.to(device)

    return model


def main(args):
    logger = logging.getLogger(__name__)
    os.makedirs(args.log_root, exist_ok=True)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.log_root, 'output.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

    logger.info('----------- Backdoored Data Initialization --------------')
    _, backdoor_data_loader = get_backdoor_loader(args)
    clean_test_loader, bad_test_loader = get_test_loader(args)

    logger.info('----------- ResNet18 Model Initialization --------------')
    net = select_model(args, dataset=args.dataset, pretrained=False)
    print(net)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    logger.info('----------- ResNet18 Model Training--------------')
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    for epoch in range(0, args.epochs + 1):
        start = time.time()
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_step(args=args, model=net, criterion=criterion, optimizer=optimizer,
                                           data_loader=backdoor_data_loader)
        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=bad_test_loader)
        scheduler.step()
        end = time.time()
        logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc)

        if epoch % args.save_every == 0 and epoch != 0 and epoch >= 50:
            # Save checkpoint at interval epoch
            is_best = True
            save_checkpoint({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'clean_acc': cl_test_acc,
                'bad_acc': po_test_acc,
                'optimizer': optimizer.state_dict(),
            }, epoch, is_best, args)
            logger.info('[INFO] Save model weight epoch {}'.format(epoch))


def get_arguments():
    parser = argparse.ArgumentParser()

    # Various path
    parser.add_argument('--save_every', type=int, default=5, help='save checkpoints every few epochs')
    parser.add_argument('--log_root', type=str, default='/kaggle/working/logs/', help='logs are saved here')
    parser.add_argument('--save_root', type=str, default='/kaggle/working/weights/', help='where to save the weight')
    parser.add_argument('--schedule', type=int, nargs='+', default=[40, 80],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')

    # Backdoor attacks
    parser.add_argument('--target_label', type=int, default=0, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
    parser.add_argument('--poison_rate', type=float, default=0.1, help='ratio of backdoor poisoned data')

    return parser


if __name__ == '__main__':
    args = get_arguments().parse_args()

    # Ensure directories exist
    os.makedirs(args.log_root, exist_ok=True)
    os.makedirs(args.save_root, exist_ok=True)

    # Run the main function
    main(args)
