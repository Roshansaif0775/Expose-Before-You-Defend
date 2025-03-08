import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import logging
import os
import sys

from datasets.poison_tool_cifar import get_test_loader, get_train_loader, split_dataset
from exposes import unlearn
from exposes.utils import load_state_dict
import models
from models.ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

# Set device (Use GPU if available on Kaggle)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# Set random seed for reproducibility
seed = 98
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
np.random.seed(seed)

if __name__ == '__main__':
    print("Execution Started...")

    # Logger setup (Removed FileHandler)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG
    )

    logger.info("Execution started.")

    # Manually defining arguments (since argparse is not needed in Kaggle)
    class Args:
        target_label = 0
        trigger_type = 'gridTrigger'
        target_type = 'all2one'
        trig_w = 3
        trig_h = 3
        dataset = 'CIFAR10'
        ratio = 0.01
        batch_size = 128
        num_classes = 10
        img_size = 32
        backdoor_model_path = '/kaggle/input/backdoor-model/ResNet18-Backdoor.pth'  # Adjust accordingly
        output_model_path = None
        output_logs_path = 'exposes/logs/'
        arch = 'resnet18'

    args = Args()

    # Data Preparation
    logger.info('----------- Data Initialization --------------')
    print("Splitting dataset and preparing data loaders...")

    split_set = split_dataset(args.dataset, args.ratio)
    logger.info(f"Splitting dataset: {args.dataset} with defense ratio: {args.ratio}")
    print("Defense dataset split successfully.")

    defense_data_loader = DataLoader(split_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    logger.info("Defense data loaded into DataLoader.")

    print("Loading clean and poisoned test sets...")
    clean_test_loader, bad_test_loader = get_test_loader(args)
    logger.info("Test sets loaded successfully.")

    data_loader = {
        'defense_loader': defense_data_loader,
        'clean_test_loader': clean_test_loader,
        'bad_test_loader': bad_test_loader
    }

    # Model Initialization
    logger.info('----------- Model Initialization --------------')
    print("Initializing the model...")

    net = ResNet18(num_classes=args.num_classes)
    logger.info("ResNet18 model initialized.")

    print("Adjusting first convolution layer for CIFAR-10...")
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    logger.info("First convolution layer modified.")

    # Model loading with error handling
    try:
        logger.info(f"Loading model from: {args.backdoor_model_path}")
        print(f"Loading backdoored model from {args.backdoor_model_path}...")
        state_dict = torch.load(args.backdoor_model_path, map_location=device)
        net.load_state_dict(state_dict, strict=False)
        logger.info("Model loaded successfully!")
        print("Model loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Model file not found at {args.backdoor_model_path}. Please check the path.")
        print(f"ERROR: Model file not found at {args.backdoor_model_path}. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading the model: {e}")
        print(f"ERROR: Failed to load the model: {e}")
        sys.exit(1)

    net = net.to(device)
    logger.info(f"Model moved to {device}.")

    # Exposing the model
    logger.info('----------- Model Exposing Strategy --------------')
    print("Applying unlearning strategy to remove backdoor...")

    try:
        unlearn_process = unlearn.Unlearning(args, logger, net, data_loader)
        unlearn_process.do_expose()
        logger.info("Unlearning process completed successfully.")
        print("Unlearning process completed successfully.")
    except Exception as e:
        logger.error(f"Unlearning process encountered an issue: {e}")
        print(f"ERROR: Unlearning failed: {e}")
        sys.exit(1)

    logger.info("Execution completed.")
    print("Process finished successfully.")
