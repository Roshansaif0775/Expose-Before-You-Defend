import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import logging
import os
import sys

from datasets.poison_tool_cifar import get_test_loader, split_dataset
from exposes import unlearn
import models
from torchvision import models as torchvision_models  # For loading ResNet18

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

    # Logger setup
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG
    )

    logger.info("Execution started.")

    # Manually defining arguments
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
        backdoor_model_path = "/kaggle/working/weights/ResNet18_epoch60.pth"  # Load latest trained model
        output_model_path = None
        output_logs_path = 'exposes/logs/'
        arch = 'resnet18'

    args = Args()

    # Data Preparation
    logger.info('----------- Data Initialization --------------')
    print("Splitting dataset and preparing data loaders...")

    split_set = split_dataset(args.dataset, args.ratio)
    defense_data_loader = DataLoader(split_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    clean_test_loader, bad_test_loader = get_test_loader(args)

    data_loader = {
        'defense_loader': defense_data_loader,
        'clean_test_loader': clean_test_loader,
        'bad_test_loader': bad_test_loader
    }

    # Model Initialization
    logger.info('----------- Model Initialization --------------')
    print("Initializing the model...")

    def load_model(arch, num_classes):
        if arch == "resnet18":
            model = torchvision_models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        raise ValueError(f"Unsupported model architecture: {arch}")

    net = load_model(args.arch, args.num_classes)
    net = net.to(device)

    # Load trained model
    try:
        logger.info(f"Loading model from: {args.backdoor_model_path}")
        print(f"Loading trained model from {args.backdoor_model_path}...")
        state_dict = torch.load(args.backdoor_model_path, map_location=device)
        net.load_state_dict(state_dict, strict=False)
        logger.info("Model loaded successfully!")
    except FileNotFoundError:
        logger.error(f"Model file not found at {args.backdoor_model_path}.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading the model: {e}")
        sys.exit(1)

    # Exposing the model
    logger.info('----------- Model Exposing Strategy --------------')
    print("Applying unlearning strategy to remove backdoor...")

    try:
        unlearn_process = unlearn.Unlearning(args, logger, net, data_loader)
        unlearn_process.do_expose()
        logger.info("Unlearning process completed successfully.")
    except Exception as e:
        logger.error(f"Unlearning process failed: {e}")
        sys.exit(1)

    logger.info("Execution completed.")
    print("Process finished successfully.")
