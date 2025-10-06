import argparse
import os
import sys
import torch
import yaml
import psutil
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Unified Configuration Parser for Super-Resolution Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # DEVICE SETTINGS

    device_group = parser.add_argument_group('Device Configuration')
    device_group.add_argument('--n_threads', type=int, default=8, help='Number of CPU threads for data loading')
    device_group.add_argument('--cpu', action='store_true', help='Force CPU only')
    device_group.add_argument('--n_GPUs', type=int, default=torch.cuda.device_count(), help='Number of available GPUs')
    device_group.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')

    # TRAINING CONFIGURATION

    train_group = parser.add_argument_group('Training Settings')
    train_group.add_argument('--model', default='HiT-RSNet', help='Model name')
    train_group.add_argument('--test_only', action='store_true', help='Run model in evaluation mode only')
    train_group.add_argument('--scale', type=str, default='3', help='Super-resolution scale(s), e.g., 2+3+4')
    train_group.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    train_group.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    train_group.add_argument('--patch_size', type=int, default=64, help='Training patch size (HR)')
    train_group.add_argument('--loss', type=str, default='1*L1', help='Loss function configuration string')

    # OPTIMIZER SETTINGS

    optim_group = parser.add_argument_group('Optimization')
    optim_group.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop'), help='Optimizer type')
    optim_group.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    optim_group.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    optim_group.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='Betas for Adam optimizer')
    optim_group.add_argument('--epsilon', type=float, default=1e-6, help='Epsilon for numerical stability')
    optim_group.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 regularization)')
    optim_group.add_argument('--decay', type=str, default='200', help='Learning rate step decay milestones')
    optim_group.add_argument('--gamma', type=float, default=0.5, help='Learning rate decay factor')
    optim_group.add_argument('--gclip', type=float, default=0, help='Gradient clipping threshold')
    optim_group.add_argument('--skip_threshold', type=float, default=1e8, help='Skip batch if loss > threshold')


    # NETWORK ARCHITECTURE

    net_group = parser.add_argument_group('Model Architecture')
    net_group.add_argument('--n_feats', type=int, default=64, help='Number of feature maps')
    net_group.add_argument('--n_resgroups', type=int, default=4, help='Number of residual groups')
    net_group.add_argument('--reduction', type=int, default=8, help='Channel reduction ratio')
    net_group.add_argument('--res_scale', type=float, default=1.0, help='Residual scaling factor')
    net_group.add_argument('--act', type=str, default='relu', help='Activation function')


    # DATA SETTINGS
    
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument('--dir_data', type=str, default='./datasets/rsscn7', help='Root directory for dataset')
    data_group.add_argument('--dir_demo', type=str, default='./datasets/rsscn7/test/', help='Demo image directory')
    data_group.add_argument('--data_train', type=str, default='./datasets/rsscn7/train', help='Training dataset(s), e.g., DIV2K+Flickr2K')
    data_group.add_argument('--data_test', type=str, default='./datasets/rsscn7/test', help='Testing dataset(s)')
    data_group.add_argument('--data_range', type=str, default='0001-1400/1401-2800', help='Train/test split range')
    data_group.add_argument('--ext', type=str, default='sep', help='Dataset file extension type')
    data_group.add_argument('--rgb_range', type=int, default=255, help='Maximum RGB value')
    data_group.add_argument('--n_colors', type=int, default=3, help='Number of input color channels')

    # MISC SETTINGS

    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('--precision', type=str, default='single', choices=('single', 'half'), help='Floating precision for inference')
    misc_group.add_argument('--reset', action='store_true', help='Force model retraining from scratch')
    misc_group.add_argument('--pre_train', type=str, default='', help='Pre-trained model path')
    misc_group.add_argument('--extend', type=str, default='.', help='Path to pre-trained model extension')
    misc_group.add_argument('--test_every', type=int, default=10, help='Test interval (in batches)')
    misc_group.add_argument('--split_batch', type=int, default=1, help='Split mini-batch into smaller chunks for memory efficiency')
    misc_group.add_argument('--save', type=str, default='test', help='Output folder name')
    misc_group.add_argument('--load', type=str, default='', help='Model checkpoint to resume from')
    misc_group.add_argument('--resume', type=int, default=0, help='Resume epoch')
    misc_group.add_argument('--save_models', action='store_true', help='Save intermediate model checkpoints')
    misc_group.add_argument('--print_every', type=int, default=10, help='Batch print frequency')
    misc_group.add_argument('--save_results', action='store_true', help='Save SR, LR, HR images after inference')
    misc_group.add_argument('--save_gt', action='store_true', help='Save ground-truth images together with results')


    parser.add_argument('--config', type=str, default='', help='Optional YAML config file (overrides CLI args)')


    args = parser.parse_args()

    # Optional YAML override
    if args.config and os.path.isfile(args.config):
        with open(args.config, 'r') as f:
            yaml_args = yaml.safe_load(f)
        print(f"Loading configuration from {args.config}")
        for key, value in yaml_args.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Format scale(s) and dataset names
    args.scale = list(map(int, args.scale.split('+')))
    args.data_train = args.data_train.split('+')
    args.data_test = args.data_test.split('+')

    # Adjust epochs
    if args.epochs == 0:
        args.epochs = int(1e8)

    # Convert True/False strings
    for k, v in vars(args).items():
        if isinstance(v, str):
            if v.lower() == 'true': setattr(args, k, True)
            elif v.lower() == 'false': setattr(args, k, False)


    if not torch.cuda.is_available() and not args.cpu:
        print("CUDA not detected — falling back to CPU.")
        args.cpu = True

    if args.n_GPUs > torch.cuda.device_count():
        print(f"Requested {args.n_GPUs} GPUs, but only {torch.cuda.device_count()} available.")
        args.n_GPUs = torch.cuda.device_count()

    if args.seed <= 0:
        print("Invalid seed, resetting to 1.")
        args.seed = 1

    return args


def print_config_summary(args):
    print("\n Configuration Summary")
    print("────────────────────────────────────────────")
    for key, value in sorted(vars(args).items()):
        print(f"{key:>20}: {value}")
    print("────────────────────────────────────────────")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Available CPUs: {psutil.cpu_count(logical=True)}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print("────────────────────────────────────────────\n")


if __name__ == '__main__':
    args = parse_arguments()
    print_config_summary(args)
