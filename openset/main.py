# main.py
import torch
import argparse
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset

from models import OrchidOpenSetModel
# 假设您原来的损失函数在models.py中
from models import compute_enhanced_adversarial_loss 
from train import train_fe_stage, train_dm_stage
from evaluate import evaluate_open_set

def main():
    parser = argparse.ArgumentParser(description='Orchid Open-Set Recognition Pipeline')
    parser.add_argument('--mode', type=str, required=True, choices=['train_fe', 'train_dm', 'evaluate'],
                        help='Execution mode: train feature extractor, train diffusion model, or evaluate.')
    parser.add_argument('--data_dir', type=str, default='/data/users/jw/data/openset/all',
                        help='Path to the OSR dataset directory.')
    parser.add_argument('--num_known_classes', type=int, default=74, help='Number of known classes for training.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for the current training stage.')
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Data Loading ---
    # (您原来的数据变换)
    train_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # --- Initialize Model ---
    model = OrchidOpenSetModel(num_known_classes=args.num_known_classes)

    # --- Execute based on mode ---
    if args.mode == 'train_fe':
        train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train_known'), train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        train_fe_stage(args.epochs, model, train_loader, DEVICE)

    elif args.mode == 'train_dm':
        model.load_state_dict(torch.load('openset_model_fe_trained.pth'))
        train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train_known'), train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        train_dm_stage(args.epochs, model, train_loader, DEVICE)

    elif args.mode == 'evaluate':
        model.load_state_dict(torch.load('openset_model_dm_trained.pth'))
        
        # Loader for prototype calculation
        train_known_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train_known'), test_transform)
        known_train_loader = DataLoader(train_known_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Loader for final testing (combines known and unknown test sets)
        test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'test'), test_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        evaluate_open_set(model, known_train_loader, test_loader, DEVICE, args.num_known_classes)

if __name__ == '__main__':
    main()