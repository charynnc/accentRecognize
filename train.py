import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from conformer import Conformer
from models.custom_model import CustomModel
from dataloader import get_dataloader

def train(args):
    wandb.init(project="accent", name=args.model, config=args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create DataLoaders
    train_loader = get_dataloader(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        split='train',
        num_workers=args.num_workers,
        n_mels=args.n_mels,
        augment=args.augment
    )
    
    val_loader = get_dataloader(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        split='val',
        num_workers=args.num_workers,
        n_mels=args.n_mels,
        augment=args.augment
    )

    # Get number of classes from dataset
    num_classes = len(train_loader.dataset.accent_to_index)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_loader.dataset.accent_to_index}")

    # Initialize Model
    if args.model == 'conformer':
        model = Conformer(
            num_classes=num_classes,
            input_dim=args.n_mels,
            encoder_dim=args.encoder_dim,
            num_encoder_layers=args.num_encoder_layers,
            num_attention_heads=args.num_attention_heads
        ).to(device)
    elif args.model == 'custom_model':
        model = CustomModel(
            num_classes=num_classes,
            input_dim=args.n_mels,
            encoder_dim=args.encoder_dim
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training Loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            inputs = batch['inputs'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            targets = batch['accent_indices'].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = model(inputs, input_lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            
            if targets.dim() > 1:
                target_indices = torch.argmax(targets, dim=1)
            else:
                target_indices = targets
                
            correct += (predicted == target_indices).sum().item()
            
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100 * correct / total})
            wandb.log({
                "train_step_loss": loss.item(),
                "train_step_acc": 100 * correct / total
            })

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Validation {epoch+1}/{args.epochs}")
            for batch in val_pbar:
                inputs = batch['inputs'].to(device)
                input_lengths = batch['input_lengths'].to(device)
                targets = batch['accent_indices'].to(device)

                outputs = model(inputs, input_lengths)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                
                if targets.dim() > 1:
                    target_indices = torch.argmax(targets, dim=1)
                else:
                    target_indices = targets
                    
                val_correct += (predicted == target_indices).sum().item()
                val_pbar.set_postfix({'val_loss': val_loss / (val_pbar.n + 1), 'val_acc': 100 * val_correct / val_total})

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved best model with Acc: {best_acc:.2f}%")

    print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Conformer for Classification')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--n_mels', type=int, default=80, help='Number of mel filterbanks')
    parser.add_argument('--encoder_dim', type=int, default=256, help='Encoder dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_attention_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--augment', type=bool, default=True, help='Whether to use data augmentation')
    parser.add_argument('--model', type=str, default='conformer', choices=['conformer', 'custom_model'], help='Model to use')

    args = parser.parse_args()
    
    # Update save_dir based on model if not explicitly set to a model-specific path
    # But user script sets it explicitly. Let's just append model name if it's a generic checkpoint dir?
    # Or better, just rely on the script to pass the right dir. 
    # However, the user asked to "modify savedir". 
    # Let's assume the script passes a base dir or we append the model name.
    # Given the existing script passes ./checkpoints/conformer, let's change the script to pass ./checkpoints 
    # and let python append the model name, OR change the script to pass ./checkpoints/custom_model.
    # I will modify the script to pass the full path, but I will also make sure the code creates the directory.
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train(args)