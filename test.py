import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from conformer import Conformer
from models.custom_model import CustomModel
# from dataloaders.speech_accent_archive import get_dataloader
from dataloaders.st_cmds_preprocessed import get_dataloader

def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create DataLoader for test split
    test_loader = get_dataloader(
        root_dir=args.data_dir,
        feature_dir=os.path.join(args.data_dir, 'features'),
        batch_size=args.batch_size,
        split='test',
        num_workers=args.num_workers,
        augment=args.augment
    )

    # Get number of classes from dataset
    num_classes = len(test_loader.dataset.accent_to_index)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {test_loader.dataset.accent_to_index}")

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
            encoder_dim=args.encoder_dim,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Load trained model
    if os.path.isfile(args.model_path):
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print(f"Error: Model file not found at {args.model_path}")
        return

    # Loss function (for reporting loss)
    criterion = nn.CrossEntropyLoss()

    # Evaluation Loop
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    # For per-class accuracy
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch in pbar:
            inputs = batch['inputs'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            targets = batch['accent_indices'].to(device)

            outputs = model(inputs, input_lengths)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            
            if targets.dim() > 1:
                target_indices = torch.argmax(targets, dim=1)
            else:
                target_indices = targets
                
            correct += (predicted == target_indices).sum().item()
            
            # Per-class accuracy
            c = (predicted == target_indices).squeeze()
            for i in range(len(targets)):
                label = target_indices[i]
                if c.ndim == 0:
                    class_correct[label] += c.item()
                else:
                    class_correct[label] += c[i].item()
                class_total[label] += 1

                pred_name = test_loader.dataset.index_to_accent[predicted[i].item()]
                true_name = test_loader.dataset.index_to_accent[label.item()]
                # pbar.write(f"Predicted: {pred_name}, True: {true_name}")

            pbar.set_postfix({'loss': test_loss / (pbar.n + 1), 'acc': 100 * correct / total})

    test_acc = 100 * correct / total
    print(f"\nTest Loss: {test_loss / len(test_loader):.4f} Acc: {test_acc:.2f}%")

    print("\nPer-class Accuracy:")
    for i in range(num_classes):
        if class_total[i] > 0:
            print(f"Class {test_loader.dataset.index_to_accent[i]}: {100 * class_correct[i] / class_total[i]:.2f}%")
        else:
            print(f"Class {test_loader.dataset.index_to_accent[i]}: N/A")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Conformer for Classification')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_mels', type=int, default=80, help='Number of mel filterbanks')
    parser.add_argument('--encoder_dim', type=int, default=256, help='Encoder dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_attention_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--augment', type=bool, default=False, help='Enable data augmentation')
    parser.add_argument('--model', type=str, default='conformer', choices=['conformer', 'custom_model'], help='Model to use')

    args = parser.parse_args()
    test(args)
