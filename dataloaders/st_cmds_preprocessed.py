import os
import csv
import random
from typing import List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class WavBIRDatasetPreprocessed(Dataset):
    """
    Dataset that reads preprocessed Mel spectrograms (.npy files).
    CSV format: filename, BIR, split
    Feature path: feature_dir/filename.npy
    """

    def __init__(self,
                 root_dir: str,
                 feature_dir: str,
                 csv_filename: str = 'metadata_split.csv',
                 split: str = 'train',
                 augment: bool = False,
                 normalize: bool = True):
        self.root_dir = root_dir
        self.feature_dir = feature_dir
        self.augment = augment
        self.normalize = normalize
        self.split = split.lower()
        
        csv_path = os.path.join(root_dir, csv_filename)
        print(f"Loading dataset from CSV: {csv_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.samples = []
        self.accent_to_index = {}
        self.index_to_accent = []

        # First pass: collect all accents (BIR) to build mapping
        all_accents = set()
        all_rows = []
        
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'filename' not in row or 'BIR' not in row or 'split' not in row:
                    continue
                all_accents.add(row['BIR'])
                all_rows.append(row)
        
        self.index_to_accent = sorted(list(all_accents))
        self.accent_to_index = {accent: idx for idx, accent in enumerate(self.index_to_accent)}

        # Second pass: filter by split and build samples
        for row in all_rows:
            row_split = row['split'].lower()
            if row_split in ['validation', 'valid']:
                row_split = 'val'
            
            target_split = self.split
            if target_split in ['validation', 'valid']:
                target_split = 'val'

            if row_split == target_split:
                filename = row['filename']
                # Construct feature path
                # Assuming filename in CSV is like "file.wav" or just "file"
                # We want "file.npy"
                base_name = os.path.splitext(filename)[0]
                feature_path = os.path.join(self.feature_dir, base_name + '.npy')
                
                self.samples.append({
                    'path': feature_path,
                    'accent': row['BIR'],
                    'accent_index': self.accent_to_index[row['BIR']],
                    'speaker': ''
                })
        
        rng = random.Random(42)
        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        path = sample['path']
        
        try:
            # Load preprocessed features (T, n_mels)
            feats = np.load(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}")

        # Note: Audio-level augmentation (pitch shift) is skipped as we don't have raw audio.

        if self.augment:
            T, n_mels = feats.shape
            # Frequency Masking
            num_masks = 2
            F = 15
            for _ in range(num_masks):
                f = random.randint(0, F)
                f0 = random.randint(0, max(0, n_mels - f))
                feats[:, f0:f0+f] = feats.min()

            # Time Masking
            num_masks = 2
            T_mask = 30
            for _ in range(num_masks):
                t = random.randint(0, T_mask)
                t0 = random.randint(0, max(0, T - t))
                feats[t0:t0+t, :] = feats.min()

        # per-utterance mean-variance normalize
        if self.normalize:
            mu = np.mean(feats, axis=0, keepdims=True)
            std = np.std(feats, axis=0, keepdims=True)
            std[std < 1e-6] = 1.0
            feats = (feats - mu) / std

        return {
            'features': torch.from_numpy(feats).float(), # (T, n_mels)
            'input_length': feats.shape[0],
            'path': path,
            'accent': sample['accent'],
            'accent_index': sample['accent_index'],
            'speaker': '',
            'num_accents': len(self.accent_to_index)
        }

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad a batch of feature sequences (time major) to the max length in the batch."""
    lengths = [item['input_length'] for item in batch]
    max_len = max(lengths)
    n_mels = batch[0]['features'].shape[1]

    inputs = batch[0]['features'].new_zeros((len(batch), max_len, n_mels))

    for i, item in enumerate(batch):
        t = item['features'].shape[0]
        inputs[i, :t, :] = item['features']

    input_lengths = torch.LongTensor(lengths)

    num_accents = batch[0].get('num_accents')
    accent_indices = torch.LongTensor([item['accent_index'] for item in batch])
    accent_onehot = inputs.new_zeros((len(batch), num_accents))
    accent_onehot.scatter_(1, accent_indices.unsqueeze(1), 1.0)

    return {
        'inputs': inputs,
        'input_lengths': input_lengths,
        'paths': [item['path'] for item in batch],
        'speakers': [item['speaker'] for item in batch],
        'accents': [item['accent'] for item in batch],
        'accent_indices': accent_indices,
        'accent_onehot': accent_onehot,
    }

def get_dataloader(root_dir: str,
                   feature_dir: str,
                   batch_size: int = 8,
                   shuffle: Optional[bool] = None,
                   num_workers: int = 0,
                   augment: Optional[bool] = None,
                   **dataset_kwargs) -> DataLoader:
    if augment is None:
        augment = (dataset_kwargs.get('split', 'train') == 'train')
        
    ds = WavBIRDatasetPreprocessed(root_dir, feature_dir, augment=augment, **dataset_kwargs)
    if shuffle is None:
        shuffle = ds.split == 'train'
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
