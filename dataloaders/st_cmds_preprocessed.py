import os
import csv
import random
from typing import List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
try:
    from pypinyin import pinyin, Style
except ImportError:
    print("pypinyin not found. Please install it using `pip install pypinyin`")
    pinyin = None


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
                 normalize: bool = True,
                 use_pinyin: bool = False):
        self.root_dir = root_dir
        self.feature_dir = feature_dir
        self.augment = augment
        self.normalize = normalize
        self.split = split.lower()
        self.use_pinyin = use_pinyin
        
        csv_path = os.path.join(root_dir, csv_filename)
        print(f"Loading dataset from CSV: {csv_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.samples = []
        self.accent_to_index = {}
        self.index_to_accent = []
        self.pinyin_to_index = {'<pad>': 0, '<unk>': 1}
        self.index_to_pinyin = ['<pad>', '<unk>']

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
                
                txt_path = None
                if self.use_pinyin:
                    # Try multiple locations
                    possible_paths = [
                        os.path.join(self.root_dir, 'recordings', base_name + '.txt'),
                        os.path.join(self.root_dir, base_name + '.txt'),
                        os.path.join(self.root_dir, 'ST-CMDS-20170001_1-OS', base_name + '.txt')
                    ]
                    for p in possible_paths:
                        if os.path.exists(p):
                            txt_path = p
                            break

                self.samples.append({
                    'path': feature_path,
                    'accent': row['BIR'],
                    'accent_index': self.accent_to_index[row['BIR']],
                    'speaker': '',
                    'txt_path': txt_path
                })
        
        if self.use_pinyin:
            vocab_path = os.path.join(self.root_dir, 'pinyin_vocab.txt')
            if os.path.exists(vocab_path):
                print(f"Loading Pinyin vocab from {vocab_path}")
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    self.index_to_pinyin = [line.strip() for line in f]
                self.pinyin_to_index = {py: idx for idx, py in enumerate(self.index_to_pinyin)}
            else:
                print("Pinyin vocab not found. Building from ALL data in CSV...")
                all_pinyins = set()
                # Scan ALL rows in CSV to ensure vocab is complete regardless of current split
                for r in all_rows:
                    # Construct path for every file in CSV
                    fname = r['filename']
                    base = os.path.splitext(fname)[0]
                    # Try multiple locations
                    possible_paths = [
                        os.path.join(self.root_dir, 'recordings', base + '.txt'),
                        os.path.join(self.root_dir, base + '.txt'),
                        os.path.join(self.root_dir, 'ST-CMDS-20170001_1-OS', base + '.txt')
                    ]
                    t_path = None
                    for p in possible_paths:
                        if os.path.exists(p):
                            t_path = p
                            break
                    
                    if t_path:
                        try:
                            with open(t_path, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                            if pinyin:
                                pys = [p[0] for p in pinyin(text, style=Style.TONE3, errors='ignore')]
                                all_pinyins.update(pys)
                        except Exception as e:
                            pass # Ignore errors during vocab build
                
                # Sort and build index
                sorted_pinyins = sorted(list(all_pinyins))
                for py in sorted_pinyins:
                    self.pinyin_to_index[py] = len(self.index_to_pinyin)
                    self.index_to_pinyin.append(py)
                
                print(f"Built Pinyin vocab (size: {len(self.index_to_pinyin)})")
                # Save it for next time
                try:
                    with open(vocab_path, 'w', encoding='utf-8') as f:
                        for py in self.index_to_pinyin:
                            f.write(py + '\n')
                    print(f"Saved Pinyin vocab to {vocab_path}")
                except Exception as e:
                    print(f"Could not save vocab file: {e}")

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

        pinyin_indices = []
        if self.use_pinyin and sample['txt_path'] and os.path.exists(sample['txt_path']):
            try:
                with open(sample['txt_path'], 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                if pinyin:
                    pys = [p[0] for p in pinyin(text, style=Style.TONE3, errors='ignore')]
                    pinyin_indices = [self.pinyin_to_index.get(py, self.pinyin_to_index['<unk>']) for py in pys]
            except Exception as e:
                print(f"Error reading {sample['txt_path']} in __getitem__: {e}")

        return {
            'features': torch.from_numpy(feats).float(), # (T, n_mels)
            'input_length': feats.shape[0],
            'path': path,
            'accent': sample['accent'],
            'accent_index': sample['accent_index'],
            'speaker': '',
            'num_accents': len(self.accent_to_index),
            'pinyin_indices': torch.LongTensor(pinyin_indices),
            'pinyin_length': len(pinyin_indices)
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

    result = {
        'inputs': inputs,
        'input_lengths': input_lengths,
        'paths': [item['path'] for item in batch],
        'speakers': [item['speaker'] for item in batch],
        'accents': [item['accent'] for item in batch],
        'accent_indices': accent_indices,
        'accent_onehot': accent_onehot,
    }

    if 'pinyin_indices' in batch[0]:
        pinyin_lengths = [item['pinyin_length'] for item in batch]
        max_pinyin_len = max(pinyin_lengths) if pinyin_lengths else 0
        # Pad with 0 (which is <pad>)
        pinyin_inputs = torch.zeros((len(batch), max_pinyin_len), dtype=torch.long)
        for i, item in enumerate(batch):
            l = item['pinyin_length']
            if l > 0:
                pinyin_inputs[i, :l] = item['pinyin_indices']
        
        result['pinyin_inputs'] = pinyin_inputs
        result['pinyin_lengths'] = torch.LongTensor(pinyin_lengths)
    
    return result

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
