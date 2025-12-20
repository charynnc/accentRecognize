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
            self.initials = ['<pad>', '<unk>']
            self.finals = ['<pad>', '<unk>']
            self.tones = ['<pad>', '<unk>', '0', '1', '2', '3', '4']
            
            init_vocab_path = os.path.join(self.root_dir, 'initials_vocab.txt')
            final_vocab_path = os.path.join(self.root_dir, 'finals_vocab.txt')
            
            if os.path.exists(init_vocab_path) and os.path.exists(final_vocab_path):
                print(f"Loading vocabs from {init_vocab_path} and {final_vocab_path}")
                with open(init_vocab_path, 'r', encoding='utf-8') as f:
                    self.initials = [line.strip() for line in f]
                with open(final_vocab_path, 'r', encoding='utf-8') as f:
                    self.finals = [line.strip() for line in f]
            else:
                print("Building decomposed vocabs from ALL data...")
                all_initials = set()
                all_finals = set()
                
                for r in all_rows:
                    fname = r['filename']
                    base = os.path.splitext(fname)[0]
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
                                inis = [p[0] for p in pinyin(text, style=Style.INITIALS, errors='ignore', strict=False)]
                                fins = [p[0] for p in pinyin(text, style=Style.FINALS, errors='ignore', strict=False)]
                                all_initials.update(inis)
                                all_finals.update(fins)
                        except Exception:
                            pass
                
                for x in sorted(list(all_initials)):
                    if x == "": x = "_" # Handle empty initial
                    if x not in self.initials: self.initials.append(x)
                for x in sorted(list(all_finals)):
                    if x not in self.finals: self.finals.append(x)
                
                # Save
                try:
                    with open(init_vocab_path, 'w', encoding='utf-8') as f:
                        for x in self.initials: f.write(x + '\n')
                    with open(final_vocab_path, 'w', encoding='utf-8') as f:
                        for x in self.finals: f.write(x + '\n')
                    print(f"Built vocabs: Initials={len(self.initials)}, Finals={len(self.finals)}")
                except Exception as e:
                    print(f"Could not save vocab file: {e}")

            self.initial_to_idx = {x: i for i, x in enumerate(self.initials)}
            self.final_to_idx = {x: i for i, x in enumerate(self.finals)}
            self.tone_to_idx = {x: i for i, x in enumerate(self.tones)}

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

        initial_indices = []
        final_indices = []
        tone_indices = []
        
        if self.use_pinyin and sample['txt_path'] and os.path.exists(sample['txt_path']):
            try:
                with open(sample['txt_path'], 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                if pinyin:
                    # Extract components
                    inis = [p[0] for p in pinyin(text, style=Style.INITIALS, errors='ignore', strict=False)]
                    fins = [p[0] for p in pinyin(text, style=Style.FINALS, errors='ignore', strict=False)]
                    # Extract tones from TONE3 style
                    pys_tone = [p[0] for p in pinyin(text, style=Style.TONE3, errors='ignore')]
                    
                    for ini, fin, py_tone in zip(inis, fins, pys_tone):
                        if ini == "": ini = "_"
                        
                        tone = "0"
                        if py_tone[-1].isdigit():
                            tone = py_tone[-1]
                            
                        initial_indices.append(self.initial_to_idx.get(ini, self.initial_to_idx['<unk>']))
                        final_indices.append(self.final_to_idx.get(fin, self.final_to_idx['<unk>']))
                        tone_indices.append(self.tone_to_idx.get(tone, self.tone_to_idx['<unk>']))
                        
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
            'initial_indices': torch.LongTensor(initial_indices),
            'final_indices': torch.LongTensor(final_indices),
            'tone_indices': torch.LongTensor(tone_indices),
            'pinyin_length': len(initial_indices)
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

    if 'initial_indices' in batch[0]:
        pinyin_lengths = [item['pinyin_length'] for item in batch]
        max_pinyin_len = max(pinyin_lengths) if pinyin_lengths else 0
        
        initial_inputs = torch.zeros((len(batch), max_pinyin_len), dtype=torch.long)
        final_inputs = torch.zeros((len(batch), max_pinyin_len), dtype=torch.long)
        tone_inputs = torch.zeros((len(batch), max_pinyin_len), dtype=torch.long)
        
        for i, item in enumerate(batch):
            l = item['pinyin_length']
            if l > 0:
                initial_inputs[i, :l] = item['initial_indices']
                final_inputs[i, :l] = item['final_indices']
                tone_inputs[i, :l] = item['tone_indices']
        
        result['initial_inputs'] = initial_inputs
        result['final_inputs'] = final_inputs
        result['tone_inputs'] = tone_inputs
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
