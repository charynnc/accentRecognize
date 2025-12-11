import os
import csv
import random
from typing import List, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa

def _load_audio_librosa(path: str, sr: int):
    y, sr0 = librosa.load(path, sr=sr)
    return y, sr

def _extract_log_mel(y: np.ndarray,
                     sr: int,
                     n_mels: int = 80,
                     n_fft: int = 400,
                     hop_length: int = 160) -> np.ndarray:

    mel = librosa.feature.melspectrogram(y=y,
                                         sr=sr,
                                         n_fft=n_fft,
                                         hop_length=hop_length,
                                         n_mels=n_mels,
                                         power=2.0)
    # convert to log scale (dB). shape: (n_mels, T)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # transpose to (T, n_mels)
    return log_mel.T

class WavBIRDataset(Dataset):
    """
    Dataset that reads from a CSV file.
    CSV format: filename, BIR, split
    Audio path: root_dir/recordings/filename.wav
    """

    def __init__(self,
                 root_dir: str,
                 csv_filename: str = 'metadata_split.csv',
                 sample_rate: int = 16000,
                 n_mels: int = 80,
                 n_fft: int = 400,
                 hop_length: int = 160,
                 split: str = 'train',
                 augment: bool = False,
                 normalize: bool = True):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
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
                # Ensure required columns exist
                if 'filename' not in row or 'BIR' not in row or 'split' not in row:
                    print(f"Skipping invalid row: {row}")
                    continue
                
                all_accents.add(row['BIR'])
                all_rows.append(row)
        
        print(f"Found {len(all_rows)} valid rows in CSV.")
        print(f"Found {len(all_accents)} unique BIRs.")

        self.index_to_accent = sorted(list(all_accents))
        self.accent_to_index = {accent: idx for idx, accent in enumerate(self.index_to_accent)}

        # Second pass: filter by split and build samples
        for row in all_rows:
            row_split = row['split'].lower()
            
            # Normalize split names
            if row_split in ['validation', 'valid']:
                row_split = 'val'
            
            target_split = self.split
            if target_split in ['validation', 'valid']:
                target_split = 'val'

            if row_split == target_split:
                filename = row['filename']
                # Ensure filename ends with .wav
                # if filename.lower().endswith('.mp3'):
                #     filename = filename[:-4] + '.wav'
                # elif not filename.lower().endswith('.wav'):
                #     filename += '.wav'
                
                audio_path = os.path.join(root_dir, 'ST-CMDS-20170001_1-OS', filename)
                self.samples.append({
                    'path': audio_path,
                    'accent': row['BIR'],
                    'accent_index': self.accent_to_index[row['BIR']],
                    'speaker': ''
                })
        
        # Shuffle samples to avoid ordering by BIR in CSV
        # Use a fixed seed to ensure reproducibility across runs
        rng = random.Random(42)
        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        path = sample['path']
        
        try:
            y, sr = _load_audio_librosa(path, self.sample_rate)
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}")

        if self.augment:
            if random.random() < 0.5:
                n_steps = random.uniform(-2, 2)
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

        feats = _extract_log_mel(y, sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)

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
    """Pad a batch of feature sequences (time major) to the max length in the batch.

    Returns dict with keys:
      - 'inputs': FloatTensor (B, T_max, n_mels)
      - 'input_lengths': LongTensor (B,)
      - 'paths': list[str]
      - 'speakers': list[str]
      - 'accents': list[str]
    """
    # batch is list of dicts
    lengths = [item['input_length'] for item in batch]
    max_len = max(lengths)
    n_mels = batch[0]['features'].shape[1]

    inputs = batch[0]['features'].new_zeros((len(batch), max_len, n_mels))

    for i, item in enumerate(batch):
        t = item['features'].shape[0]
        inputs[i, :t, :] = item['features']

    input_lengths = torch.LongTensor(lengths)

    num_accents = batch[0].get('num_accents')
    if num_accents is None:
        raise ValueError("Batch items missing 'num_accents'. Ensure WavBIRDataset returns accent metadata.")

    accent_indices = torch.LongTensor([item['accent_index'] for item in batch])
    accent_onehot = inputs.new_zeros((len(batch), num_accents))
    accent_onehot.scatter_(1, accent_indices.unsqueeze(1), 1.0)

    for item in batch:
        item.pop('num_accents', None)

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
                   batch_size: int = 8,
                   shuffle: Optional[bool] = None,
                   num_workers: int = 0,
                   augment: Optional[bool] = None,
                   **dataset_kwargs) -> DataLoader:
    if augment is None:
        # Default to augment only for training split
        augment = (dataset_kwargs.get('split', 'train') == 'train')
        
    ds = WavBIRDataset(root_dir, augment=augment, **dataset_kwargs)
    if shuffle is None:
        shuffle = ds.split == 'train'
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
