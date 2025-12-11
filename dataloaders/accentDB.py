import os
import random
from typing import List, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa


def _find_wav_files(root_dir: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith('.wav'):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


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


def _infer_accent_speaker(path: str, root_dir: str) -> Tuple[str, str]:
    """Return (accent, speaker) inferred from path relative to root."""
    rel = os.path.relpath(path, root_dir)
    parts = rel.split(os.sep)
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    if len(parts) == 2:
        return '', parts[0]
    return '', ''


def _split_counts(num_items: int, ratios: Sequence[float]) -> Tuple[int, int, int]:
    if num_items < 0:
        raise ValueError('num_items must be non-negative')
    if len(ratios) != 3:
        raise ValueError('ratios must have exactly three elements (train, val, test)')
    total = float(sum(ratios))
    if total <= 0:
        raise ValueError('ratios must sum to a positive value')
    norm = [r / total for r in ratios]
    train = int(num_items * norm[0])
    val = int(num_items * norm[1])
    test = max(0, num_items - train - val)
    # ensure we do not lose samples due to rounding by pushing leftovers to test split
    if train + val + test < num_items:
        test += num_items - (train + val + test)
    return train, val, test


class SpeechDataset(Dataset):
    """A simple dataset that walks a root directory and loads WAV files.

    Expected directory layout (example):
    data/<accent>/<speaker>/*.wav

    Each item returned is a dict with keys:
      - 'features': FloatTensor, shape (T, n_mels)
      - 'input_length': int (number of frames T)
      - 'path': str (original wav path)
      - 'speaker': str (speaker folder name) if available
      - 'accent': str (accent/language folder name) if available
    """

    def __init__(self,
                 root_dir: str,
                 sample_rate: int = 16000,
                 n_mels: int = 80,
                 n_fft: int = 400,
                 hop_length: int = 160,
                 normalize: bool = True,
                 split: str = 'train',
                 split_ratio: Sequence[float] = (0.8, 0.1, 0.1),
                 shuffle_within_speaker: bool = True,
                 split_seed: int = 13,
                 augment: bool = False):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalize = normalize
        self.augment = augment
        self.split = split.lower()
        if self.split not in {'train', 'val', 'valid', 'validation', 'test'}:
            raise ValueError(f"Unsupported split '{split}'. Use train/val/test")

        self.accent_to_index: Dict[str, int] = {}
        self.index_to_accent: List[str] = []

        self.samples = self._build_split_samples(split_ratio, shuffle_within_speaker, split_seed)

    def _build_split_samples(self,
                             split_ratio: Sequence[float],
                             shuffle_within_speaker: bool,
                             split_seed: int) -> List[Dict[str, str]]:
        grouped: Dict[Tuple[str, str], List[str]] = {}
        for path in _find_wav_files(self.root_dir):
            accent, speaker = _infer_accent_speaker(path, self.root_dir)
            grouped.setdefault((accent, speaker), []).append(path)

        unique_accents = sorted({accent for accent, _ in grouped.keys()})
        if not unique_accents:
            unique_accents = ['']
        self.index_to_accent = unique_accents
        self.accent_to_index = {accent: idx for idx, accent in enumerate(unique_accents)}

        rng = random.Random(split_seed)
        samples: List[Dict[str, str]] = []
        target_split = 'val' if self.split in {'val', 'valid', 'validation'} else self.split

        for (accent, speaker), paths in grouped.items():
            paths.sort()
            if shuffle_within_speaker:
                rng.shuffle(paths)
            train_n, val_n, test_n = _split_counts(len(paths), split_ratio)
            idx_train_end = train_n
            idx_val_end = train_n + val_n

            if target_split == 'train':
                selected = paths[:idx_train_end]
            elif target_split == 'val':
                selected = paths[idx_train_end:idx_val_end]
            else:  # test
                selected = paths[idx_val_end:idx_val_end + test_n]

            for path in selected:
                samples.append({
                    'path': path,
                    'accent': accent,
                    'speaker': speaker,
                    'accent_index': self.accent_to_index[accent],
                })

        # keep deterministic ordering for reproducibility
        samples.sort(key=lambda x: (x['accent'], x['speaker'], x['path']))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        path = sample['path']
        # load
        try:
            y, sr = _load_audio_librosa(path, self.sample_rate)
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}")

        if self.augment:
            # Pitch shift (randomly between -2 and 2 semitones)
            # Note: This is slow. If training is too slow, consider disabling or pre-processing.
            if random.random() < 0.5:
                n_steps = random.uniform(-2, 2)
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

        feats = _extract_log_mel(y, sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)

        if self.augment:
            # SpecAugment (Frequency Masking & Time Masking)
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

        feats = torch.from_numpy(feats).float()

        return {
            'features': feats,
            'input_length': feats.shape[0],
            'path': path,
            'speaker': sample['speaker'],
            'accent': sample['accent'],
            'accent_index': sample['accent_index'],
            'num_accents': len(self.accent_to_index),
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
        raise ValueError("Batch items missing 'num_accents'. Ensure SpeechDataset returns accent metadata.")

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
        
    ds = SpeechDataset(root_dir, augment=augment, **dataset_kwargs)
    if shuffle is None:
        shuffle = ds.split == 'train'
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

__all__ = ['SpeechDataset', 'collate_fn', 'get_dataloader']
