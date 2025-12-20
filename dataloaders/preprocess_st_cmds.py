import os
import csv
import argparse
import numpy as np
import librosa
from tqdm import tqdm
import concurrent.futures

# Parameters from dataloaders/st_cmds.py
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160

def _load_audio_librosa(path, sr):
    try:
        # Load audio with librosa
        y, sr0 = librosa.load(path, sr=sr)
        return y, sr
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None

def _extract_log_mel(y, sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
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

def process_file(row, root_dir, output_dir):
    filename = row['filename']
    # Construct source path based on st_cmds.py logic
    audio_path = os.path.join(root_dir, 'ST-CMDS-20170001_1-OS', filename)
    
    # Construct target path
    # Save as .npy with the same basename
    save_name = os.path.splitext(filename)[0] + '.npy'
    save_path = os.path.join(output_dir, save_name)
    
    # Skip if already exists
    if os.path.exists(save_path):
        return
        
    # Check if audio file exists
    if not os.path.exists(audio_path):
        # Try appending .wav if it doesn't exist (common issue)
        if os.path.exists(audio_path + '.wav'):
            audio_path += '.wav'
        else:
            # print(f"File not found: {audio_path}")
            return

    # Load and process
    y, sr = _load_audio_librosa(audio_path, SAMPLE_RATE)
    if y is None:
        return

    feats = _extract_log_mel(y, sr)
    
    # Save features
    np.save(save_path, feats)

def main():
    parser = argparse.ArgumentParser(description="Preprocess ST-CMDS dataset to Mel spectrograms")
    parser.add_argument('--root_dir', type=str, default='/data_extend/fcy/dl/datasets/', help="Root directory of the dataset containing metadata_split.csv")
    parser.add_argument('--csv_filename', type=str, default='metadata_split.csv', help="Metadata CSV filename")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of parallel workers")
    
    args = parser.parse_args()
    output_dir = os.path.join(args.root_dir, 'features')
    
    csv_path = os.path.join(args.root_dir, args.csv_filename)
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'filename' in row:
                rows.append(row)
                
    print(f"Found {len(rows)} files to process.")
    print(f"Saving features to {output_dir}")
    
    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_file, row, args.root_dir, output_dir) for row in rows]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(rows), desc="Processing"):
            pass
            
    print("Preprocessing complete.")

if __name__ == '__main__':
    main()
