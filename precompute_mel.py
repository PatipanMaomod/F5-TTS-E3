# precompute_mel.py — รันครั้งเดียวก่อนเทรน
import torch, torchaudio, pandas as pd, os
from pathlib import Path

DATA_PATH   = "./data/train"
SAMPLE_RATE = 24000
N_MEL       = 100
CACHE_DIR   = f"{DATA_PATH}/mel_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_mels=N_MEL,
    n_fft=1024, hop_length=256, win_length=1024,
)

df = pd.read_csv(f"{DATA_PATH}/metadata.csv", sep="|", names=["path","text"])
for i, row in df.iterrows():
    cache_path = f"{CACHE_DIR}/{row['path']}.pt"
    if os.path.exists(cache_path):
        continue
    wav_path = f"{DATA_PATH}/wavs/{row['path']}"
    wav, sr  = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    mel = mel_transform(wav)
    mel = torch.log(mel.clamp(min=1e-5)).squeeze(0).T  # [T, n_mel]
    torch.save(mel, cache_path)
    if i % 500 == 0:
        print(f"{i}/{len(df)}")

print("Done!")