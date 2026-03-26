import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import os
from config import F5Config
import random


class VoiceDataset(Dataset):
    def __init__(self, data_path, vocab_path, sample_rate=24000, n_mel=100):
        self.data_path   = str(data_path)
        self.sample_rate = sample_rate
        self.n_mel       = n_mel

        # โหลด vocab
        with open(vocab_path, encoding="utf-8") as f:
            chars = [l.strip() for l in f if l.strip()]
        self.char2id = {c: i+1 for i, c in enumerate(chars)}  # 0 = pad
        self.vocab_size = len(self.char2id)


        # โหลด metadata
        df = pd.read_csv(f'{self.data_path}/metadata.csv', sep="|",
                         names=["path", "text"], skiprows=1)
        self.items = df.values.tolist()

        # Mel filterbank
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mel,
            n_fft=1024, hop_length=256, win_length=1024,
        )

    def tokenize(self, text):
        return [self.char2id.get(c, 0) for c in text]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, text = self.items[idx]
        if not isinstance(text, str):
            text = ""

        # โหลด mel จาก cache แทน compute ใหม่
        cache_path = f"{self.data_path}/mel_cache/{fname}.pt"
        if os.path.exists(cache_path):
            mel = torch.load(cache_path, weights_only=True)
        else:
            # fallback คำนวณสดถ้าไม่มี cache
            audio_path = f"{self.data_path}/wavs/{fname}"
            wav, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            mel = self.mel(wav)
            mel = torch.log(mel.clamp(min=1e-5)).squeeze(0).T

        tokens = torch.tensor(self.tokenize(text), dtype=torch.long)
        return mel, tokens

def collate_fn(batch,max_frames):
    mels, tokens = zip(*batch)
    n_mel = mels[0].size(1)

    # เตรียม tensor สำหรับ mel, mask, tokens
    mel_pad = torch.zeros(len(mels), max_frames, n_mel)
    mask = torch.zeros(len(mels), max_frames, dtype=torch.bool)
    T_text = max(t.size(0) for t in tokens)
    tok_pad = torch.zeros(len(tokens), T_text, dtype=torch.long)

    for i, (m, t) in enumerate(zip(mels, tokens)):
        L = m.size(0)
        if L > max_frames:
            # random crop ถ้าเสียงยาวกว่า 1200 frames
            start = torch.randint(0, L - max_frames + 1, (1,)).item()
            mel_pad[i] = m[start:start + max_frames]
            mask[i] = True
        else:
            # pad ถ้าเสียงสั้น
            mel_pad[i, :L] = m
            mask[i, :L] = True

        # pad token
        tok_pad[i, :t.size(0)] = t

    return mel_pad, tok_pad, mask






def augment_audio(wav, sample_rate=24000, pitch_range=2, speed_range=0.05):
    """
    wav: Tensor [1, T] หรือ [T]
    pitch_range: สูงสุด ±จำนวน semitone
    speed_range: ±% ของความเร็ว
    """
    # 1. Random pitch shift
    semitone_shift = random.uniform(-pitch_range, pitch_range)
    if semitone_shift != 0:
        wav = T.Resample(orig_freq=sample_rate, new_freq=int(sample_rate * 2 ** (semitone_shift/12)))(wav)

    # 2. Random speed change
    speed_factor = random.uniform(1-speed_range, 1+speed_range)
    if speed_factor != 1:
        wav = T.Resample(orig_freq=sample_rate, new_freq=int(sample_rate*speed_factor))(wav)

    # 3. ปรับให้กลับมาที่ sample_rate เดิม
    wav = T.Resample(orig_freq=wav.shape[-1], new_freq=sample_rate)(wav) if wav.shape[-1] != sample_rate else wav

    return wav