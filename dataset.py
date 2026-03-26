import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import os
from config import F5Config


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
                         names=["path", "text"])
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

        audio_path = f"{self.data_path}/wavs/{fname}"

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File not found: {audio_path}")

        wav, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.mean(0)  # mono

        mel = self.mel(wav)                    # [n_mel, T]
        mel = torch.log(mel.clamp(min=1e-5))
        mel = mel.T                            # [T, n_mel]

        tokens = torch.tensor(self.tokenize(text), dtype=torch.long)
        return mel, tokens

def collate_fn(batch):
    mels, tokens = zip(*batch)
    n_mel = mels[0].size(1)

    # Crop T_max
    T_max = min(max(m.size(0) for m in mels), F5Config().max_mel_len)
    T_text = max(t.size(0) for t in tokens)

    mel_pad = torch.zeros(len(mels), T_max, n_mel)
    tok_pad = torch.zeros(len(tokens), T_text, dtype=torch.long)
    mask    = torch.zeros(len(mels), T_max, dtype=torch.bool)

    for i, (m, t) in enumerate(zip(mels, tokens)):
        L = min(m.size(0), T_max)
        mel_pad[i, :L] = m[:L]
        mask[i, :L] = True
        tok_pad[i, :t.size(0)] = t

    return mel_pad, tok_pad, mask
