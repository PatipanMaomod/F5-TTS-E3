import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
from pathlib import Path


class VoiceDataset(Dataset):
    def __init__(self, data_path, vocab_path, sample_rate=24000, n_mel=100):
        self.data_path   = Path(data_path)
        self.sample_rate = sample_rate
        self.n_mel       = n_mel

        # โหลด vocab
        with open(vocab_path, encoding="utf-8") as f:
            chars = [l.strip() for l in f if l.strip()]
        self.char2id = {c: i+1 for i, c in enumerate(chars)}  # 0 = pad

        # โหลด metadata
        df = pd.read_csv(self.data_path / "metadata.csv", sep="|",
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
        audio_path, text = self.items[idx]
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

    T_max    = max(m.size(0) for m in mels)
    T_text   = max(t.size(0) for t in tokens)
    n_mel    = mels[0].size(1)

    mel_pad = torch.zeros(len(mels), T_max, n_mel)
    tok_pad = torch.zeros(len(tokens), T_text, dtype=torch.long)
    mask    = torch.zeros(len(mels), T_max, dtype=torch.bool)

    for i, (m, t) in enumerate(zip(mels, tokens)):
        mel_pad[i, :m.size(0)] = m
        tok_pad[i, :t.size(0)] = t
        mask[i, :m.size(0)]    = True

    return mel_pad, tok_pad, mask