import torch
import torch.nn.functional as F
import gradio as gr
import numpy as np
import torchaudio

from model import F5TTS
from config import F5Config
from vocos import Vocos


# ─────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
cfg = F5Config()
cfg.n_mel_channels = 100   # ต้อง match vocos

vocab_path = "/content/F5-TTS-E3/dataset/vocab.txt"
ckpt_path  = "/content/F5-TTS-E3/ckpts/step_00005000.pt"


# ─────────────────────────────────────────
# VOCAB
# ─────────────────────────────────────────
with open(vocab_path, encoding="utf-8") as f:
    chars = [l.strip() for l in f if l.strip()]

char2id = {c: i + 1 for i, c in enumerate(chars)}


# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
model = F5TTS(cfg, vocab_size=len(char2id)).to(device)

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

state_dict = ckpt["model"]
state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict, strict=True)
model.eval()


# ─────────────────────────────────────────
# VOCODER
# ─────────────────────────────────────────
print("🚀 loading vocoder...")
vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)


# ─────────────────────────────────────────
# FLOW MATCHING (ดีขึ้น)
# ─────────────────────────────────────────
def generate_mel(text, steps=60):
    tokens = torch.tensor(
        [char2id.get(c, 0) for c in text],
        dtype=torch.long
    ).unsqueeze(0).to(device)

    B = 1
    T = max(100, len(text) * 12)

    mel = torch.randn(B, T, cfg.n_mel_channels).to(device)
    mask = torch.ones(B, T, dtype=torch.bool).to(device)

    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full((B,), i / steps, device=device)

        with torch.no_grad():
            pred = model(mel, t, tokens, mask)

        # Euler integration (stable)
        mel = mel + dt * (pred - mel)

    return mel


# ─────────────────────────────────────────
# MEL → WAV
# ─────────────────────────────────────────
def mel_to_waveform(mel):
    mel = torch.clamp(mel, -6, 6)
    mel = mel.transpose(1, 2)

    with torch.no_grad():
        wav = vocoder.decode(mel)

    wav = wav.squeeze()

    # normalize
    wav = wav - wav.mean()
    wav = wav / (wav.abs().max() + 1e-6)

    wav = torchaudio.functional.resample(
        wav,
        orig_freq=24000,
        new_freq=16000
    )

    return wav.cpu().numpy().astype("float32")


# ─────────────────────────────────────────
# TTS
# ─────────────────────────────────────────
def tts(text):
    if not text.strip():
        return None

    mel = generate_mel(text)
    wav = mel_to_waveform(mel)

    return (16000, wav)


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
demo = gr.Interface(
    fn=tts,
    inputs=gr.Textbox(label="ข้อความ (Thai / Isan)"),
    outputs=gr.Audio(label="เสียง"),
    title="Custom F5-TTS + Vocos",
)

demo.launch(share=True)