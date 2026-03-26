import torch
import torchaudio
import gradio as gr
from model import F5TTS
from config import F5Config

# ── โหลดโมเดล ──
cfg = F5Config()
vocab_path = "/content/F5-TTS-E3/dataset/vocab.txt"
ckpt_path = "output/step_100.pt"  # เปลี่ยนเป็นไฟล์ checkpoint ของคุณ

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# โหลด vocab
with open(vocab_path, encoding="utf-8") as f:
    chars = [l.strip() for l in f if l.strip()]
char2id = {c: i + 1 for i, c in enumerate(chars)}

# โหลดโมเดล
vocab_size = len(char2id)
model = F5TTS(cfg, vocab_size=vocab_size).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()


# ── ฟังก์ชัน inference ──
def tts_infer(text):
    tokens = torch.tensor([char2id.get(c, 0) for c in text], dtype=torch.long).unsqueeze(0).to(device)
    # สร้าง mask สำหรับ batch size = 1
    mask = torch.ones(1, tokens.size(1), dtype=torch.bool).to(device)

    # ทำ dummy mel input (สำหรับ flow matching TTS ต้องมี mel shape)
    T = 10  # frame length สั้น ๆ สำหรับ inference
    mel_dim = cfg.n_mel_channels
    mel_input = torch.randn(1, T, mel_dim).to(device)
    t_vec = torch.rand(1, device=device)

    with torch.no_grad():
        pred = model(mel_input, t_vec, tokens, mask)  # [1, T, n_mel]

    # แปลงกลับเป็น waveform (ง่าย ๆ ใช้ Griffin-Lim)
    mel_spec = pred[0].T.cpu()  # [n_mel, T]
    mel_spec = torch.exp(mel_spec)  # undo log
    waveform = torchaudio.functional.griffinlim(
        mel_spec,
        n_iter=32,
        n_fft=1024,
        hop_length=cfg.hop_length,
        win_length=1024
    )
    return waveform.cpu(), cfg.sample_rate


# ── สร้าง interface ──
demo = gr.Interface(
    fn=tts_infer,
    inputs=gr.Textbox(label="ข้อความ (Thai/Isan)"),
    outputs=gr.Audio(label="เสียงออก"),
    live=False,
)

demo.launch()