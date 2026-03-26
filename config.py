from dataclasses import dataclass

@dataclass
class F5Config:
    # Model architecture
    dim: int = 1024
    depth: int = 22
    heads: int = 16
    ff_mult: int = 2
    text_dim: int = 512
    conv_layers: int = 4
    pe_attn_head: int = 1

    # Mel
    n_mel_channels: int = 100
    hop_length: int = 256
    sample_rate: int = 24000

    # Training
    learning_rate: float = 7.5e-5
    warmup_steps: int = 20000
    total_steps: int = 1_000_000
    grad_accum: int = 8          # เพิ่ม accum
    max_grad_norm: float = 1.0
    batch_size_per_gpu: int = 4

    max_mel_len: int = 1200

    # Paths
    vocab_path: str = "/content/F5-TTS-E3/dataset/vocab.txt"
    data_path: str = "/content/F5-TTS-E3/dataset"
    output_dir: str = "./ckpts/f5tts"
    resume_ckpt: str = ""
    num_workers: int = 2