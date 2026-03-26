from dataclasses import dataclass

@dataclass
class F5Config:
    # Model architecture (ต้องตรงกับ pretrained)
    dim: int = 1024
    depth: int = 22
    heads: int = 16
    ff_mult: int = 2
    text_dim: int = 512
    conv_layers: int = 4          # CNN blocks ก่อน DiT
    pe_attn_head: int = 1

    # Mel
    n_mel_channels: int = 100
    hop_length: int = 256
    sample_rate: int = 24000
    mel_spec_type: str = "vocos"  # vocos | bigvgan

    # Training
    learning_rate: float = 1e-5
    warmup_steps: int = 1000
    total_steps: int = 50000
    grad_accum: int = 2
    max_grad_norm: float = 1.0
    batch_size: int = 4800        # frames
    num_workers: int = 4

    # Paths
    pretrain_path: str = "./pretrained"
    vocab_path: str = "./dataset/vocab.txt"
    data_path: str = "./dataset"
    output_dir: str = "./ckpts/f5tts"