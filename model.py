import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat


# ─────────────────────────────────────────
# 1. Rotary Positional Embedding
# ─────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q, k, cos, sin):
    # q, k: [B, heads, T, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# ─────────────────────────────────────────
# 2. ConvNeXt-V2 Block (CNN part ใน F5-TTS)
# ─────────────────────────────────────────
class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt V2 block ที่ F5-TTS ใช้เป็น 'conv_layers'
    ก่อนส่งเข้า DiT transformer
    """
    def __init__(self, dim, mult=2, kernel_size=7):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim          # depthwise
        )
        self.norm = nn.LayerNorm(dim)
        inner_dim = int(dim * mult)
        self.pw1 = nn.Linear(dim, inner_dim)
        self.pw2 = nn.Linear(inner_dim, dim)
        # GRN (Global Response Normalization) — V2 เพิ่มมา
        self.grn_gamma = nn.Parameter(torch.zeros(inner_dim))
        self.grn_beta  = nn.Parameter(torch.zeros(inner_dim))

    def _grn(self, x):
        # x: [B, T, inner_dim]
        gx = torch.norm(x, p=2, dim=1, keepdim=True)        # [B, 1, C]
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.grn_gamma * (x * nx) + self.grn_beta + x

    def forward(self, x):
        # x: [B, T, dim]
        residual = x
        # depthwise conv — ต้อง transpose ก่อน
        x_t = rearrange(x, "b t c -> b c t")
        x_t = self.dw_conv(x_t)
        x   = rearrange(x_t, "b c t -> b t c")
        x   = self.norm(x)
        x   = self.pw1(x)
        x   = F.gelu(x)
        x   = self._grn(x)
        x   = self.pw2(x)
        return x + residual


# ─────────────────────────────────────────
# 3. Adaptive LayerNorm (AdaLN) — DiT condition
# ─────────────────────────────────────────
class AdaLayerNorm(nn.Module):
    """
    Modulate LayerNorm ด้วย time embedding (สำหรับ diffusion step)
    scale, shift มาจาก MLP(time_emb)
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(dim, dim * 2)

    def forward(self, x, cond):
        # cond: [B, dim]
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift


# ─────────────────────────────────────────
# 4. DiT Attention Block
# ─────────────────────────────────────────
class DiTAttention(nn.Module):
    def __init__(self, dim, heads, pe_attn_head=1):
        super().__init__()
        assert dim % heads == 0
        self.heads     = heads
        self.head_dim  = dim // heads
        self.pe_heads  = pe_attn_head   # กี่ head ที่ใช้ RoPE

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.rope   = RotaryEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        qkv = self.to_qkv(x)                          # [B, T, 3*dim]
        q, k, v = rearrange(qkv, "b t (three h d) -> three b h t d",
                             three=3, h=self.heads).unbind(0)

        # Apply RoPE เฉพาะ pe_attn_head หัวแรก
        cos, sin = self.rope(T, x.device)
        q[:, :self.pe_heads], k[:, :self.pe_heads] = apply_rotary(
            q[:, :self.pe_heads], k[:, :self.pe_heads], cos, sin
        )

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn  = torch.einsum("bhid,bhjd->bhij", q, k) * scale

        if mask is not None:
            # mask: [B, T], True = valid
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
            attn = attn.masked_fill(~attn_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out  = torch.einsum("bhij,bhjd->bhid", attn, v)
        out  = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


# ─────────────────────────────────────────
# 5. DiT Block (Attention + FFN + AdaLN)
# ─────────────────────────────────────────
class DiTBlock(nn.Module):
    def __init__(self, dim, heads, ff_mult=2, pe_attn_head=1):
        super().__init__()
        self.norm1 = AdaLayerNorm(dim)
        self.attn  = DiTAttention(dim, heads, pe_attn_head)
        self.norm2 = AdaLayerNorm(dim)
        inner      = int(dim * ff_mult)
        self.ff    = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Linear(inner, dim),
        )

    def forward(self, x, time_emb, mask=None):
        x = x + self.attn(self.norm1(x, time_emb), mask)
        x = x + self.ff(self.norm2(x, time_emb))
        return x


# ─────────────────────────────────────────
# 6. Time Embedding (Sinusoidal → MLP)
# ─────────────────────────────────────────
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        # t: [B] float (0~1 diffusion time)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        emb = t.unsqueeze(-1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # [B, dim]
        return self.mlp(emb)


# ─────────────────────────────────────────
# 7. Text Encoder (Character-level)
# ─────────────────────────────────────────
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, dim, text_dim):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, text_dim, padding_idx=0)
        self.proj    = nn.Linear(text_dim, dim)
        self.conv    = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
        )

    def forward(self, tokens):
        # tokens: [B, T_text]
        x = self.embed(tokens)           # [B, T, text_dim]
        x = self.proj(x)                 # [B, T, dim]
        x = rearrange(x, "b t c -> b c t")
        x = self.conv(x)
        x = rearrange(x, "b c t -> b t c")
        return x


# ─────────────────────────────────────────
# 8. F5-TTS Main Model
# ─────────────────────────────────────────
class F5TTS(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        dim = cfg.dim

        # Input projection: mel → dim
        self.input_proj = nn.Linear(cfg.n_mel_channels, dim)

        # Text encoder
        self.text_encoder = TextEncoder(vocab_size, dim, cfg.text_dim)

        # Time embedding
        self.time_embed = TimeEmbedding(dim)

        # CNN blocks (conv_layers ชั้น ก่อน DiT)
        self.conv_blocks = nn.ModuleList([
            ConvNeXtBlock(dim) for _ in range(cfg.conv_layers)
        ])

        # DiT blocks
        self.dit_blocks = nn.ModuleList([
            DiTBlock(dim, cfg.heads, cfg.ff_mult, cfg.pe_attn_head)
            for _ in range(cfg.depth)
        ])

        # Output projection: dim → mel
        self.norm_out  = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, cfg.n_mel_channels)

    def forward(self, noisy_mel, time, text_tokens, mask=None):
        """
        noisy_mel   : [B, T_mel, n_mel]   noisy mel spectrogram (xt)
        time        : [B]                  diffusion timestep (0~1)
        text_tokens : [B, T_text]          character token ids
        mask        : [B, T_mel]           True = valid frame

        Returns → predicted velocity [B, T_mel, n_mel]
        """
        B, T, _ = noisy_mel.shape

        # 1. Project mel
        x = self.input_proj(noisy_mel)          # [B, T, dim]

        # 2. Text condition (concat along time dim, same as F5-TTS)
        text = self.text_encoder(text_tokens)    # [B, T_text, dim]
        # Pad/trim text to same length as mel for concat conditioning
        T_text = text.size(1)
        if T_text < T:
            text = F.pad(text, (0, 0, 0, T - T_text))
        else:
            text = text[:, :T, :]
        x = x + text                            # additive conditioning

        # 3. Time embedding
        t_emb = self.time_embed(time)            # [B, dim]

        # 4. CNN blocks
        for conv in self.conv_blocks:
            x = conv(x)

        # 5. DiT blocks
        for dit in self.dit_blocks:
            x = dit(x, t_emb, mask)

        # 6. Output
        x = self.norm_out(x)
        return self.output_proj(x)              # [B, T, n_mel]


# ─────────────────────────────────────────
# 9. Load Pretrained Weights
# ─────────────────────────────────────────
def load_pretrained(model: F5TTS, ckpt_path: str, strict: bool = False):
    """
    โหลด pretrained weights เข้า model ที่เราสร้างเอง
    strict=False เผื่อ key ไม่ตรงกันบางส่วน
    """
    print(f"📥 Loading pretrained weights from: {ckpt_path}")

    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path, device="cpu")
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

    # ── Key mapping ──
    # ชื่อ key ใน pretrained อาจต่างจากที่เราตั้ง
    # ปรับ mapping นี้ถ้า key ไม่ตรง
    KEY_MAP = {
        "transformer.":      "dit_blocks.",
        "mel_encoder.":      "input_proj.",
        "text_embed.":       "text_encoder.",
        "time_embed.":       "time_embed.",
        "conv_embed.":       "conv_blocks.",
        "to_pred.":          "output_proj.",
        "final_norm.":       "norm_out.",
    }

    new_state = {}
    for k, v in state_dict.items():
        new_k = k
        for old, new in KEY_MAP.items():
            if new_k.startswith(old):
                new_k = new_k.replace(old, new, 1)
                break
        new_state[new_k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=strict)

    if missing:
        print(f"⚠️  Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    print("✅ Pretrained weights loaded!")
    return model