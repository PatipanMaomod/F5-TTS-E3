import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


# ─────────────────────────────────────────
# Rotary Positional Embedding
# ─────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t     = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


# ─────────────────────────────────────────
# ConvNeXt V2 Block
# ─────────────────────────────────────────
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, mult=2, kernel_size=7):
        super().__init__()
        self.dw_conv   = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.norm      = nn.LayerNorm(dim)
        inner          = int(dim * mult)
        self.pw1       = nn.Linear(dim, inner)
        self.pw2       = nn.Linear(inner, dim)
        self.grn_gamma = nn.Parameter(torch.zeros(inner))
        self.grn_beta  = nn.Parameter(torch.zeros(inner))

    def _grn(self, x):
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.grn_gamma * (x * nx) + self.grn_beta + x

    def forward(self, x):
        res = x
        x   = rearrange(self.dw_conv(rearrange(x, "b t c -> b c t")), "b c t -> b t c")
        x   = self.norm(x)
        x   = self._grn(F.gelu(self.pw1(x)))
        return self.pw2(x) + res


# ─────────────────────────────────────────
# Adaptive LayerNorm
# ─────────────────────────────────────────
class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(dim, dim * 2)

    def forward(self, x, cond):
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ─────────────────────────────────────────
# DiT Attention  ← แก้ inplace bug
# ─────────────────────────────────────────
class DiTAttention(nn.Module):
    def __init__(self, dim, heads, pe_attn_head=1):
        super().__init__()
        self.heads    = heads
        self.head_dim = dim // heads
        self.pe_heads = pe_attn_head
        self.to_qkv   = nn.Linear(dim, dim * 3, bias=False)
        self.to_out   = nn.Linear(dim, dim, bias=False)
        self.rope     = RotaryEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q, k, v = rearrange(
            self.to_qkv(x), "b t (three h d) -> three b h t d",
            three=3, h=self.heads
        ).unbind(0)

        cos, sin = self.rope(T, x.device)

        # ใช้ torch.cat แทน inplace assignment
        if self.pe_heads > 0:
            q_rope, k_rope = apply_rotary(
                q[:, :self.pe_heads].contiguous(),
                k[:, :self.pe_heads].contiguous(),
                cos, sin
            )
            q = torch.cat([q_rope, q[:, self.pe_heads:]], dim=1)
            k = torch.cat([k_rope, k[:, self.pe_heads:]], dim=1)

        attn = torch.einsum("bhid,bhjd->bhij", q, k) * (self.head_dim ** -0.5)
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        out = torch.einsum("bhij,bhjd->bhid", F.softmax(attn, dim=-1), v)
        return self.to_out(rearrange(out, "b h t d -> b t (h d)"))


# ─────────────────────────────────────────
# DiT Block
# ─────────────────────────────────────────
class DiTBlock(nn.Module):
    def __init__(self, dim, heads, ff_mult=2, pe_attn_head=1):
        super().__init__()
        self.norm1 = AdaLayerNorm(dim)
        self.attn  = DiTAttention(dim, heads, pe_attn_head)
        self.norm2 = AdaLayerNorm(dim)
        inner      = int(dim * ff_mult)
        self.ff    = nn.Sequential(nn.Linear(dim, inner), nn.GELU(), nn.Linear(inner, dim))

    def forward(self, x, t_emb, mask=None):
        x = x + self.attn(self.norm1(x, t_emb), mask)
        x = x + self.ff(self.norm2(x, t_emb))
        return x


# ─────────────────────────────────────────
# Time Embedding
# ─────────────────────────────────────────
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        emb   = torch.cat([
            (t.unsqueeze(-1) * freqs).sin(),
            (t.unsqueeze(-1) * freqs).cos()
        ], dim=-1)
        return self.mlp(emb)


# ─────────────────────────────────────────
# Text Encoder
# ─────────────────────────────────────────
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, dim, text_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, text_dim, padding_idx=0)
        self.proj  = nn.Linear(text_dim, dim)
        self.conv  = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, tokens):
        x = self.proj(self.embed(tokens))
        x = rearrange(x, "b t c -> b c t")
        x = F.gelu(self.conv(x))
        return rearrange(x, "b c t -> b t c")


# ─────────────────────────────────────────
# F5-TTS
# ─────────────────────────────────────────
class F5TTS(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        dim = cfg.dim
        self.input_proj   = nn.Linear(cfg.n_mel_channels, dim)
        self.text_encoder = TextEncoder(vocab_size, dim, cfg.text_dim)
        self.time_embed   = TimeEmbedding(dim)
        self.conv_blocks  = nn.ModuleList([ConvNeXtBlock(dim) for _ in range(cfg.conv_layers)])
        self.dit_blocks   = nn.ModuleList([
            DiTBlock(dim, cfg.heads, cfg.ff_mult, cfg.pe_attn_head)
            for _ in range(cfg.depth)
        ])
        self.norm_out    = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, cfg.n_mel_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Conv1d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, noisy_mel, time, text_tokens, mask=None):
        B, T, _ = noisy_mel.shape

        # 1. project mel
        x = self.input_proj(noisy_mel)          # [B, T, dim]

        # 2. text conditioning
        text = self.text_encoder(text_tokens)    # [B, T_text, dim]
        T_text = text.size(1)
        if T_text < T:
            text = F.pad(text, (0, 0, 0, T - T_text))
        else:
            text = text[:, :T, :]
        x = x + text

        # 3. time embedding
        t_emb = self.time_embed(time)            # [B, dim]

        # 4. CNN blocks
        for conv in self.conv_blocks:
            x = conv(x)

        # 5. DiT blocks
        for dit in self.dit_blocks:
            x = dit(x, t_emb, mask)

        # 6. output
        return self.output_proj(self.norm_out(x))  # [B, T, n_mel]