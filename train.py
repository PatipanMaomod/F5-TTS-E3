import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import os
from config import F5Config
from model import F5TTS, load_pretrained
from dataset import VoiceDataset, collate_fn


def get_scheduler(optimizer, warmup_steps, total_steps):
    warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0,
                      total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer,
                               T_max=total_steps - warmup_steps,
                               eta_min=1e-7)
    return SequentialLR(optimizer, [warmup, cosine],
                        milestones=[warmup_steps])


def sample_diffusion_time(batch_size, device):
    """Flow matching: sample t ~ U(0,1)"""
    return torch.rand(batch_size, device=device)


def flow_matching_loss(model, clean_mel, text_tokens, mask):
    """
    Conditional Flow Matching loss
    x0 = noise, x1 = clean mel
    xt = (1-t)*x0 + t*x1
    target velocity = x1 - x0
    """
    B = clean_mel.size(0)
    device = clean_mel.device

    t   = sample_diffusion_time(B, device)          # [B]
    x0  = torch.randn_like(clean_mel)               # noise
    t_  = t.view(B, 1, 1)
    xt  = (1 - t_) * x0 + t_ * clean_mel           # interpolate
    vel = clean_mel - x0                            # target velocity

    pred = model(xt, t, text_tokens, mask)          # predicted velocity

    # loss เฉพาะ valid frames
    loss = F.mse_loss(pred[mask], vel[mask])
    return loss


def train():
    cfg    = F5Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Dataset ──
    dataset = VoiceDataset(cfg.data_path, cfg.vocab_path,
                           cfg.sample_rate, cfg.n_mel_channels)
    loader  = DataLoader(dataset, batch_size=16, shuffle=True,
                         collate_fn=collate_fn,
                         num_workers=cfg.num_workers, pin_memory=True)

    # ── Model ──
    vocab_size = len(dataset.char2id) + 1
    model = F5TTS(cfg, vocab_size).to(device)

    # ── โหลด pretrained weights ──
    model = load_pretrained(model, cfg.pretrain_path, strict=False)

    # ── Optimizer & Scheduler ──
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate,
                      betas=(0.9, 0.98), weight_decay=1e-2)
    scheduler = get_scheduler(optimizer, cfg.warmup_steps, cfg.total_steps)

    # ── Training Loop ──
    model.train()
    step    = 0
    accum   = 0
    optimizer.zero_grad()

    print(f"🚀 Start training | vocab={vocab_size} | device={device}")

    while step < cfg.total_steps:
        for mel, tokens, mask in loader:
            mel    = mel.to(device)
            tokens = tokens.to(device)
            mask   = mask.to(device)

            loss = flow_matching_loss(model, mel, tokens, mask)
            loss = loss / cfg.grad_accum
            loss.backward()

            accum += 1
            if accum % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 100 == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"step {step:>6} | loss {loss.item()*cfg.grad_accum:.4f}"
                          f" | lr {lr:.2e}")

                if step % 2000 == 0:
                    ckpt = {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    path = f"{cfg.output_dir}/step_{step:07d}.pt"
                    torch.save(ckpt, path)
                    print(f"💾 Saved: {path}")

                if step >= cfg.total_steps:
                    break

    print("✅ Training complete!")


if __name__ == "__main__":
    train()