import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.amp import autocast, GradScaler
from config import F5Config
from model import F5TTS
from dataset import VoiceDataset, collate_fn
import bitsandbytes as bnb


def flow_matching_loss(model, clean_mel, text_tokens, mask):
    B      = clean_mel.size(0)
    device = clean_mel.device
    t      = torch.rand(B, device=device)
    x0     = torch.randn_like(clean_mel)
    xt     = (1 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * clean_mel
    target = clean_mel - x0
    pred   = model(xt, t, text_tokens, mask)
    return F.mse_loss(pred[mask], target[mask])


def build_scheduler(optimizer, warmup_steps, total_steps):
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-8, end_factor=1.0,
                     total_iters=warmup_steps),
            CosineAnnealingLR(optimizer,
                              T_max=total_steps - warmup_steps,
                              eta_min=1e-7),
        ],
        milestones=[warmup_steps],
    )


def maybe_resume(model, optimizer, scheduler, scaler, cfg):
    start_step = 0
    if cfg.resume_ckpt and os.path.exists(cfg.resume_ckpt):
        print(f"🔄 Resuming from {cfg.resume_ckpt}")
        ckpt = torch.load(cfg.resume_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        print(f"   → resumed at step {start_step}")
    return start_step


def train():
    cfg    = F5Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Dataset ──
    dataset = VoiceDataset(cfg.data_path, cfg.vocab_path,
                           cfg.sample_rate, cfg.n_mel_channels)
    loader  = DataLoader(
        dataset,
        batch_size=cfg.batch_size_per_gpu,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    # ── Vocab size ──
    with open(cfg.vocab_path, encoding="utf-8") as f:
        vocab_size = sum(1 for l in f if l.strip())

    # ── Model ──
    model = F5TTS(cfg, vocab_size=vocab_size).to(device)

    # torch.compile — เร็วขึ้น ~20% และประหยัด memory
    model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"📐 Model params: {total_params:.1f}M")


    # แทนที่ optimizer เดิม
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=1e-2,
        eps=1e-8,
    )
    scheduler  = build_scheduler(optimizer, cfg.warmup_steps, cfg.total_steps)
    scaler     = GradScaler()                    # fp16 scaler

    start_step = maybe_resume(model, optimizer, scheduler, scaler, cfg)

    # ── Training Loop ──
    model.train()
    step     = start_step
    accum    = 0
    run_loss = 0.0
    optimizer.zero_grad()

    print(f"\n🚀 Training from scratch | steps: {cfg.total_steps:,} | device: {device}\n")

    while step < cfg.total_steps:
        for mel, tokens, mask in loader:
            mel    = mel.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            mask   = mask.to(device, non_blocking=True)

            # ── fp16 forward ──
            with autocast(device_type="cuda", dtype=torch.float16):
                loss = flow_matching_loss(model, mel, tokens, mask) / cfg.grad_accum

            scaler.scale(loss).backward()
            run_loss += loss.item()
            accum    += 1

            if accum % cfg.grad_accum != 0:
                continue

            # ── gradient step ──
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            step  += 1
            accum  = 0

            if step % 100 == 0:
                avg_loss = run_loss * cfg.grad_accum / 100
                lr       = scheduler.get_last_lr()[0]
                print(f"step {step:>7,} | loss {avg_loss:.4f} | lr {lr:.2e}")
                run_loss = 0.0

            if step % 5000 == 0 or step == cfg.total_steps:
                path = f"{cfg.output_dir}/step_{step:08d}.pt"
                torch.save({
                    "step":      step,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler":    scaler.state_dict(),
                    "cfg":       cfg,
                }, path)
                print(f"💾 Saved → {path}")

            if step >= cfg.total_steps:
                break

    print("\n✅ Done!")


if __name__ == "__main__":
    # ลด fragmentation
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    train()