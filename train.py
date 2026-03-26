import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.amp import autocast, GradScaler
from config import F5Config
from model import F5TTS
from dataset import VoiceDataset, collate_fn
import bitsandbytes as bnb
from tqdm import tqdm
from functools import partial


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


def load_loss_history(path):
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        return {
            "train": {int(k): v for k, v in raw["train"].items()},
            "eval":  {int(k): v for k, v in raw["eval"].items()},
        }
    return {"train": {}, "eval": {}}


def save_loss_history(path, history):
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def maybe_resume(model, optimizer, scheduler, scaler, cfg):
    start_step = 0

    ckpt_path = cfg.resume_ckpt
    if not ckpt_path:
        ckpts = sorted([
            f for f in os.listdir(cfg.output_dir)
            if f.startswith("step_") and f.endswith(".pt")
        ]) if os.path.exists(cfg.output_dir) else []
        if ckpts:
            ckpt_path = os.path.join(cfg.output_dir, ckpts[-1])

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        print(f"   → resumed at step {start_step:,}")
    else:
        print("No checkpoint found, training from scratch")

    return start_step


@torch.no_grad()
def run_eval(model, loader, device, cfg):
    model.eval()
    total_loss    = 0.0
    total_batches = 0

    for mel, tokens, mask in loader:
        mel    = mel.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        mask   = mask.to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.float16):
            loss = flow_matching_loss(model, mel, tokens, mask)

        total_loss    += loss.item()
        total_batches += 1

        if cfg.eval_max_batches != -1 and total_batches >= cfg.eval_max_batches:
            break

    model.train()
    return total_loss / max(total_batches, 1)


def train():
    cfg    = F5Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    LOSS_HISTORY_PATH = os.path.join(cfg.output_dir, "loss_history.json")

    # ── Dataset ──
    train_dataset = VoiceDataset(cfg.train_data_path, cfg.vocab_path,
                                 cfg.sample_rate, cfg.n_mel_channels)
    val_dataset   = VoiceDataset(cfg.eval_data_path, cfg.vocab_path,
                                 cfg.sample_rate, cfg.n_mel_channels)

    collate = partial(collate_fn, max_frames=cfg.max_mel_len)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size_per_gpu,
                              shuffle=True, collate_fn=collate,
                              num_workers=cfg.num_workers, pin_memory=True,
                              persistent_workers=cfg.num_workers > 0)
    val_loader   = DataLoader(val_dataset, batch_size=cfg.batch_size_per_gpu,
                              shuffle=False, collate_fn=collate,
                              num_workers=cfg.num_workers, pin_memory=True,
                              persistent_workers=cfg.num_workers > 0)

    print(f"Dataset  →  train: {len(train_dataset):,}  |  val: {len(val_dataset):,}")

    # ── Vocab / Model ──
    with open(cfg.vocab_path, encoding="utf-8") as f:
        vocab_size = sum(1 for l in f if l.strip())

    model = F5TTS(cfg, vocab_size=vocab_size).to(device)
    model = torch.compile(model)
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=cfg.learning_rate,
                                    betas=(0.9, 0.98), weight_decay=1e-2, eps=1e-8)
    scheduler = build_scheduler(optimizer, cfg.warmup_steps, cfg.total_steps)
    scaler    = GradScaler()

    start_step   = maybe_resume(model, optimizer, scheduler, scaler, cfg)
    loss_history = load_loss_history(LOSS_HISTORY_PATH)  # โหลดจาก JSON

    # ── Training Loop ──
    model.train()
    step     = start_step
    accum    = 0
    run_loss = 0.0
    optimizer.zero_grad()

    print(f"\nTraining | total: {cfg.total_steps:,} | eval every: {cfg.eval_every:,} | device: {device}\n")
    pbar = tqdm(total=cfg.total_steps, initial=step, desc="Training", unit="step", dynamic_ncols=True)

    while step < cfg.total_steps:
        for mel, tokens, mask in train_loader:
            mel    = mel.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            mask   = mask.to(device, non_blocking=True)

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

            avg_train_loss = run_loss * cfg.grad_accum
            lr             = scheduler.get_last_lr()[0]
            run_loss       = 0.0

            # บันทึก train loss
            loss_history["train"][step] = avg_train_loss
            save_loss_history(LOSS_HISTORY_PATH, loss_history)

            # ── Eval ──
            eval_loss = None
            if step % cfg.eval_every == 0 or step == cfg.total_steps:
                eval_loss = run_eval(model, val_loader, device, cfg)
                loss_history["eval"][step] = eval_loss
                save_loss_history(LOSS_HISTORY_PATH, loss_history)
                tqdm.write(
                    f"[step {step:>7,}]  train: {avg_train_loss:.4f}"
                    f"  |  eval: {eval_loss:.4f}  |  lr: {lr:.2e}"
                )

            pbar.set_postfix({
                "train": f"{avg_train_loss:.4f}",
                "eval":  f"{eval_loss:.4f}" if eval_loss is not None else "-",
                "lr":    f"{lr:.2e}",
            })
            pbar.update(1)

            # ── Checkpoint ──
            if step % cfg.save_checkpoint == 0 or step == cfg.total_steps:
                path = f"{cfg.output_dir}/step_{step:08d}.pt"
                torch.save({
                    "step":      step,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler":    scaler.state_dict(),
                    "cfg":       cfg,
                }, path)
                print(f"Saved → {path}")

            if step >= cfg.total_steps:
                break

    pbar.close()
    print(f"\nDone! Loss history → {LOSS_HISTORY_PATH}")


if __name__ == "__main__":
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    train()