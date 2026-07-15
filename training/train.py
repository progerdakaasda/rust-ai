import os
import time
import signal
import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.gpt import GPT
from model.config import GPTConfig

from training.dataset import TokenDataset
from training.checkpoint import save_checkpoint, load_checkpoint
from training.scheduler import cosine_scheduler


# ==========================
# Performance settings
# ==========================

torch.set_num_threads(4)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# ==========================
# Arguments
# ==========================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--resume",
    action="store_true"
)

parser.add_argument(
    "--minutes",
    type=float,
    default=None
)

parser.add_argument(
    "--hours",
    type=float,
    default=None
)

args = parser.parse_args()


# ==========================
# Distributed setup
# ==========================

ddp = "LOCAL_RANK" in os.environ and torch.cuda.is_available()

if ddp:
    dist.init_process_group(
        backend="nccl"
    )

    local_rank = int(
        os.environ["LOCAL_RANK"]
    )

    torch.cuda.set_device(local_rank)

    device = f"cuda:{local_rank}"
else:
    local_rank = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

is_main_process = local_rank == 0

if is_main_process:
    print(f"Using device: {device}")
    if ddp:
        print(f"Distributed training with {dist.get_world_size()} GPUs")


# ==========================
# Settings
# ==========================

batch_size = 8  # per GPU in DDP
gradient_accumulation_steps = 2

context_length = 1024
max_steps = 200_000

learning_rate = 3e-4
min_learning_rate = 3e-5
warmup_steps = 2000
weight_decay = 0.1

validation_interval = 500
checkpoint_interval = 500

num_workers = 4


# ==========================
# Timer
# ==========================

start_time = time.time()
time_limit = None

if args.minutes:
    time_limit = args.minutes * 60

if args.hours:
    time_limit = args.hours * 3600


# ==========================
# Dataset
# ==========================

if is_main_process:
    print("Loading dataset...")

DATASET_DIR = "/kaggle/input/datasets/ducky69/dataset-rust"

train_dataset = TokenDataset(
    f"{DATASET_DIR}/train.bin",
    context_length
)

val_dataset = TokenDataset(
    f"{DATASET_DIR}/validation.bin",
    context_length
)

if ddp:
    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        drop_last=False
    )

    val_sampler = DistributedSampler(
        val_dataset,
        shuffle=False,
        drop_last=False
    )
else:
    train_sampler = None
    val_sampler = None

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    sampler=val_sampler,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None
)

train_iter = iter(train_loader)


# ==========================
# Model
# ==========================

if is_main_process:
    print("Creating model...")

model = GPT(
    GPTConfig()
).to(device)

if ddp:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False
    )

base_model = model.module if ddp else model

if is_main_process:
    parameters = sum(
        p.numel()
        for p in base_model.parameters()
    )
    print(f"Parameters: {parameters:,}")


# ==========================
# Optimizer
# ==========================

optimizer = torch.optim.AdamW(
    base_model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

scheduler = cosine_scheduler(
    optimizer,
    warmup_steps,
    max_steps,
    min_learning_rate,
    learning_rate
)

scaler = torch.amp.GradScaler("cuda")

loss_fn = nn.CrossEntropyLoss()


# ==========================
# Resume
# ==========================

step = 0
best_loss = float("inf")
tokens_seen = 0

if args.resume:
    if is_main_process:
        print("Loading checkpoint...")

    step, best_loss, tokens_seen = load_checkpoint(
        "checkpoints/latest.pt",
        base_model,
        optimizer,
        scheduler,
        scaler
    )

    if is_main_process:
        print(f"Resumed from step {step}")


# ==========================
# Shutdown
# ==========================

running = True


def shutdown(sig, frame):
    global running
    if is_main_process:
        print("\nStopping safely...")
    running = False


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


# ==========================
# Validation
# ==========================

@torch.no_grad()
def validate():
    model.eval()

    total_loss = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)

    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            logits = model(x)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

        total_loss += loss.detach()
        count += 1.0

        if count.item() >= 100:
            break

    if ddp:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

    model.train()

    return (total_loss / count).item()


# ==========================
# Training
# ==========================

model.train()

epoch = 0

while step < max_steps and running:
    if time_limit and (time.time() - start_time > time_limit):
        if is_main_process:
            print("Time limit reached")
        break

    if ddp and train_sampler is not None:
        train_sampler.set_epoch(epoch)

    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0

    for micro_step in range(gradient_accumulation_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            epoch += 1
            if ddp and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        sync_context = (
            model.no_sync()
            if ddp and micro_step < gradient_accumulation_steps - 1
            else nullcontext()
        )

        with sync_context:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                logits = model(x)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()
            total_loss += loss.item()

    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(
        base_model.parameters(),
        1.0
    )

    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    tokens_seen += (
        batch_size *
        context_length *
        gradient_accumulation_steps *
        (dist.get_world_size() if ddp else 1)
    )

    step += 1

    if is_main_process and step % 10 == 0:
        print(
            f"step {step}/{max_steps} | "
            f"loss {total_loss:.4f}"
        )

    # ======================
    # Validation + best.pt
    # ======================

    if step % validation_interval == 0:
        if ddp:
            dist.barrier()

        val_loss = validate()

        if is_main_process:
            print(f"Validation loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                print("⭐ New best model!")

                save_checkpoint(
                    "checkpoints/best.pt",
                    base_model,
                    optimizer,
                    scheduler,
                    scaler,
                    step,
                    best_loss,
                    tokens_seen
                )

        if ddp:
            dist.barrier()

    # ======================
    # latest.pt
    # ======================

    if step % checkpoint_interval == 0:
        if ddp:
            dist.barrier()

        if is_main_process:
            save_checkpoint(
                "checkpoints/latest.pt",
                base_model,
                optimizer,
                scheduler,
                scaler,
                step,
                best_loss,
                tokens_seen
            )
            print("Saved latest.pt")

        if ddp:
            dist.barrier()


# ==========================
# Exit save
# ==========================

if ddp:
    dist.barrier()

if is_main_process:
    print("Saving latest checkpoint...")

    save_checkpoint(
        "checkpoints/latest.pt",
        base_model,
        optimizer,
        scheduler,
        scaler,
        step,
        best_loss,
        tokens_seen
    )

    print("Finished!")

if ddp:
    dist.destroy_process_group()
