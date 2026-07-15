import os
import time
import signal
import argparse
import shutil
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as torch_mp

from torch.utils.data import Dataset, DataLoader

from model.gpt import GPT
from model.config import GPTConfig
from training.checkpoint import save_checkpoint, load_checkpoint
from training.scheduler import cosine_scheduler


# ==========================
# Kaggle-specific fixes
# ==========================
# 1) Kaggle's dual-T4 instances don't support NCCL P2P/InfiniBand properly
#    inside their container. Without disabling these, dist.init_process_group
#    (and later all_reduce/broadcast calls) will hang forever with no error.
#    Set these BEFORE any torch.distributed call. Only set them if the user
#    hasn't already overridden them.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
# Helps surface a real error instead of a silent hang if something is still wrong.
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

# 2) Kaggle containers ship with a very small /dev/shm (often 64MB). The default
#    DataLoader sharing strategy ("file_descriptor"/shared memory) needs much
#    more than that once you have multiple workers + pin_memory queuing batches,
#    and it hangs (not crashes) when it runs out, which looks exactly like
#    "stuck, RAM climbing, GPU idle". Switching to "file_system" avoids /dev/shm
#    entirely.
torch_mp.set_sharing_strategy("file_system")


class TokenDataset(Dataset):

    def __init__(self, filename, context_length):
        self.data = np.memmap(
            filename,
            dtype=np.uint16,
            mode="r"
        )
        self.context_length = context_length
        self.true_length = len(self.data) - context_length - 1

    def __len__(self):
        # Return small fixed number so no sampler ever builds a huge index
        return 10_000

    def __getitem__(self, idx):
        # Always sample randomly from the FULL file
        idx = random.randint(0, self.true_length - 1)

        x = torch.from_numpy(
            self.data[idx : idx + self.context_length].astype(np.int64)
        )
        y = torch.from_numpy(
            self.data[idx + 1 : idx + self.context_length + 1].astype(np.int64)
        )

        return x, y


def main():
    # ==========================
    # Arguments
    # ==========================
    parser = argparse.ArgumentParser()

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--minutes", type=float, default=None)
    parser.add_argument("--hours", type=float, default=None)

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of DataLoader worker processes PER GPU. Use 0 to disable "
             "multiprocessing entirely (safest fallback if you still see hangs)."
    )

    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="torch intra-op thread count (was previously conflated with DataLoader workers)."
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/kaggle/input/datasets/ducky69/dataset-rust"
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints"
    )

    args = parser.parse_args()

    # ==========================
    # Distributed setup
    # ==========================
    ddp = "LOCAL_RANK" in os.environ

    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    else:
        local_rank = 0
        world_size = 1

    # Every process must have a GPU
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA devices found. This script requires at least one GPU."
        )

    # Assign one GPU per process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if ddp:
        # timeout guards against another silent hang: if init still can't
        # complete (e.g. NCCL truly broken), fail loudly after 2 minutes
        # instead of hanging for the rest of the session.
        dist.init_process_group(
            backend="nccl",
            timeout=torch.distributed.default_pg_timeout if hasattr(torch.distributed, "default_pg_timeout") else None,
        )

    is_main_process = (local_rank == 0)

    if is_main_process:
        print(f"Using device: {device}", flush=True)
        print(f"CUDA device name: {torch.cuda.get_device_name(local_rank)}", flush=True)
        if ddp:
            print(f"Distributed training with {world_size} GPUs", flush=True)
        else:
            print("Single-GPU training (LOCAL_RANK not set — did you launch with torchrun?)", flush=True)

    # ==========================
    # Performance settings
    # ==========================
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ==========================
    # Settings
    # ==========================
    batch_size                  = 8
    gradient_accumulation_steps = 2
    context_length              = 1024
    max_steps                   = 200_000

    learning_rate               = 3e-4
    min_learning_rate           = 3e-5
    warmup_steps                = 2000
    weight_decay                = 0.1

    validation_interval         = 500
    checkpoint_interval         = 500

    # ==========================
    # Timer
    # ==========================
    start_time = time.time()
    time_limit = None

    if args.minutes is not None:
        time_limit = args.minutes * 60

    if args.hours is not None:
        time_limit = args.hours * 3600

    # ==========================
    # Paths
    # ==========================
    if is_main_process:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    if ddp:
        dist.barrier()

    latest_checkpoint = os.path.join(args.checkpoint_dir, "latest.pt")
    best_checkpoint   = os.path.join(args.checkpoint_dir, "best.pt")

    input_latest = os.path.join(args.dataset_dir, "checkpoints", "latest.pt")
    input_best   = os.path.join(args.dataset_dir, "checkpoints", "best.pt")

    if args.resume and is_main_process:
        if (not os.path.exists(latest_checkpoint)) and os.path.exists(input_latest):
            shutil.copy2(input_latest, latest_checkpoint)

        if (not os.path.exists(best_checkpoint)) and os.path.exists(input_best):
            shutil.copy2(input_best, best_checkpoint)

    if ddp:
        dist.barrier()

    # ==========================
    # Dataset
    # ==========================
    if is_main_process:
        print("Loading dataset...", flush=True)

    train_path = os.path.join(args.dataset_dir, "train.bin")
    val_path   = os.path.join(args.dataset_dir, "validation.bin")

    if is_main_process:
        print(f"Train file:      {train_path}", flush=True)
        print(f"Validation file: {val_path}",  flush=True)
        # Sanity-check the files actually exist and report their size, since a
        # bad --dataset-dir path silently produces a 0-length memmap that will
        # also look like "stuck at startup".
        for p in (train_path, val_path):
            if os.path.exists(p):
                size_gb = os.path.getsize(p) / 1024**3
                print(f"  {p}: {size_gb:.2f} GB", flush=True)
            else:
                print(f"  WARNING: {p} does not exist!", flush=True)

    train_dataset = TokenDataset(train_path, context_length)
    val_dataset   = TokenDataset(val_path,   context_length)

    if is_main_process:
        print(f"Dataset ready! Tokens in train: {train_dataset.true_length:,}", flush=True)

    # No DistributedSampler - random sampling handles distribution naturally
    # Each GPU gets different random samples automatically
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        drop_last=True,
        prefetch_factor=(2 if args.workers > 0 else None),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        drop_last=False,
        prefetch_factor=(2 if args.workers > 0 else None),
    )

    train_iter = iter(train_loader)

    if is_main_process:
        print("Dataloader ready!", flush=True)

    # ==========================
    # Model -- built directly on the target GPU
    # ==========================
    if is_main_process:
        print("Creating model...", flush=True)

    base_model = GPT(GPTConfig()).to(device)

    if is_main_process:
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.memory_allocated(i) / 1024**2
            print(f"GPU {i} VRAM after model load: {mem:.1f} MB", flush=True)

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    else:
        model = base_model

    if is_main_process:
        parameters = sum(p.numel() for p in base_model.parameters())
        print(f"Parameters: {parameters:,}", flush=True)

    # ==========================
    # Optimizer / scheduler
    # ==========================
    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=True,
    )

    scheduler = cosine_scheduler(
        optimizer,
        warmup_steps,
        max_steps,
        min_learning_rate,
        learning_rate,
    )

    scaler  = torch.amp.GradScaler("cuda")
    loss_fn = nn.CrossEntropyLoss()

    # ==========================
    # Resume
    # ==========================
    step        = 0
    best_loss   = float("inf")
    tokens_seen = 0

    if args.resume and os.path.exists(latest_checkpoint):
        if is_main_process:
            print("Loading checkpoint...", flush=True)

        step, best_loss, tokens_seen = load_checkpoint(
            latest_checkpoint,
            base_model,
            optimizer,
            scheduler,
            scaler,
        )

        if is_main_process:
            print(f"Resumed from step {step}", flush=True)

        if ddp:
            for param in base_model.parameters():
                dist.broadcast(param.data, src=0)

    elif args.resume and is_main_process:
        print("No latest.pt found, starting from scratch.", flush=True)

    # ==========================
    # Shutdown signal handler
    # ==========================
    running = True

    def shutdown(sig, frame):
        nonlocal running
        if is_main_process:
            print("\nStopping safely...", flush=True)
        running = False

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ==========================
    # Validation helper
    # ==========================
    @torch.no_grad()
    def validate():
        model.eval()

        total_loss = torch.zeros(1, device=device)
        count      = torch.zeros(1, device=device)

        for batch_idx, (x, y) in enumerate(val_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss   = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                )

            total_loss += loss
            count      += 1

            if count.item() >= 100:
                break

        if ddp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(count,      op=dist.ReduceOp.SUM)

        model.train()
        return (total_loss / count.clamp(min=1)).item()

    # ==========================
    # Training loop
    # ==========================
    model.train()
    epoch = 0

    if is_main_process:
        print("Starting training...", flush=True)

    while step < max_steps and running:
        if time_limit and (time.time() - start_time > time_limit):
            if is_main_process:
                print("Time limit reached", flush=True)
            break

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for micro_step in range(gradient_accumulation_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                epoch += 1
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
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss   = loss_fn(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                    )
                    loss = loss / gradient_accumulation_steps

                scaler.scale(loss).backward()
                total_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        tokens_seen += (
            batch_size
            * context_length
            * gradient_accumulation_steps
            * world_size
        )

        step += 1

        if is_main_process and step % 10 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"step {step}/{max_steps} | loss {total_loss:.4f} | lr {lr_now:.2e}",
                flush=True,
            )

        # ======================
        # Validation + best.pt
        # ======================
        if step % validation_interval == 0:
            if ddp:
                dist.barrier()

            val_loss = validate()

            if is_main_process:
                print(f"Validation loss: {val_loss:.4f}", flush=True)

                if val_loss < best_loss:
                    best_loss = val_loss
                    print("⭐ New best model!", flush=True)

                    save_checkpoint(
                        best_checkpoint,
                        base_model,
                        optimizer,
                        scheduler,
                        scaler,
                        step,
                        best_loss,
                        tokens_seen,
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
                    latest_checkpoint,
                    base_model,
                    optimizer,
                    scheduler,
                    scaler,
                    step,
                    best_loss,
                    tokens_seen,
                )
                print(f"Saved latest.pt (step {step})", flush=True)

            if ddp:
                dist.barrier()

    # ==========================
    # Exit save
    # ==========================
    if ddp:
        dist.barrier()

    if is_main_process:
        print("Saving final checkpoint...", flush=True)

        save_checkpoint(
            latest_checkpoint,
            base_model,
            optimizer,
            scheduler,
            scaler,
            step,
            best_loss,
            tokens_seen,
        )

        print("Finished!", flush=True)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
