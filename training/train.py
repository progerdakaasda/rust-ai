import os
import time
import signal
import argparse
import shutil
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.gpt import GPT
from model.config import GPTConfig
from training.checkpoint import save_checkpoint, load_checkpoint
from training.scheduler import cosine_scheduler


class BinTokenDataset(Dataset):
    """
    Memory-mapped token dataset for flat binary token files.

    Expects:
      - train.bin
      - validation.bin

    where each file is a flat array of token IDs stored as uint16 or uint32.
    """
    def __init__(self, path: str, context_length: int, dtype: np.dtype):
        self.path = path
        self.context_length = context_length
        self.dtype = np.dtype(dtype)
        self._data = None
        self._n_tokens = os.path.getsize(path) // self.dtype.itemsize

        if self._n_tokens <= context_length + 1:
            raise ValueError(
                f"File {path} is too small for context_length={context_length}"
            )

    def _open(self):
        if self._data is None:
            self._data = np.memmap(self.path, dtype=self.dtype, mode="r")
        return self._data

    def __len__(self):
        return self._n_tokens - self.context_length - 1

    def __getitem__(self, idx):
        data = self._open()

        x = np.asarray(
            data[idx : idx + self.context_length],
            dtype=np.int64
        )
        y = np.asarray(
            data[idx + 1 : idx + 1 + self.context_length],
            dtype=np.int64
        )

        return torch.from_numpy(x), torch.from_numpy(y)


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
        default=4
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="uint16",
        choices=["uint16", "uint32"]
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
    # Performance settings
    # ==========================
    torch.set_num_threads(args.workers)
    torch.set_num_interop_threads(1)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ==========================
    # Distributed setup
    # ==========================
    ddp = "LOCAL_RANK" in os.environ and torch.cuda.is_available()

    if ddp:
        dist.init_process_group(backend="nccl")

        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main_process = (local_rank == 0)

    if is_main_process:
        print(f"Using device: {device}", flush=True)
        if ddp:
            print(f"Distributed training with {world_size} GPUs", flush=True)

    # ==========================
    # Settings
    # ==========================
    batch_size = 8
    gradient_accumulation_steps = 2
    context_length = 1024
    max_steps = 200_000

    learning_rate = 3e-4
    min_learning_rate = 3e-5
    warmup_steps = 2000
    weight_decay = 0.1

    validation_interval = 500
    checkpoint_interval = 500

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
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    latest_checkpoint = os.path.join(args.checkpoint_dir, "latest.pt")
    best_checkpoint = os.path.join(args.checkpoint_dir, "best.pt")

    input_latest = os.path.join(args.dataset_dir, "checkpoints", "latest.pt")
    input_best = os.path.join(args.dataset_dir, "checkpoints", "best.pt")

    # Bootstrap checkpoint from Kaggle input if needed
    if args.resume:
        if (not os.path.exists(latest_checkpoint)) and os.path.exists(input_latest):
            shutil.copy2(input_latest, latest_checkpoint)

        if (not os.path.exists(best_checkpoint)) and os.path.exists(input_best):
            shutil.copy2(input_best, best_checkpoint)

    # ==========================
    # Dataset
    # ==========================
    if is_main_process:
        print("Loading dataset...", flush=True)

    dtype = np.uint16 if args.dtype == "uint16" else np.uint32

    train_path = os.path.join(args.dataset_dir, "train.bin")
    val_path = os.path.join(args.dataset_dir, "validation.bin")

    if is_main_process:
        print(f"Train file: {train_path}", flush=True)
        print(f"Validation file: {val_path}", flush=True)

    train_dataset = BinTokenDataset(
        train_path,
        context_length,
        dtype=dtype
    )

    val_dataset = BinTokenDataset(
        val_path,
        context_length,
        dtype=dtype
    )

    if ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            drop_last=False
        )
    else:
        train_sampler = None

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        drop_last=True
    )

    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **{**loader_kwargs, "drop_last": False}
    )

    train_iter = iter(train_loader)

    # ==========================
    # Model
    # ==========================
    if is_main_process:
        print("Creating model...", flush=True)

    base_model = GPT(GPTConfig()).to(device)

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
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
            print("Loading checkpoint...", flush=True)

        if os.path.exists(latest_checkpoint):
            step, best_loss, tokens_seen = load_checkpoint(
                latest_checkpoint,
                base_model,
                optimizer,
                scheduler,
                scaler
            )

            if is_main_process:
                print(f"Resumed from step {step}", flush=True)
        elif is_main_process:
            print("No latest.pt found, starting from scratch.", flush=True)

    # ==========================
    # Shutdown
    # ==========================
    running = True

    def shutdown(sig, frame):
        nonlocal running
        if is_main_process:
            print("\nStopping safely...", flush=True)
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ==========================
    # Validation
    # ==========================
    @torch.no_grad()
    def validate():
        model.eval()

        total_loss = 0.0
        count = 0

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

            total_loss += loss.item()
            count += 1

            if count >= 100:
                break

        model.train()
        return total_loss / max(count, 1)

    # ==========================
    # Training
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
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        tokens_seen += (
            batch_size *
            context_length *
            gradient_accumulation_steps *
            world_size
        )

        step += 1

        if is_main_process and step % 10 == 0:
            print(
                f"step {step}/{max_steps} | loss {total_loss:.4f}",
                flush=True
            )

        # ======================
        # Validation + best.pt
        # ======================
        if step % validation_interval == 0:
            if ddp:
                dist.barrier()

            if is_main_process:
                val_loss = validate()
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
                    latest_checkpoint,
                    base_model,
                    optimizer,
                    scheduler,
                    scaler,
                    step,
                    best_loss,
                    tokens_seen
                )
                print("Saved latest.pt", flush=True)

            if ddp:
                dist.barrier()

    # ==========================
    # Exit save
    # ==========================
    if ddp:
        dist.barrier()

    if is_main_process:
        print("Saving latest checkpoint...", flush=True)

        save_checkpoint(
            latest_checkpoint,
            base_model,
            optimizer,
            scheduler,
            scaler,
            step,
            best_loss,
            tokens_seen
        )

        print("Finished!", flush=True)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
