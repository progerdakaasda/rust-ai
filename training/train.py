import time
import signal
import argparse
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from model.gpt import GPT
from model.config import GPTConfig

from training.dataset import TokenDataset
from training.checkpoint import save_checkpoint, load_checkpoint
from training.scheduler import cosine_scheduler


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
# Settings
# ==========================

device = "cuda"

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


if args.minutes:
    time_limit = args.minutes * 60


if args.hours:
    time_limit = args.hours * 3600



# ==========================
# Dataset
# ==========================

print("Loading dataset...")


train_dataset = TokenDataset(
    "datasets/train.bin",
    context_length
)


val_dataset = TokenDataset(
    "datasets/validation.bin",
    context_length
)



train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


train_iter = iter(train_loader)



# ==========================
# Model
# ==========================

print("Creating model...")


model = GPT(
    GPTConfig()
)


model.to(device)



parameters = sum(
    p.numel()
    for p in model.parameters()
)


print(
    f"Parameters: {parameters:,}"
)



# ==========================
# Optimizer
# ==========================

optimizer = torch.optim.AdamW(
    model.parameters(),
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



scaler = torch.amp.GradScaler(
    "cuda"
)


loss_fn = nn.CrossEntropyLoss()



# ==========================
# Resume
# ==========================

step = 0

best_loss = float("inf")

tokens_seen = 0



if args.resume:

    print(
        "Loading checkpoint..."
    )


    step, best_loss, tokens_seen = load_checkpoint(
        "checkpoints/latest.pt",
        model,
        optimizer,
        scheduler,
        scaler
    )


    print(
        f"Resumed from step {step}"
    )



# ==========================
# Shutdown
# ==========================

running = True



def shutdown(sig, frame):

    global running

    print(
        "\nStopping safely..."
    )

    running = False



signal.signal(
    signal.SIGINT,
    shutdown
)



# ==========================
# Validation
# ==========================

@torch.no_grad()
def validate():

    model.eval()

    total_loss = 0

    count = 0


    for x, y in val_loader:


        x = x.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )


        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            logits = model(x)


            loss = loss_fn(
                logits.view(
                    -1,
                    logits.size(-1)
                ),
                y.view(-1)
            )


        total_loss += loss.item()

        count += 1


        if count >= 100:
            break



    model.train()


    return total_loss / count



# ==========================
# Training
# ==========================

model.train()



while step < max_steps and running:


    if time_limit:

        if time.time() - start_time > time_limit:

            print(
                "Time limit reached"
            )

            break



    optimizer.zero_grad()



    total_loss = 0



    for _ in range(
        gradient_accumulation_steps
    ):


        try:

            x, y = next(train_iter)


        except StopIteration:

            train_iter = iter(train_loader)

            x, y = next(train_iter)



        x = x.to(
            device,
            non_blocking=True
        )


        y = y.to(
            device,
            non_blocking=True
        )



        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):


            logits = model(x)


            loss = loss_fn(
                logits.view(
                    -1,
                    logits.size(-1)
                ),
                y.view(-1)
            )


            loss = loss / gradient_accumulation_steps



        scaler.scale(loss).backward()


        total_loss += loss.item()



    scaler.unscale_(
        optimizer
    )


    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )


    scaler.step(
        optimizer
    )


    scaler.update()


    scheduler.step()



    tokens_seen += (
        batch_size *
        context_length *
        gradient_accumulation_steps
    )


    step += 1



    if step % 10 == 0:

        print(
            f"step {step}/{max_steps} | "
            f"loss {total_loss:.4f}"
        )



    # ======================
    # Validation + best.pt
    # ======================

    if step % validation_interval == 0:


        val_loss = validate()


        print(
            f"Validation loss: {val_loss:.4f}"
        )



        if val_loss < best_loss:


            best_loss = val_loss


            print(
                "⭐ New best model!"
            )


            save_checkpoint(
                "checkpoints/best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                step,
                best_loss,
                tokens_seen
            )



    # ======================
    # latest.pt
    # ======================

    if step % checkpoint_interval == 0:


        save_checkpoint(
            "checkpoints/latest.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            step,
            best_loss,
            tokens_seen
        )


        print(
            "Saved latest.pt"
        )



# ==========================
# Exit save
# ==========================

print(
    "Saving latest checkpoint..."
)



save_checkpoint(
    "checkpoints/latest.pt",
    model,
    optimizer,
    scheduler,
    scaler,
    step,
    best_loss,
    tokens_seen
)



print(
    "Finished!"
)
