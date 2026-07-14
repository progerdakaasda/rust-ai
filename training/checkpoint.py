import os
import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    step,
    best_loss,
    tokens_seen
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )


    checkpoint = {

        "model":
            model.state_dict(),

        "optimizer":
            optimizer.state_dict(),

        "scheduler":
            scheduler.state_dict()
            if scheduler else None,

        "scaler":
            scaler.state_dict()
            if scaler else None,

        "step":
            step,

        "best_loss":
            best_loss,

        "tokens_seen":
            tokens_seen
    }


    torch.save(
        checkpoint,
        path
    )



def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None
):

    checkpoint = torch.load(
        path,
        map_location="cuda"
    )


    model.load_state_dict(
        checkpoint["model"]
    )


    if optimizer:
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )


    if scheduler and checkpoint["scheduler"]:
        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )


    if scaler and checkpoint["scaler"]:
        scaler.load_state_dict(
            checkpoint["scaler"]
        )


    return (
        checkpoint["step"],
        checkpoint["best_loss"],
        checkpoint["tokens_seen"]
    )
