import math
import torch


def cosine_scheduler(
    optimizer,
    warmup_steps,
    max_steps,
    min_lr,
    base_lr
):

    def lr_lambda(step):

        if step < warmup_steps:
            return (
                step + 1
            ) / warmup_steps


        progress = (
            step - warmup_steps
        ) / (
            max_steps - warmup_steps
        )


        progress = min(
            max(progress, 0.0),
            1.0
        )


        cosine = 0.5 * (
            1 + math.cos(math.pi * progress)
        )


        min_ratio = min_lr / base_lr


        return (
            min_ratio
            +
            (1 - min_ratio)
            *
            cosine
        )


    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda
    )
