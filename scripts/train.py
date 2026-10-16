"""Train one configuration of melanoma_cnn/experiments.py.

    python -m scripts.train resnet_ft_layer34
    python -m scripts.train cnn_simple --epochs 20 --tensorboard

Writes results/history/<name>__<date>.json and checkpoints/<name>.pt (final weights).
"""
import argparse
import json
import time

import torch
import torch.nn as nn

from melanoma_cnn.config import (CHECKPOINT_DIR, HISTORY_DIR, NUM_EPOCHS, SEED, TENSORBOARD_DIR,
                                 TRAIN_DIR, VAL_DIR)
from melanoma_cnn.data import make_loader
from melanoma_cnn.engine import fit
from melanoma_cnn.experiments import (EXPERIMENTS, build_model, build_optimizer, build_scheduler,
                                      transforms_for)
from melanoma_cnn.models import count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tensorboard", action="store_true", help="also log to runs/<experiment>")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)  # same initial weights and same shuffling for every run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf, val_tf = transforms_for(args.experiment)
    train_loader = make_loader(TRAIN_DIR, train_tf, shuffle=True)
    val_loader = make_loader(VAL_DIR, val_tf, shuffle=False)
    classes = train_loader.dataset.classes

    model = build_model(args.experiment, len(classes)).to(device)
    optimizer = build_optimizer(args.experiment, model)
    scheduler = build_scheduler(args.experiment, optimizer)
    trainable, total = count_parameters(model)
    print(f"{args.experiment} on {device} | trainable parameters {trainable:,} / {total:,}")

    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(TENSORBOARD_DIR / args.experiment))

    history = fit(model, train_loader, val_loader, nn.CrossEntropyLoss(), optimizer, device,
                  args.epochs, scheduler=scheduler, writer=writer)
    if writer is not None:
        writer.close()

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M")
    record = {"experiment": args.experiment, "seed": args.seed, "classes": classes,
              "trainable_params": trainable, "total_params": total, "history": history}
    history_file = HISTORY_DIR / f"{args.experiment}__{run_id}.json"
    history_file.write_text(json.dumps(record, indent=1), encoding="utf-8")
    torch.save(model.state_dict(), CHECKPOINT_DIR / f"{args.experiment}.pt")
    print(f"History : {history_file} | weights : {CHECKPOINT_DIR / args.experiment}.pt")


if __name__ == "__main__":
    main()
