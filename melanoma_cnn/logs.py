"""Read the console logs printed during training and turn them into histories.

A history is a dict of lists, one value per epoch:
    {"train_loss": [...], "val_loss": [...], "train_acc": [...], "val_acc": [...],
     "lr": [...], "seconds": [...]}
Keys that a log does not contain are simply absent (e.g. run 09 only printed val accuracy).

This module does not import torch, so results can be inspected on any machine.
"""
import re
from pathlib import Path

EPOCH_LINE = re.compile(r"^Epoch\s+(\d+)/(\d+)\s*\|(.*)$")

FIELDS = {
    "Loss train": "train_loss",
    "Loss val": "val_loss",
    "Acc train": "train_acc",
    "Acc val": "val_acc",
    "LR": "lr",
}


def parse_epoch_line(line):
    """Return (epoch, {key: value}) for an 'Epoch k/n | ...' line, or None."""
    match = EPOCH_LINE.match(line.strip())
    if match is None:
        return None
    epoch = int(match.group(1))
    values = {}
    for part in match.group(3).split("|"):
        part = part.strip()
        if part.endswith("s") and part[:-1].replace(".", "").isdigit():
            values["seconds"] = float(part[:-1])
            continue
        for label, key in FIELDS.items():
            if part.startswith(label + " "):
                values[key] = float(part[len(label) + 1:])
    return epoch, values


def split_runs(lines):
    """Split a log into training runs: a new run starts at every 'Epoch 1/...' line."""
    runs = []
    for line in lines:
        parsed = parse_epoch_line(line)
        if parsed is None:
            continue
        epoch, values = parsed
        if epoch == 1:
            runs.append({})
        history = runs[-1]
        for key, value in values.items():
            history.setdefault(key, []).append(value)
    return runs


def read_log(path):
    """Parse a log file and return the list of histories it contains (comment lines ignored)."""
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines()
             if not line.startswith("#")]
    return split_runs(lines)


def summarize(history, last_k=5):
    """Key numbers of a run, computed on the validation accuracy curve."""
    val_acc = history["val_acc"]
    best = max(val_acc)
    summary = {
        "epochs": len(val_acc),
        "best_val_acc": best,
        "best_epoch": val_acc.index(best) + 1,
        "final_val_acc": val_acc[-1],
        "mean_last_val_acc": sum(val_acc[-last_k:]) / len(val_acc[-last_k:]),
    }
    if "train_acc" in history:
        summary["final_train_acc"] = history["train_acc"][-1]
    return summary
