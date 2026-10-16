"""Write the recorded runs (results/history/*.json) as TensorBoard logs, without retraining.

    python -m scripts.export_tensorboard
    tensorboard --logdir runs          # then open http://localhost:6006

The curves are replayed from the console logs of May 2026. New runs can log directly with
`python -m scripts.train <experiment> --tensorboard`.
"""
from torch.utils.tensorboard import SummaryWriter

from melanoma_cnn.config import TENSORBOARD_DIR
from melanoma_cnn.engine import log_epoch
from scripts.make_figures import load_histories
from scripts.import_logs import load_manifest


def main():
    manifest = load_manifest()
    histories = load_histories(manifest)
    for order, run in enumerate(r for r in manifest["runs"] if r["reference"]):
        history = histories[(run["experiment"], run["trial"])]
        writer = SummaryWriter(str(TENSORBOARD_DIR / f"{order + 1:02d}_{run['experiment']}"))
        for epoch, values in enumerate(zip(history["train_loss"], history["val_loss"],
                                           history["train_acc"], history["val_acc"]), start=1):
            log_epoch(writer, epoch, *values)
        writer.close()
        print(f"runs/{order + 1:02d}_{run['experiment']}")


if __name__ == "__main__":
    main()
