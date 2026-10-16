"""Convert the raw console logs into one JSON history per run.

    python -m scripts.import_logs

Reads results/runs.json, writes results/history/<experiment>__trial<k>.json.
Does not need torch nor the dataset.
"""
import json

from melanoma_cnn.config import HISTORY_DIR, RESULTS_DIR
from melanoma_cnn.logs import read_log

RAW_LOGS_DIR = RESULTS_DIR / "raw_logs"


def load_manifest():
    return json.loads((RESULTS_DIR / "runs.json").read_text(encoding="utf-8"))


def history_path(run):
    return HISTORY_DIR / f"{run['experiment']}__trial{run['trial']}.json"


def main():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    cache = {}
    for run in load_manifest()["runs"]:
        if run["log"] not in cache:
            cache[run["log"]] = read_log(RAW_LOGS_DIR / run["log"])
        history = cache[run["log"]][run["segment"]]
        record = {"source": run["log"], "segment": run["segment"], "history": history}
        history_path(run).write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"{history_path(run)} : {len(history['val_acc'])} epochs")


if __name__ == "__main__":
    main()
