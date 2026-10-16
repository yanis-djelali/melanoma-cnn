"""Checks on the recorded results. No torch, no dataset needed.

    python -m unittest discover tests
"""
import ast
import unittest
from pathlib import Path

from melanoma_cnn.logs import parse_epoch_line, read_log, summarize
from scripts.import_logs import RAW_LOGS_DIR, load_manifest

ROOT = Path(__file__).resolve().parent.parent

# Best validation accuracies announced in the oral presentation (May 2026).
ANNOUNCED_BEST = {
    "cnn_simple": 0.936, "cnn_norm": 0.945, "cnn_norm_aug": 0.925, "cnn_steplr": 0.925,
    "resnet_frozen": 0.886, "resnet_ft_layer4": 0.942, "resnet_ft_layer34": 0.954,
}

# Variable names of original/main.py whose values were typed in by hand -> run they come from.
HARDCODED_IN_ORIGINAL = {
    "history": ("cnn_simple", 3), "history_norm": ("cnn_norm", 2),
    "history_aug": ("cnn_norm_aug", 1),
    "history_sched": ("cnn_steplr", 1), "history_resnet": ("resnet_frozen", 1),
    "history_ft": ("resnet_ft_layer4", 1), "history_ft_l3": ("resnet_ft_layer34", 1),
}


def manifest_histories():
    histories = {}
    for run in load_manifest()["runs"]:
        segments = read_log(RAW_LOGS_DIR / run["log"])
        histories[(run["experiment"], run["trial"])] = segments[run["segment"]]
    return histories


def hardcoded_lists(path):
    """{(variable, key): list} for every `variable["key"] = [numbers]` in a Python file."""
    found = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List) and node.value.elts:
            target = node.targets[0]
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                key = ast.literal_eval(target.slice)  # Python >= 3.9
                found[(target.value.id, key)] = ast.literal_eval(node.value)
    return found


class ParseLineTest(unittest.TestCase):
    def test_full_line(self):
        epoch, values = parse_epoch_line(
            "Epoch   8/20 | Loss train 0.2678 | Loss val 0.2532 | Acc train 0.884 | "
            "Acc val 0.914 | LR 1.00e-04 | 10.9s")
        self.assertEqual(epoch, 8)
        self.assertEqual(values, {"train_loss": 0.2678, "val_loss": 0.2532, "train_acc": 0.884,
                                  "val_acc": 0.914, "lr": 1e-4, "seconds": 10.9})

    def test_short_line_and_noise(self):
        self.assertEqual(parse_epoch_line("Epoch  13/20 | Acc val 0.951"), (13, {"val_acc": 0.951}))
        self.assertIsNone(parse_epoch_line(">>> Figure sauvegardee : erreurs.png"))


class RecordedResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.histories = manifest_histories()

    def test_every_run_has_20_epochs(self):
        for name, history in self.histories.items():
            for key, values in history.items():
                self.assertEqual(len(values), 20, f"{name} {key}")

    def test_reference_runs_match_the_presentation(self):
        for run in load_manifest()["runs"]:
            if run["reference"] and run["experiment"] in ANNOUNCED_BEST:
                best = summarize(self.histories[(run["experiment"], run["trial"])])["best_val_acc"]
                self.assertAlmostEqual(best, ANNOUNCED_BEST[run["experiment"]], places=3)

    def test_hardcoded_values_of_original_main_come_from_the_logs(self):
        lists = hardcoded_lists(ROOT / "original" / "main.py")
        for variable, run in HARDCODED_IN_ORIGINAL.items():
            for key in ("train_loss", "val_loss", "train_acc", "val_acc"):
                self.assertEqual(lists[(variable, key)], self.histories[run][key],
                                 f"{variable} {key}")

    def test_confusion_matrix_is_consistent(self):
        matrix = load_manifest()["confusion_matrix"]["matrix"]
        # the validation set has 1000 images per class
        self.assertEqual([sum(row) for row in matrix], [1000, 1000])
        accuracy = (matrix[0][0] + matrix[1][1]) / 2000
        final_acc_run09 = self.histories[("resnet_ft_layer4", 3)]["val_acc"][-1]
        self.assertAlmostEqual(accuracy, final_acc_run09, places=3)


if __name__ == "__main__":
    unittest.main()
