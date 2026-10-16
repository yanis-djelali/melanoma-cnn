"""Results table and comparison figures, computed from results/history/*.json.

    python -m scripts.import_logs   # once, raw logs -> JSON
    python -m scripts.make_figures

Writes results/summary.md and figures/generated/*.png. Does not need torch nor the dataset.
"""
import json

from melanoma_cnn import plots
from melanoma_cnn.config import FIGURES_DIR, RESULTS_DIR
from melanoma_cnn.logs import summarize
from scripts.import_logs import history_path, load_manifest

OUT_DIR = FIGURES_DIR / "generated"


def load_histories(manifest):
    """{(experiment, trial): history} for every run of the manifest."""
    return {
        (run["experiment"], run["trial"]):
            json.loads(history_path(run).read_text(encoding="utf-8"))["history"]
        for run in manifest["runs"]
    }


def pct(value):
    return f"{100 * value + 1e-9:.1f} %"  # epsilon: 0.9275 is stored as 0.92749999...


def reference_table(manifest, histories):
    lines = [
        "| Configuration | Run | Meilleure acc. val (epoch) | Acc. val finale | Moyenne 5 dernieres | "
        "Acc. train finale | Parametres entrainables |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in manifest["runs"]:
        if not run["reference"]:
            continue
        info = manifest["experiments"][run["experiment"]]
        s = summarize(histories[(run["experiment"], run["trial"])])
        params = f"{info['trainable_params']:,}".replace(",", " ")
        lines.append(
            f"| {info['label']} | {run['log'][:5]} | {pct(s['best_val_acc'])} ({s['best_epoch']}) | "
            f"{pct(s['final_val_acc'])} | {pct(s['mean_last_val_acc'])} | "
            f"{pct(s['final_train_acc'])} | {params} |"
        )
    return lines


def trials_table(manifest, histories):
    """Same configuration trained several times: how much do the numbers move?"""
    lines = ["| Configuration | Essai | Run | Meilleure acc. val | Acc. val finale |",
             "|---|---|---|---|---|"]
    by_experiment = {}
    for run in manifest["runs"]:
        by_experiment.setdefault(run["experiment"], []).append(run)
    for experiment, runs in by_experiment.items():
        if len(runs) < 2:
            continue
        for run in runs:
            s = summarize(histories[(experiment, run["trial"])])
            label = manifest["experiments"][experiment]["label"]
            lines.append(f"| {label} | {run['trial']} | {run['log'][:5]} | "
                         f"{pct(s['best_val_acc'])} | {pct(s['final_val_acc'])} |")
    return lines


def confusion_table(confusion):
    (tn, fp), (fn, tp) = confusion["matrix"]
    n = tn + fp + fn + tp
    return [
        f"Source : {confusion['source']} ({confusion['model']}).",
        "",
        "| | Predit Benign | Predit Malignant |", "|---|---|---|",
        f"| Vrai Benign | {tn} | {fp} |", f"| Vrai Malignant | {fn} | {tp} |",
        "",
        f"- Accuracy : {pct((tn + tp) / n)}",
        f"- Rappel Malignant (sensibilite) : {pct(tp / (tp + fn))}, "
        f"soit {fn} lesions malignes non detectees",
        f"- Precision Malignant : {pct(tp / (tp + fp))}",
        f"- Rappel Benign (specificite) : {pct(tn / (tn + fp))}",
    ]


def write_summary(manifest, histories):
    text = ["# Resultats (generes depuis les logs bruts par scripts/make_figures.py)", "",
            "## Runs de reference (tableau de la soutenance)", "",
            *reference_table(manifest, histories), "",
            "La 'meilleure acc. val' est le maximum sur les 20 epochs, choisi sur le jeu de test "
            "utilise comme validation : elle est optimiste. "
            "L'accuracy finale est moins biaisee.", "",
            "## Variabilite entre essais d'une meme configuration", "",
            *trials_table(manifest, histories), "",
            "## Matrice de confusion", "", *confusion_table(manifest["confusion_matrix"]), ""]
    path = RESULTS_DIR / "summary.md"
    path.write_text("\n".join(text), encoding="utf-8")
    print(f"Tableau : {path}")


def make_figures(manifest, histories):
    h = histories
    label = {name: info["label"] for name, info in manifest["experiments"].items()}
    plots.learning_curves(h[("cnn_simple", 3)], "CNN simple", OUT_DIR / "courbes_cnn_simple.png")
    plots.learning_curves(h[("resnet_frozen", 1)], "ResNet18 gele",
                          OUT_DIR / "courbes_resnet_gele.png")
    plots.learning_curves(h[("resnet_ft_layer34", 1)], "ResNet18 FT layer3 + layer4",
                          OUT_DIR / "courbes_resnet_ft_layer34.png")
    pipelines = {label[e]: h[(e, t)] for e, t in
                 (("cnn_simple", 3), ("cnn_norm", 2), ("cnn_norm_aug", 1))}
    plots.compare_runs(pipelines, "val_acc", "Accuracy de validation selon le pretraitement",
                       OUT_DIR / "comparaison_pretraitements_acc.png", ylim=(0.8, 1.0))
    plots.compare_runs(pipelines, "val_loss", "Loss de validation selon le pretraitement",
                       OUT_DIR / "comparaison_pretraitements_loss.png")
    optimizers = {"Adam lr=1e-3": h[("cnn_norm_aug", 2)], "Adam lr=1e-4": h[("cnn_adam_lr1e-4", 1)],
                  "SGD lr=1e-2, momentum 0.9": h[("cnn_sgd", 1)]}
    plots.compare_runs(optimizers, "val_acc", "Optimiseurs (run 04, pipeline norm + augmentation)",
                       OUT_DIR / "comparaison_optimiseurs.png", ylim=(0.75, 1.0))
    scheduler = {"LR fixe 1e-3": h[("cnn_norm_aug", 1)], "StepLR": h[("cnn_steplr", 1)]}
    plots.compare_runs(scheduler, "val_acc", "Avec et sans scheduler",
                       OUT_DIR / "comparaison_scheduler.png", ylim=(0.75, 1.0))
    plots.learning_rate(h[("cnn_steplr", 1)]["lr"], OUT_DIR / "evolution_lr.png")
    transfer = {label[e]: h[(e, t)] for e, t in (("cnn_norm_aug", 1), ("resnet_frozen", 1),
                                                 ("resnet_ft_layer4", 1), ("resnet_ft_layer34", 1))}
    plots.compare_runs(transfer, "val_acc", "CNN maison et ResNet18",
                       OUT_DIR / "comparaison_finetuning.png", ylim=(0.8, 1.0))
    confusion = manifest["confusion_matrix"]
    plots.confusion_matrix(confusion["matrix"], confusion["classes"],
                           "ResNet18 FT layer4 (run 09)", OUT_DIR / "matrice_confusion.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    histories = load_histories(manifest)
    write_summary(manifest, histories)
    make_figures(manifest, histories)


if __name__ == "__main__":
    main()
