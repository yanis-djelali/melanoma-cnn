# Changelog

## v2 (octobre 2026) : finalisation et réorganisation

Réalisée après la soutenance, avec l'aide d'un assistant IA. Le code v1 est conservé sans modification dans `original/`.

### Ajouté : éléments de l'énoncé non terminés en mai 2026

- Grille d'exemples de 12 images (6 par classe) au lieu de 4 : `scripts/explore_data.py`.
- Visualisation d'un batch de validation (`batch_val.png`), demandée en partie 3 : `scripts/explore_data.py`.
- Résumé du modèle (architecture, forme du tenseur après chaque bloc, paramètres de chaque configuration) : `scripts/model_summary.py`.
- TensorBoard (partie 10, bonus) : option `--tensorboard` de `scripts/train.py`, et `scripts/export_tensorboard.py` qui rejoue les runs de mai 2026 depuis leurs logs.
- Matrice de confusion du meilleur modèle : `scripts/train.py` sauvegarde les poids (`checkpoints/<config>.pt`) et `scripts/analyze_errors.py` les recharge. En v1, faute de sauvegarde, l'analyse portait sur un ResNet layer4 ré-entraîné et non sur le meilleur modèle (layer3 + layer4).
- Démonstration sur une image : `scripts/predict.py`.

### Corrigé

- **Historiques recopiés à la main** dans les blocs `else` de `main.py` : remplacés par les sorties console réelles (`results/raw_logs/`), converties en JSON par `scripts/import_logs.py`. Le test `test_hardcoded_values_of_original_main_come_from_the_logs` confirme que les valeurs v1 étaient exactes.
- **Exemples d'erreurs biaisés** : la v1 prenait les 8 premières erreurs d'un jeu trié par classe, donc uniquement des Benign prédits Malignant ; aucun faux négatif n'était montré. La v2 tire les exemples au hasard (graine fixe) dans chaque classe.
- **Reproductibilité** : graine fixée (`torch.manual_seed`) et fichiers triés dans le Dataset (l'ordre de `os.listdir` dépend du système de fichiers).
- **Moyenne et écart-type** : la v1 moyennait les écarts-types par batch, ce qui sous-estime légèrement l'écart-type global ; `scripts/compute_stats.py` affiche les deux calculs.
- **Courbes** : la longueur de l'axe des epochs est déduite des données (erreur « x and y must have same first dimension » rencontrée en v1).
- **Figures** : backend matplotlib sans affichage, pour éviter les erreurs X11 sur machine distante.

### Modifié (qualité du code, logique d'apprentissage inchangée)

- Un fichier par rôle (`melanoma_cnn/`), scripts courts (`scripts/`), plus de script unique de 947 lignes.
- Les 9 configurations sont décrites dans `melanoma_cnn/experiments.py` au lieu de 9 boucles copiées-collées.
- Les trois fonctions de chargement du ResNet sont remplacées par `resnet18_classifier(num_classes, trainable_blocks)`.
- Les métriques et le format d'affichage des epochs sont identiques à la v1 ; une colonne LR est ajoutée.
- Imports inutilisés et commentaires obsolètes supprimés ; noms en anglais dans le code.
- Ajout de `requirements.txt`, `.gitignore`, `ruff.toml` et de tests (`python -m unittest discover tests`).

### Non modifié volontairement

- Hyperparamètres, transformations, architectures et learning rates identiques à la v1, pour que les résultats restent comparables aux logs de mai 2026.
- Les limites méthodologiques (pas de validation séparée du test, BatchNorm du ResNet « gelé » en mode train, normalisation du dataset utilisée pour le ResNet) sont documentées dans le README mais pas corrigées : les corriger changerait les expériences.

## v1 (avril-mai 2026) : version de la soutenance

- Pipeline complet en 13 parties dans `main.py` (exploration, Dataset/DataLoader, CNN, normalisation, augmentation, optimiseurs, scheduler, ResNet18 gelé et fine-tuné, matrice de confusion).
- 15 entraînements de 20 epochs sur GPU, 18 figures, exposé oral de 10 minutes.
