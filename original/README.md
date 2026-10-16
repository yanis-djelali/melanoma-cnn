# Code d'origine (v1, avril-mai 2026)

Fichiers tels qu'ils ont été exécutés sur la machine GPU de l'école pour la soutenance, sans aucune modification. Ils sont conservés pour la traçabilité ; le code à utiliser est dans `melanoma_cnn/` et `scripts/` (v2).

Utilisation d'origine : placer le dossier `melanoma-cancer-dataset/` à côté de ces fichiers, puis `python main.py`.

Points à connaître avant de le lire :

- `main.py` enchaîne toutes les parties du projet. Les drapeaux `LANCER_TRAIN_PARTIE_X` / `LANCER_PARTIE_X` valent `False` : dans cet état, les entraînements des parties 5 à 12 ne sont pas relancés et leurs historiques sont recopiés en dur dans des blocs `else`. Le test `tests/test_logs.py` vérifie que ces valeurs sont identiques aux sorties console réelles (`results/raw_logs/`).
- Seule la partie 13 est relancée (environ 15 minutes de GPU).
- Les défauts relevés sont listés dans le `CHANGELOG.md` (section v2, rubrique Corrigé).
