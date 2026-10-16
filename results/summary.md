# Resultats (generes depuis les logs bruts par scripts/make_figures.py)

## Runs de reference (tableau de la soutenance)

| Configuration | Run | Meilleure acc. val (epoch) | Acc. val finale | Moyenne 5 dernieres | Acc. train finale | Parametres entrainables |
|---|---|---|---|---|---|---|
| CNN simple | run03 | 93.6 % (20) | 93.6 % | 92.7 % | 94.5 % | 12 869 410 |
| CNN + normalisation | run03 | 94.5 % (14) | 94.4 % | 93.9 % | 98.9 % | 12 869 410 |
| CNN + norm + augmentation (Adam 1e-3) | run03 | 92.5 % (18) | 92.5 % | 91.0 % | 90.0 % | 12 869 410 |
| CNN + norm + aug, Adam 1e-4 | run04 | 92.7 % (18) | 91.6 % | 91.8 % | 89.8 % | 12 869 410 |
| CNN + norm + aug, SGD 1e-2 momentum 0.9 | run04 | 91.7 % (19) | 88.9 % | 88.2 % | 88.8 % | 12 869 410 |
| CNN + norm + aug, Adam 1e-3 + StepLR | run05 | 92.5 % (14) | 92.1 % | 92.2 % | 90.0 % | 12 869 410 |
| ResNet18 gele | run06 | 88.6 % (13) | 88.0 % | 88.1 % | 86.1 % | 1 026 |
| ResNet18 fine-tuning layer4 | run07 | 94.2 % (19) | 94.1 % | 93.7 % | 98.4 % | 8 394 754 |
| ResNet18 fine-tuning layer3 + layer4 | run08 | 95.4 % (8) | 94.7 % | 94.8 % | 98.6 % | 10 494 466 |

La 'meilleure acc. val' est le maximum sur les 20 epochs, choisi sur le jeu de test utilise comme validation : elle est optimiste. L'accuracy finale est moins biaisee.

## Variabilite entre essais d'une meme configuration

| Configuration | Essai | Run | Meilleure acc. val | Acc. val finale |
|---|---|---|---|---|
| CNN simple | 1 | run01 | 92.8 % | 91.6 % |
| CNN simple | 2 | run02 | 93.6 % | 93.6 % |
| CNN simple | 3 | run03 | 93.6 % | 93.6 % |
| CNN + normalisation | 1 | run02 | 95.0 % | 94.7 % |
| CNN + normalisation | 2 | run03 | 94.5 % | 94.4 % |
| CNN + norm + augmentation (Adam 1e-3) | 1 | run03 | 92.5 % | 92.5 % |
| CNN + norm + augmentation (Adam 1e-3) | 2 | run04 | 90.5 % | 90.4 % |
| ResNet18 fine-tuning layer4 | 1 | run07 | 94.2 % | 94.1 % |
| ResNet18 fine-tuning layer4 | 2 | run08 | 94.5 % | 94.2 % |
| ResNet18 fine-tuning layer4 | 3 | run09 | 95.1 % | 94.0 % |

## Matrice de confusion

Source : run09_analyse_erreurs.log (resnet_ft_layer4, essai 3, poids de fin d'epoch 20).

| | Predit Benign | Predit Malignant |
|---|---|---|
| Vrai Benign | 947 | 53 |
| Vrai Malignant | 68 | 932 |

- Accuracy : 94.0 %
- Rappel Malignant (sensibilite) : 93.2 %, soit 68 lesions malignes non detectees
- Precision Malignant : 94.6 %
- Rappel Benign (specificite) : 94.7 %
