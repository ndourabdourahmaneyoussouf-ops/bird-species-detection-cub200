# 🐦 Détection automatique d'oiseaux ravageurs — Riziculture, vallée du fleuve Sénégal

Projet final — Certificat en Intelligence Artificielle (sujet conçu par Alla LO)
Version **ingénierie / réutilisable** du notebook : `Projet_Oiseaux_Ravageurs_Engineering.ipynb`

## 1. Objectif

Ce notebook construit un pipeline complet de classification d'images d'oiseaux (contrôle
qualité des données → split stratifié → augmentation → CNN baseline → transfert learning
MobileNetV2 → fine-tuning → évaluation → analyse des erreurs → benchmark d'inférence →
démonstration Gradio → artefacts reproductibles), dans l'optique d'identifier des oiseaux
potentiellement ravageurs pour les cultures de riz de la vallée du fleuve Sénégal.

> **Point scientifique important (rappelé dans le notebook) :** le sous-ensemble de
> CUB-200-2011 utilisé ici contient 10 espèces nord-américaines. Il ne permet pas, à lui
> seul, d'affirmer qu'une espèce est un ravageur du riz au Sénégal. Le statut « ravageur
> potentiel » (`RAVAGEUR_CLASSES` dans la configuration) est donc **volontairement laissé
> vide par défaut** et doit être renseigné uniquement après une annotation experte ou avec
> un jeu de données local validé.

## 2. Contenu du dépôt

| Fichier | Description |
|---|---|
| `Projet_Oiseaux_Ravageurs_Engineering.ipynb` | Notebook Colab complet (18 sections + heatmap normalisée ajoutée) |
| `requirements.txt` | Dépendances Python |
| `README.md` | Ce fichier |
| `rapport_technique_projet_oiseaux.pdf` | Rapport technique (10 pages, résultats réels) |
| `presentation_projet_oiseaux.pptx` | Présentation (8 slides, résultats réels) |

## 3. Prérequis avant l'exécution

Ce notebook est conçu pour tourner sur **Google Colab avec Google Drive monté**, et attend
que le dataset CUB-200-2011 soit **déjà présent dans votre Drive** à l'emplacement suivant :

```
/content/drive/MyDrive/PROJET_OISEAUX_RAVAGEURS/
└── dataset/
    └── CUB_200_2011/
        ├── classes.txt
        └── images/
            ├── 009.Brewer_Blackbird/
            ├── 010.Red_winged_Blackbird/
            ├── 011.Rusty_Blackbird/
            ├── 012.Yellow_headed_Blackbird/
            ├── 113.Baird_Sparrow/
            ├── 114.Black_throated_Sparrow/
            ├── 115.Brewer_Sparrow/
            ├── 116.Chipping_Sparrow/
            ├── 117.Clay_colored_Sparrow/
            └── 118.House_Sparrow/
```

Le dataset complet CUB-200-2011 se télécharge depuis le
[site officiel Caltech/Visipedia](http://www.vision.caltech.edu/datasets/cub_200_2011/) ou
via Kaggle / Hugging Face Datasets ; seuls le fichier `classes.txt` et les 10 dossiers
d'images listés ci-dessus sont nécessaires pour ce notebook (le sous-ensemble peut être
copié tel quel dans l'arborescence Drive attendue).

Les dossiers `models/` et `résultats/` sont créés automatiquement par le notebook au
premier lancement.

## 4. Installation et exécution

### Sur Google Colab (recommandé, seul environnement testé)
1. Uploader `Projet_Oiseaux_Ravageurs_Engineering.ipynb` sur votre Google Drive et l'ouvrir
   avec Google Colaboratory.
2. Activer un runtime GPU : `Exécution > Modifier le type d'exécution > GPU (T4)`.
3. Exécuter la première cellule (installation de `gradio` si absent) puis la cellule
   « REPRODUCTIBILITÉ ET GOOGLE DRIVE », qui demandera l'autorisation d'accéder à votre Drive.
4. Vérifier/adapter la cellule **2. CONFIGURATION CENTRALE** :
   - `RUN_TRAINING = True` pour réentraîner les modèles depuis zéro (sinon le notebook
     s'attend à retrouver un modèle déjà sauvegardé dans `models/`).
   - `RUN_VALID_COMPARISON = True` pour recalculer la comparaison CNN vs MobileNetV2 sur un
     pied d'égalité expérimentale stricte.
   - `RAVAGEUR_CLASSES` : à ne renseigner qu'après validation experte (cf. avertissement
     ci-dessus).
5. Exécuter les cellules dans l'ordre (`Exécution > Tout exécuter`).

### En local (sans Google Drive)
Le notebook dépend de `google.colab.drive` pour le montage du Drive : pour une exécution
locale, remplacer la cellule 3 par un chemin local vers votre copie du dataset et adapter
`BASE_DIR` dans la configuration en conséquence.

```bash
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Projet_Oiseaux_Ravageurs_Engineering.ipynb
```

## 5. Pipeline (18 sections)

1. Environnement et importations
2. Configuration centrale (chemins, hyperparamètres, drapeaux `RUN_TRAINING` / `RUN_VALID_COMPARISON`)
3. Reproductibilité (seed) et montage Google Drive
4. Classes et manifeste des images (10 classes explicites, pas de recherche par mots-clés)
5. Contrôle qualité des images (détection des fichiers corrompus)
6. Split stratifié train / validation / test (70/15/15)
7. Pipeline `tf.data` avec augmentation (flip, rotation, zoom, contraste)
8. Définition des modèles (CNN baseline et MobileNetV2 transfer learning) et fonctions d'entraînement
9. Entraînement ou rechargement du modèle final
10. Courbes d'apprentissage (accuracy / loss)
11. Évaluation finale (accuracy, precision, recall, F1 — pondérés et macro)
12. Matrice de confusion et analyse des erreurs
13. Analyse spécifique des faux négatifs sur les classes « ravageur »
14. Fonction de prédiction réutilisable + test sur le jeu de test
15. Benchmark du temps d'inférence
16. Comparaison expérimentale optionnelle CNN vs MobileNetV2 (à conditions égales)
17. Interface de démonstration Gradio (même fonction de prétraitement que le notebook)
18. Sauvegarde des artefacts (modèle, classes, configuration JSON, CSV de résultats) et synthèse finale

## 6. Modèles

1. **Baseline** : CNN from scratch (4 blocs Conv2D + BatchNorm + MaxPooling, tête dense).
2. **Principal** : MobileNetV2 pré-entraîné (ImageNet), backbone gelé puis fine-tuning des
   dernières couches (`FINE_TUNE_LAYERS = 30`).

## 7. Artefacts produits (dans Google Drive)

- `models/modele_oiseaux_final.keras` — modèle final sauvegardé
- `models/classes.json` — noms des classes
- `models/project_config.json` — configuration complète de l'expérience (dataset, hyperparamètres, métriques, environnement)
- `résultats/classification_report.csv`, `matrice_confusion.csv`, `predictions_test_completes.csv`, `performance_par_classe.csv`, `temps_inference.csv`, `synthese_finale.csv`
- `résultats/distribution_classes.png`, `exemples_augmentation.png`

## 8. Démonstration

La dernière section lance une interface Gradio (`demo.launch(share=True)`, à décommenter)
qui réutilise exactement la même fonction de prétraitement que le reste du notebook, pour
éviter tout écart entre les résultats du notebook et de la démo.

## 9. Limites et considérations éthiques

- Le sous-ensemble de 10 classes CUB-200-2011 est nord-américain : il ne couvre pas les
  espèces réellement présentes dans la vallée du fleuve Sénégal (Quelea quelea, Dendrocygna
  viduata, etc.).
- Le statut « ravageur potentiel » doit être validé par une expertise locale avant tout
  usage réel — voir `RAVAGEUR_CLASSES` dans la configuration.
- Toute image locale collectée sur le terrain doit être anonymisée (pas de personnes,
  habitations ou véhicules identifiables) avant tout usage.
- La validation humaine reste nécessaire avant toute décision de lutte contre les oiseaux.

## 10. Auteur

Projet réalisé dans le cadre du Certificat en Intelligence Artificielle — sujet conçu par
Alla LO.
