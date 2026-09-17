# Détection automatique d’oiseaux ravageurs pour les cultures de riz

**Concepteur : Alla LO**  
**Certificat : Intelligence Artificielle**  
**Domaine : Vision par ordinateur / Classification d’images**

## 1. Présentation

Ce projet développe un prototype d’intelligence artificielle capable de classifier automatiquement des images d’oiseaux. L’objectif applicatif est de contribuer, à terme, à la surveillance des oiseaux potentiellement nuisibles aux cultures de riz dans la vallée du fleuve Sénégal.

La version pédagogique réalisée utilise le dataset **CUB-200-2011** et un sous-ensemble de **10 classes**. Le modèle principal est **MobileNetV2**, utilisé avec **Transfer Learning puis Fine-Tuning**.

> **Important :** le modèle actuel est un classifieur multi-classes d’espèces d’oiseaux. Il ne constitue pas encore un système de détection terrain validé dans la vallée du fleuve Sénégal. Le statut « ravageur » doit être défini par une annotation agronomique validée, et non déduit automatiquement du nom d’une espèce.

## 2. Objectifs

- préparer et contrôler un jeu de données d’images ;
- construire un CNN baseline ;
- exploiter MobileNetV2 pré-entraîné ;
- réaliser un fine-tuning léger ;
- comparer les performances ;
- analyser les erreurs et les classes difficiles ;
- mesurer le temps d’inférence ;
- fournir une fonction de prédiction réutilisable ;
- proposer une démonstration Gradio ;
- préparer des artefacts reproductibles pour une future intégration applicative.

## 3. Données

Dataset principal : **Caltech-UCSD Birds-200-2011 (CUB-200-2011)**.

Sous-ensemble utilisé :
- 10 classes ;
- 583 images valides ;
- Train : 408 images ;
- Validation : 87 images ;
- Test : 88 images ;
- taille d’entrée : 224 × 224 × 3.

Les classes utilisées dans l’expérience sont :

1. 009.Brewer_Blackbird
2. 010.Red_winged_Blackbird
3. 011.Rusty_Blackbird
4. 012.Yellow_headed_Blackbird
5. 113.Baird_Sparrow
6. 114.Black_throated_Sparrow
7. 115.Brewer_Sparrow
8. 116.Chipping_Sparrow
9. 117.Clay_colored_Sparrow
10. 118.House_Sparrow

## 4. Pipeline

```text
CUB-200-2011
      ↓
Sélection des classes
      ↓
Contrôle des images
      ↓
Split stratifié Train / Validation / Test
      ↓
Redimensionnement 224×224
      ↓
Augmentation du train
      ↓
CNN baseline
      ↓
MobileNetV2 Transfer Learning
      ↓
MobileNetV2 Fine-Tuning
      ↓
Évaluation
      ↓
Analyse des erreurs
      ↓
Prédiction réutilisable
      ↓
Gradio
```

## 5. Modélisation

### CNN baseline
Un réseau convolutionnel simple entraîné depuis zéro sert de référence.

### MobileNetV2
MobileNetV2 est utilisé comme extracteur de caractéristiques pré-entraîné, puis adapté au problème.

### Fine-Tuning
Une partie des dernières couches du backbone est débloquée avec un faible taux d’apprentissage afin d’adapter les représentations aux classes du projet.

Paramètres principaux :
- image : 224 × 224 ;
- batch : 32 ;
- seed : 42 ;
- normalisation : 1./255 ;
- augmentation : RandomFlip, RandomRotation, RandomZoom ;
- fine-tuning : léger ;
- optimisation : Adam.

## 6. Résultats

Le modèle final MobileNetV2 Fine-Tuning obtient :

| Métrique | Résultat |
|---|---:|
| Accuracy | 72,73 % |
| Precision weighted | 74,18 % |
| Recall weighted | 72,73 % |
| F1-score weighted | 73,09 % |

Le jeu de test contient 88 images, dont 64 correctement classées et 24 mal classées.

La confiance moyenne rapportée est de 72,54 %. Elle est de 78,72 % pour les prédictions correctes contre 56,07 % pour les prédictions incorrectes.

## 7. Analyse des erreurs

Les performances sont hétérogènes selon les classes.

Meilleure performance :
- Yellow-headed Blackbird : 100 %.

Classe la plus difficile :
- Chipping Sparrow : 44,44 % d’accuracy, avec 5 erreurs sur 9 images de test.

Autres classes relativement difficiles :
- Baird Sparrow : 57,14 % ;
- Brewer Sparrow : 66,67 % ;
- Clay-colored Sparrow : 66,67 % ;
- House Sparrow : 66,67 %.

Les erreurs sont cohérentes avec la difficulté d’une classification fine entre espèces visuellement proches.

## 8. Faux négatifs

Le cahier des charges donne une priorité au recall des oiseaux ravageurs. Dans la version actuelle, aucune liste agronomique fiable de classes ravageuses n’est codée de manière définitive.

Par conséquent, le notebook évite de présenter arbitrairement une espèce CUB comme « ravageur ».

Pour une version terrain, il faudra :
1. définir les espèces/catégories ravageuses avec un expert ;
2. ajouter une annotation `ravageur_potentiel` ;
3. calculer la matrice de confusion binaire ;
4. calculer le recall, la precision, le F1 et le taux de faux négatifs spécifiquement pour cette catégorie ;
5. calibrer éventuellement un seuil d’alerte.

## 9. Démonstration

L’interface Gradio accepte une image et renvoie :
- la classe prédite ;
- un score de confiance ;
- le Top-5 des prédictions ;
- un niveau de confiance ;
- une interprétation prudente du résultat.

Le notebook contient également un test de rechargement du modèle sauvegardé.

## 10. Reproductibilité

Le projet centralise :
- les chemins ;
- les paramètres ;
- la seed ;
- les noms de classes ;
- le modèle final ;
- les résultats.

Le modèle final est sauvegardé au format Keras `.keras`, accompagné des informations nécessaires à son utilisation.

## 11. Arborescence recommandée

```text
PROJET_OISEAUX_RAVAGEURS/
├── dataset/
│   ├── CUB_200_2011/
│   └── bird_classification/
├── models/
│   ├── modele_oiseaux_mobilenetv2_final.keras
│   ├── classes.json
│   └── configuration.json
├── résultats/
│   ├── comparaison_modeles.csv
│   ├── classification_report.csv
│   ├── matrice_confusion.png
│   ├── analyse_erreurs.png
│   └── temps_inference.csv
├── notebook/
│   └── Projet_Oiseaux_Ravageurs_Engineering.ipynb
├── requirements.txt
└── README.md
```

## 12. Installation

```bash
pip install -r requirements.txt
```

Le notebook est prévu pour Google Colab. Le dataset et les modèles peuvent être stockés dans Google Drive.

## 13. Limites

- CUB-200-2011 n’est pas un dataset local de la vallée du fleuve Sénégal.
- Le système actuel est un classifieur d’espèces, pas un détecteur d’objets.
- Une image contenant plusieurs oiseaux n’est pas explicitement traitée comme une scène de détection.
- Les performances sur des images terrain peuvent être différentes.
- Les scores de softmax ne doivent pas être interprétés comme une probabilité parfaitement calibrée.
- Une validation humaine reste nécessaire avant toute décision agricole.

## 14. Perspectives

- constitution d’un dataset local sénégalais ;
- annotation par espèce et par statut ravageur/non ravageur ;
- passage à la classification binaire pour l’alerte ;
- détection d’objets avec YOLO ou modèle léger équivalent ;
- calibration des seuils ;
- augmentation du nombre d’images ;
- tests sur différentes conditions de lumière et distances ;
- déploiement sur Hugging Face Spaces ou appareil léger ;
- intégration future dans une application mobile ou un dispositif IoT.

## 15. Auteur

**Alla LO** — Certificat en Intelligence Artificielle
