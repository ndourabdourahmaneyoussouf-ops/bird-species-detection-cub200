# Détection automatique d’oiseaux ravageurs pour les cultures de riz

**Concepteur : Alla LO**  
**Certificat : Intelligence Artificielle**  
**Domaine : Vision par ordinateur / Classification d’images**

## 1. Présentation

Ce projet développe un prototype d’intelligence artificielle capable de classifier automatiquement des images d’oiseaux. L’objectif applicatif est de contribuer, à terme, à la surveillance des oiseaux potentiellement nuisibles aux cultures de riz dans la vallée du fleuve Sénégal.

La version réalisée utilise le dataset **CUB-200-2011** et un sous-ensemble de **10 classes**. Le modèle principal repose sur **MobileNetV2**, exploité avec **Transfer Learning puis Fine-Tuning**.

> **Important :** Le sous-ensemble CUB utilisé dans l'expérience actuelle contient 10 espèces. Il ne permet pas, à lui seul, d'affirmer qu'une espèce est un ravageur du riz dans la vallée du fleuve Sénégal. Le statut « ravageur potentiel » est configurable et doit être validé par une annotation experte ou un jeu de données local.

## 2. Objectifs

- Préparer, nettoyer et contrôler la qualité du jeu de données ;
- Mettre en place un split stratifié reproductible (Train / Validation / Test) ;
- Construire et entraîner un CNN baseline à partir de zéro (*from scratch*) ;
- Exploiter MobileNetV2 pré-entraîné sur ImageNet pour du Transfer Learning ;
- Réaliser un Fine-Tuning ciblé sur les 30 dernières couches du réseau ;
- Comparer rigoureusement les performances des modèles ;
- Analyser les erreurs, le taux de faux négatifs et la performance par classe ;
- Mesurer le temps d’inférence moyen par image (benchmark) ;
- Fournir une fonction de prédiction réutilisable pour de nouvelles images ;
- Proposer une démonstration interactive avec Gradio ;
- Sauvegarder des artefacts reproductibles et un fichier de configuration pour une future intégration.

## 3. Données

Dataset principal : **Caltech-UCSD Birds-200-2011 (CUB-200-2011)**.

Sous-ensemble utilisé :
- 10 classes explicitement définies ;
- 583 images valides au total ;
- Split stratifié (70 % / 15 % / 15 %) :
  - **Train** : 408 images
  - **Validation** : 87 images
  - **Test** : 88 images
- Taille d’entrée : 224 × 224 × 3.

Les classes utilisées dans l’expérience sont :

1. `009.Brewer_Blackbird`
2. `010.Red_winged_Blackbird`
3. `011.Rusty_Blackbird`
4. `012.Yellow_headed_Blackbird`
5. `113.Baird_Sparrow`
6. `114.Black_throated_Sparrow`
7. `115.Brewer_Sparrow`
8. `116.Chipping_Sparrow`
9. `117.Clay_colored_Sparrow`
10. `118.House_Sparrow`

## 4. Pipeline

```text
CUB-200-2011
      ↓
Sélection des 10 classes
      ↓
Contrôle qualité des images (PIL verify)
      ↓
Split stratifié Train / Validation / Test
      ↓
Pipeline tf.data + Augmentation à la volée
      ↓
CNN baseline (From Scratch)
      ↓
MobileNetV2 Transfer Learning (Feature Extraction)
      ↓
MobileNetV2 Fine-Tuning (30 couches dégelées)
      ↓
Évaluation & Matrice de confusion
      ↓
Analyse des erreurs & Benchmark d'inférence
      ↓
Prédiction réutilisable
      ↓
Interface Gradio & Exécution d'artefacts
