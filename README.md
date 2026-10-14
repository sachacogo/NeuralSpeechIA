# NeuralSpeechIA

[![Project Video](https://img.shields.io/badge/Watch%20the-Video-FF0000?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=B5_9zbCxxHk)

Reconnaissance vocale de mots-clés **entièrement embarquée sur Arduino Due** : on appuie sur un bouton, on prononce « bonjour » ou « merci », et l'écran OLED affiche le mot reconnu avec son score de confiance.

Toute la chaîne est codée à la main, sans framework de ML embarqué : acquisition audio par registres, filtrage numérique, extraction des MFCC sur la carte et inférence d'un petit réseau de neurones convolutif écrit en C.

*Projet étudiant réalisé en 2025.*

## Fonctionnement

```
Micro ──► ADC 32 kHz ──► Filtre RIF + décimation ×4 ──► 8 kHz ──► MFCC ──► CNN ──► OLED
         (timer TC0)     (53 coefficients, Q15)                  (48×13)   (C pur)
```

| Étape | Détails |
|---|---|
| Acquisition | Timer TC0 et ADC configurés directement par registres (SAM3X8E), échantillonnage à ~32 kHz sur A0 |
| Filtrage | Filtre passe-bas RIF de 53 coefficients en virgule fixe Q15, puis décimation par 4 → 8 kHz |
| Trames | 256 échantillons par trame, avec recouvrement de 89 échantillons |
| Caractéristiques | 26 filtres Mel → 13 coefficients MFCC par trame, 48 trames (≈ 1 s d'audio) |
| Normalisation | Z-score par coefficient, identique en Python et en C |
| Modèle | Conv2D(8) → MaxPool → Conv2D(16) → MaxPool → Dense(8) → Dense(2), ≈ 5 900 paramètres (~23 Ko) |
| Inférence | Convolution, max-pooling, couches denses et softmax réimplémentés en C |

## Matériel

- Arduino Due
- Micro analogique amplifié, sortie sur **A0**
- Bouton poussoir entre la broche **52** et GND (pull-up interne activé)
- Écran OLED SSD1306 128×64 en I2C (adresse `0x3C`)

## Structure du dépôt

```
arduino/
├── audio_capture/      Capture de l'audio filtré brut, envoyé sur le port série (debug)
├── mfcc_export/        Calcul des MFCC sur la carte et envoi au PC (création du dataset)
└── voice_classifier/   Programme final : MFCC + CNN + affichage OLED
    └── model_weights.h Poids du réseau, générés par train_cnn.py

python/
├── mfcc_receiver.py    Enregistre les MFCC envoyés par la carte dans un fichier dataset
├── train_cnn.py        Entraîne le CNN et exporte les poids en en-tête C
├── audio_receiver.py   Enregistre l'audio brut envoyé par audio_capture (debug)
└── affichage_mfcc/     Affichage du signal capturé (debug)

dataset/
└── dataset_perso/      50 « bonjour », 50 « merci » et 10 exemples de test
```

## Utilisation

### Dépendances

- **Arduino** : bibliothèques Adafruit SSD1306 et Adafruit GFX, et la bibliothèque `arduinoMFCC` (non incluse dans ce dépôt).
- **Python** : `pip install numpy tensorflow pyserial matplotlib`

### 1. Créer le dataset

Téléverser `arduino/mfcc_export` sur la carte, puis, depuis le dossier `python/` :

```bash
python mfcc_receiver.py bonjour --n 50 --port COM5
python mfcc_receiver.py merci --n 50 --port COM5
```

À chaque enregistrement : appuyer sur le bouton, prononcer le mot, puis `Entrée` pour garder, `r` pour recommencer ou `q` pour quitter. Placer les fichiers obtenus dans `dataset/dataset_perso/`. Le fichier `datatest.txt` contient 5 « bonjour » suivis de 5 « merci ».

### 2. Entraîner le modèle

```bash
cd python
python train_cnn.py
```

Le script génère `model_weights.h`, à copier dans `arduino/voice_classifier/`.

### 3. Lancer la reconnaissance

Téléverser `arduino/voice_classifier` sur la carte, appuyer sur le bouton et prononcer « bonjour » ou « merci ». L'écran affiche « Hello » ou « Thank you » avec le score de confiance.

## Limites

- Le modèle ne connaît que deux mots : tout autre son sera classé comme l'un des deux.
- Le dataset a été enregistré par une seule personne, la reconnaissance est donc moins fiable avec d'autres voix.
