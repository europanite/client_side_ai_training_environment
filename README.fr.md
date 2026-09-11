# [Client Side AI Training Environment](https://github.com/europanite/client_side_ai_training_environment "Client Side AI Training Environment")

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![OS](https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-blue)
[![CI](https://github.com/europanite/client_side_ai_training_environment/actions/workflows/ci.yml/badge.svg)](https://github.com/europanite/client_side_ai_training_environment/actions/workflows/ci.yml)
[![docker](https://github.com/europanite/client_side_ai_training_environment/actions/workflows/docker.yml/badge.svg)](https://github.com/europanite/client_side_ai_training_environment/actions/workflows/docker.yml)
[![GitHub Pages](https://github.com/europanite/client_side_ai_training_environment/actions/workflows/pages.yml/badge.svg)](https://github.com/europanite/client_side_ai_training_environment/actions/workflows/pages.yml)

![React Native](https://img.shields.io/badge/react_native-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white)
![Jest](https://img.shields.io/badge/-jest-%23C21325?style=for-the-badge&logo=jest&logoColor=white)
![Expo](https://img.shields.io/badge/expo-1C1E24?style=for-the-badge&logo=expo&logoColor=#D04A37)

<p align="right">
  <a href="./README.md">🇺🇸 English</a> |
  <a href="./README.hi.md">🇮🇳 हिंदी</a> |
  <a href="./README.ja.md">🇯🇵 日本語</a> |
  <a href="./README.zh-CN.md">🇨🇳 简体中文</a> |
  <a href="./README.es.md">🇪🇸 Español</a> |
  <a href="./README.pt-BR.md">🇧🇷 Português (Brasil)</a> |
  <a href="./README.ko.md">🇰🇷 한국어</a> |
  <a href="./README.de.md">🇩🇪 Deutsch</a> |
  <a href="./README.fr.md">🇫🇷 Français</a>
</p>

!["web_ui"](./assets/images/web_ui.png)

 [PlayGround](https://europanite.github.io/client_side_ai_training_environment/)

Un playground de transfer learning pour l'IA d'images, basé sur le navigateur et construit sur TensorFlow.js MobileNet.

Client Side AI Training Environment est un **playground basé sur le navigateur** pour expérimenter la
classification d'images avec **TensorFlow.js MobileNet**.  
Vous pouvez importer vos propres dossiers d'images étiquetées, entraîner un petit classifier head entièrement
dans le navigateur et exécuter des prédictions — **sans envoyer vos images à aucun serveur**.

---

## ✨ Fonctionnalités

- **Entraînement uniquement côté client**  
  - Utilise TensorFlow.js dans le navigateur. Aucun backend ni aucune API externe n'est nécessaire pour l'entraînement.

- **Transfer learning au-dessus de MobileNet**  
  - Charge un modèle MobileNet préentraîné et extrait des embeddings.
  - Entraîne un petit dense classifier head (couches fully connected avec softmax) sur votre dataset.

- **Import de dataset basé sur les dossiers**  
  - Importez une arborescence de dossiers comme `/DATA_DIRECTORY/<class_name>/<images...>`.
  - Les class names sont automatiquement déduits des noms de répertoires.
  - Affiche le nombre d'images par classe et des miniatures de prévisualisation.

- **UI étape par étape**  
  - **1. Import training data** (dossier d'images étiquetées)  
  - **2. Train head model** sur les features MobileNet  
  - **3. Test & Predict** avec une image de test séparée

- **Privacy by default**  
  - Tous les calculs et toutes les données restent dans l'onglet du navigateur.  
  - Aucun I/O réseau pour les images utilisateur (sauf si vous partagez manuellement des captures d'écran ou des logs).

- **Fonctionne comme une app Expo Web**  
  - Implémentée comme une app Expo / React Native exportée vers le web  
    et servie via GitHub Pages à `/client_side_ai_training_environment`.

---

## 🧰 Fonctionnement

Sous le capot, l'app suit ce flux :

1. **Charger MobileNet (base model)**  
   - Au démarrage, l'app charge MobileNet v2 via TensorFlow.js et prépare le backend WebGL si disponible.

2. **Importer les images étiquetées comme un dossier**  
   - Sur le web, vous pouvez sélectionner un dossier qui ressemble à :
     - `DATA_DIRECTORY/cats/*.jpg`
     - `DATA_DIRECTORY/dogs/*.png`
   - Le nom du répertoire parent (par exemple, `cats`, `dogs`) devient le class label.

3. **Extraire les features et entraîner un classifier head (transfer learning)**  
   - Pour chaque image d'entraînement, MobileNet est utilisé pour calculer un embedding vector.
   - Un petit modèle `tf.sequential()` est créé avec :
     - Dense layer (par exemple, 128 units, ReLU)
     - Dropout
     - Final dense layer avec softmax sur toutes les classes
   - Le head est entraîné avec categorical cross-entropy et Adam optimizer.
   - Il s'agit d'une configuration classique de **transfer learning** : le base network (MobileNet) est gelé, et seul le classifier head est entraîné sur vos données.

4. **Exécuter des prédictions sur une image de test**  
   - Une image de test séparée passe dans MobileNet pour obtenir un embedding.
   - Le head model prédit les class probabilities.
   - L'UI affiche :
     - Top predicted label
     - Top-k class confidences en pourcentages

Comme le base model est gelé et que seul le classifier head est entraîné dans le navigateur, l'entraînement est

---

## Data

### Sample Data
- https://www.robots.ox.ac.uk/~vgg/data/flowers/
-https://fishnet-2023.github.io/

### Data Structure

<pre>
DATA_DIRECTORY
├── CLASS_NAME_1
│   ├── image_01.png
│   ├── image_02.png
│   ├── image_03.png
│   ├── ...
├── CLASS_NAME_2
│   ├── image_01.png
│   ├── image_02.png
│   ├── image_03.png
│   ├── ...
├── CLASS_NAME_3
│   ├── image_01.png
│   ├── image_02.png
│   ├── image_03.png
│   ├── ...
 ...

</pre>

Sur le web, vous sélectionnez le dossier de niveau supérieur (par exemple, DATA_DIRECTORY).
L'app parcourt l'arborescence, déduit les labels à partir des noms de dossiers et compte les images par label.

---

## Environnement recommandé

- Un navigateur desktop moderne (Chrome, Edge ou Firefox)
- WebGL enabled (pour l'accélération GPU de TensorFlow.js).
- Des dossiers d'images locaux accessibles depuis votre système de fichiers.

Note: Certains navigateurs mobiles peuvent ne pas prendre en charge l'upload de dossiers (webkitdirectory) ou offrir une expérience dégradée.

---

## 🚀 Démarrage

### 1. Prerequisites
- [Docker Compose](https://docs.docker.com/compose/)

### 2. Build et démarrage de tous les services :

```bash
# set environment variables:
export REACT_NATIVE_PACKAGER_HOSTNAME=${YOUR_HOST}

# Build the image
docker compose build

# Run the container
docker compose up
```

### 3. Test:
```bash
docker compose \
-f docker-compose.test.yml up \
--build --exit-code-from \
frontend_test 
```

---

# License
- Apache License 2.0
