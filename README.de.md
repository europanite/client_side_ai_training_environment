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

Ein browserbasiertes Image-AI-Transfer-Learning-Playground, aufgebaut auf TensorFlow.js MobileNet.

Client Side AI Training Environment ist ein **browserbasiertes Playground** zum Experimentieren mit
Bildklassifikation auf Basis von **TensorFlow.js MobileNet**.  
Du kannst eigene beschriftete Bildordner importieren, einen kleinen classifier head vollständig
im Browser trainieren und Vorhersagen ausführen — **ohne deine Bilder an irgendeinen Server zu senden**.

---

## ✨ Funktionen

- **Ausschließlich clientseitiges Training**  
  - Verwendet TensorFlow.js im Browser. Für das Training sind kein Backend und keine externen APIs erforderlich.

- **Transfer Learning auf MobileNet**  
  - Lädt ein vortrainiertes MobileNet-Modell und extrahiert embeddings.
  - Trainiert einen kleinen dense classifier head (fully connected layers mit softmax) auf deinem dataset.

- **Ordnerbasierter Dataset-Import**  
  - Importiere eine Ordnerstruktur wie `/DATA_DIRECTORY/<class_name>/<images...>`.
  - Class names werden automatisch aus den Verzeichnisnamen abgeleitet.
  - Zeigt Bildanzahlen pro Klasse und Vorschau-Thumbnails an.

- **Schrittweise UI**  
  - **1. Import training data** (Ordner mit beschrifteten Bildern)  
  - **2. Train head model** auf MobileNet features  
  - **3. Test & Predict** mit einem separaten Testbild

- **Privacy by default**  
  - Alle Berechnungen und Daten bleiben innerhalb des Browser-Tabs.  
  - Kein Netzwerk-I/O für Benutzerbilder (es sei denn, du teilst manuell Screenshots oder Logs).

- **Funktioniert als Expo Web app**  
  - Implementiert als Expo / React Native app, die ins Web exportiert wird  
    und über GitHub Pages unter `/client_side_ai_training_environment` bereitgestellt wird.

---

## 🧰 Funktionsweise

Unter der Haube folgt die App diesem Ablauf:

1. **MobileNet (base model) laden**  
   - Beim Start lädt die App MobileNet v2 über TensorFlow.js und bereitet, falls verfügbar, das WebGL backend vor.

2. **Beschriftete Bilder als Ordner importieren**  
   - Im Web kannst du einen Ordner auswählen, der so aussieht:
     - `DATA_DIRECTORY/cats/*.jpg`
     - `DATA_DIRECTORY/dogs/*.png`
   - Der Name des übergeordneten Verzeichnisses (z. B. `cats`, `dogs`) wird zum class label.

3. **Features extrahieren und einen classifier head trainieren (transfer learning)**  
   - Für jedes Trainingsbild wird MobileNet verwendet, um einen embedding vector zu berechnen.
   - Ein kleines `tf.sequential()` model wird erstellt mit:
     - Dense layer (z. B. 128 units, ReLU)
     - Dropout
     - Final dense layer mit softmax über alle Klassen
   - Der Head wird mit categorical cross-entropy und Adam optimizer trainiert.
   - Dies ist ein klassisches **transfer learning** setup: Das base network (MobileNet) ist eingefroren, und nur der classifier head wird auf deinen Daten trainiert.

4. **Vorhersagen für ein Testbild ausführen**  
   - Ein separates Testbild wird durch MobileNet geleitet, um ein embedding zu erhalten.
   - Das head model sagt class probabilities voraus.
   - Die UI zeigt:
     - Top predicted label
     - Top-k class confidences als Prozentwerte

Da das base model eingefroren ist und nur der classifier head im Browser trainiert wird, ist das Training

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

Im Web wählst du den obersten Ordner aus (z. B. DATA_DIRECTORY).
Die App durchläuft den Baum, leitet labels aus Ordnernamen ab und zählt Bilder pro label.

---

## Empfohlene Umgebung

- Ein moderner Desktop-Browser (Chrome, Edge oder Firefox)
- WebGL enabled (für TensorFlow.js GPU acceleration).
- Lokale Bildordner, die über dein Dateisystem zugänglich sind.

Note: Einige mobile Browser unterstützen möglicherweise keinen Ordner-Upload (webkitdirectory) oder bieten nur eine eingeschränkte Erfahrung.

---

## 🚀 Erste Schritte

### 1. Prerequisites
- [Docker Compose](https://docs.docker.com/compose/)

### 2. Alle Services bauen und starten:

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
