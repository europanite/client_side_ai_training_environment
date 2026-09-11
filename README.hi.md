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

TensorFlow.js MobileNet पर आधारित एक ब्राउज़र-आधारित Image AI Transfer Learning Playground।

Client Side AI Training Environment **TensorFlow.js MobileNet** के ऊपर इमेज क्लासिफिकेशन के प्रयोगों के लिए एक **ब्राउज़र-आधारित playground** है।  
आप अपने स्वयं के labeled image folders इंपोर्ट कर सकते हैं, पूरी तरह ब्राउज़र में एक छोटा classifier head ट्रेन कर सकते हैं, और predictions चला सकते हैं — **अपनी images किसी भी server पर भेजे बिना**।

---

## ✨ विशेषताएँ

- **केवल client-side training**  
  - ब्राउज़र में TensorFlow.js का उपयोग करता है। Training के लिए किसी backend या external APIs की आवश्यकता नहीं है।

- **MobileNet के ऊपर transfer learning**  
  - एक pre-trained MobileNet model लोड करता है और embeddings निकालता है।
  - आपके dataset पर एक छोटा dense classifier head (softmax के साथ fully connected layers) ट्रेन करता है।

- **Folder-based dataset import**  
  - `/DATA_DIRECTORY/<class_name>/<images...>` जैसे folder tree को इंपोर्ट करें।
  - Class names directory names से अपने आप निकाले जाते हैं।
  - Per-class image counts और preview thumbnails दिखाता है।

- **Step-by-step UI**  
  - **1. Import training data** (labeled images का folder)  
  - **2. Train head model** MobileNet features के ऊपर  
  - **3. Test & Predict** एक अलग test image के साथ

- **Privacy by default**  
  - सभी computation और data browser tab के अंदर ही रहते हैं।  
  - User images के लिए कोई network I/O नहीं है (जब तक आप screenshots या logs manually share नहीं करते)।

- **Expo Web app के रूप में काम करता है**  
  - इसे एक Expo / React Native app के रूप में implement किया गया है जिसे web पर export किया जाता है  
    और `/client_side_ai_training_environment` पर GitHub Pages के माध्यम से serve किया जाता है।

---

## 🧰 यह कैसे काम करता है

अंदरूनी रूप से, app यह flow follow करता है:

1. **MobileNet (base model) लोड करें**  
   - Startup पर, app TensorFlow.js के माध्यम से MobileNet v2 लोड करता है और उपलब्ध होने पर WebGL backend तैयार करता है।

2. **Labeled images को folder के रूप में import करें**  
   - Web पर, आप ऐसा folder select कर सकते हैं:
     - `DATA_DIRECTORY/cats/*.jpg`
     - `DATA_DIRECTORY/dogs/*.png`
   - Parent directory name (जैसे, `cats`, `dogs`) class label बन जाता है।

3. **Features निकालें और classifier head ट्रेन करें (transfer learning)**  
   - हर training image के लिए, MobileNet का उपयोग embedding vector compute करने के लिए किया जाता है।
   - एक छोटा `tf.sequential()` model बनाया जाता है जिसमें:
     - Dense layer (जैसे, 128 units, ReLU)
     - Dropout
     - सभी classes पर softmax के साथ final dense layer
   - Head को categorical cross-entropy और Adam optimizer के साथ train किया जाता है।
   - यह एक classic **transfer learning** setup है: base network (MobileNet) frozen रहता है, और केवल classifier head आपके data पर train होता है।

4. **Test image पर predictions चलाएँ**  
   - Embedding पाने के लिए एक अलग test image को MobileNet से pass किया जाता है।
   - Head model class probabilities predict करता है।
   - UI दिखाता है:
     - Top predicted label
     - Top-k class confidences percentages के रूप में

क्योंकि base model frozen है और केवल classifier head browser में train होता है, training है

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

Web पर, आप top-level folder (जैसे, DATA_DIRECTORY) select करते हैं।
App tree को traverse करता है, folder names से labels infer करता है, और per label images count करता है।

---

## Recommended environment

- एक modern desktop browser (Chrome, Edge, या Firefox)
- WebGL enabled (TensorFlow.js GPU acceleration के लिए)।
- आपके file system से accessible local image folders।

Note: कुछ mobile browsers folder upload (webkitdirectory) support नहीं कर सकते हैं या degraded experience दे सकते हैं।

---

## 🚀 Getting Started

### 1. Prerequisites
- [Docker Compose](https://docs.docker.com/compose/)

### 2. सभी services build और start करें:

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
