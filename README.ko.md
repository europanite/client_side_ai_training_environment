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

TensorFlow.js MobileNet 위에 구축된 브라우저 기반 이미지 AI 전이 학습 playground입니다.

Client Side AI Training Environment는 **TensorFlow.js MobileNet** 위에서 이미지 분류를 실험하기 위한 **브라우저 기반 playground**입니다.  
직접 준비한 labeled image folders를 가져와 브라우저 안에서 작은 classifier head를 학습시키고 predictions를 실행할 수 있습니다 — **이미지를 어떤 서버에도 보내지 않습니다**.

---

## ✨ 기능

- **Client-side training only**  
  - 브라우저에서 TensorFlow.js를 사용합니다. 학습에는 backend나 external APIs가 필요하지 않습니다.

- **MobileNet 위의 transfer learning**  
  - Pre-trained MobileNet model을 로드하고 embeddings를 추출합니다.
  - 사용자의 dataset에서 작은 dense classifier head(softmax가 포함된 fully connected layers)를 학습합니다.

- **Folder-based dataset import**  
  - `/DATA_DIRECTORY/<class_name>/<images...>` 같은 folder tree를 가져올 수 있습니다.
  - Class names는 directory names에서 자동으로 생성됩니다.
  - 클래스별 이미지 수와 preview thumbnails를 표시합니다.

- **Step-by-step UI**  
  - **1. Import training data** (labeled images 폴더)  
  - **2. Train head model** on top of MobileNet features  
  - **3. Test & Predict** with a separate test image

- **Privacy by default**  
  - 모든 계산과 데이터는 browser tab 안에 머무릅니다.  
  - 사용자 이미지에 대한 network I/O는 없습니다(스크린샷이나 로그를 직접 공유하는 경우 제외).

- **Expo Web app으로 동작**  
  - Web으로 export되는 Expo / React Native app으로 구현되어 있으며,  
    `/client_side_ai_training_environment`에서 GitHub Pages를 통해 제공됩니다.

---

## 🧰 작동 방식

내부적으로 app은 다음 흐름을 따릅니다.

1. **MobileNet (base model) 로드**  
   - 시작 시 app은 TensorFlow.js를 통해 MobileNet v2를 로드하고, 사용 가능하면 WebGL backend를 준비합니다.

2. **Labeled images를 folder로 가져오기**  
   - Web에서는 다음과 같은 folder를 선택할 수 있습니다.
     - `DATA_DIRECTORY/cats/*.jpg`
     - `DATA_DIRECTORY/dogs/*.png`
   - Parent directory name(예: `cats`, `dogs`)이 class label이 됩니다.

3. **Features 추출 및 classifier head 학습 (transfer learning)**  
   - 각 training image에 대해 MobileNet을 사용해 embedding vector를 계산합니다.
   - 작은 `tf.sequential()` model이 다음 구성으로 만들어집니다.
     - Dense layer (예: 128 units, ReLU)
     - Dropout
     - 모든 클래스에 대한 softmax를 사용하는 final dense layer
   - Head는 categorical cross-entropy와 Adam optimizer로 학습됩니다.
   - 이는 전형적인 **transfer learning** 구성입니다. Base network(MobileNet)는 frozen 상태이며, 사용자의 data에서 classifier head만 학습됩니다.

4. **Test image에서 predictions 실행**  
   - 별도의 test image를 MobileNet에 통과시켜 embedding을 얻습니다.
   - Head model이 class probabilities를 예측합니다.
   - UI는 다음을 표시합니다.
     - Top predicted label
     - Top-k class confidences as percentages

Base model은 frozen 상태이고 browser에서 classifier head만 학습되므로, training is

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

Web에서 top-level folder(예: DATA_DIRECTORY)를 선택합니다.
App은 tree를 순회하고, folder names에서 labels를 추론하며, label별 이미지를 계산합니다.

---

## 권장 환경

- Modern desktop browser (Chrome, Edge 또는 Firefox)
- WebGL enabled (TensorFlow.js GPU acceleration용).
- 파일 시스템에서 접근 가능한 local image folders.

Note: 일부 mobile browsers는 folder upload(webkitdirectory)를 지원하지 않거나, 사용 경험이 제한될 수 있습니다.

---

## 🚀 시작하기

### 1. Prerequisites
- [Docker Compose](https://docs.docker.com/compose/)

### 2. 모든 services를 build하고 start합니다.

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
