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

一个基于 TensorFlow.js MobileNet 构建的浏览器端图像 AI 迁移学习 playground。

Client Side AI Training Environment 是一个基于 **TensorFlow.js MobileNet** 的**浏览器端 playground**，用于实验图像分类。  
你可以导入自己的带标签图片文件夹，在浏览器中完整训练一个小型分类头，并运行预测 — **无需将图片发送到任何服务器**。

---

## ✨ 功能

- **仅在客户端训练**  
  - 在浏览器中使用 TensorFlow.js。训练不需要后端或外部 API。

- **基于 MobileNet 的迁移学习**  
  - 加载预训练的 MobileNet 模型并提取 embeddings。
  - 在你的数据集上训练一个小型 dense 分类头（带 softmax 的全连接层）。

- **基于文件夹的数据集导入**  
  - 导入类似 `/DATA_DIRECTORY/<class_name>/<images...>` 的文件夹树。
  - 类名会自动从目录名推导出来。
  - 显示每个类别的图片数量和预览缩略图。

- **分步骤 UI**  
  - **1. Import training data**（带标签图片文件夹）  
  - **2. Train head model**，基于 MobileNet features  
  - **3. Test & Predict**，使用单独的测试图片

- **默认保护隐私**  
  - 所有计算和数据都保留在浏览器标签页内。  
  - 用户图片没有网络 I/O（除非你手动分享截图或日志）。

- **作为 Expo Web app 运行**  
  - 实现为可导出到 Web 的 Expo / React Native app，  
    并通过 GitHub Pages 在 `/client_side_ai_training_environment` 提供服务。

---

## 🧰 工作原理

在底层，该 app 遵循以下流程：

1. **加载 MobileNet（base model）**  
   - 启动时，app 通过 TensorFlow.js 加载 MobileNet v2，并在可用时准备 WebGL backend。

2. **以文件夹形式导入带标签图片**  
   - 在 Web 上，你可以选择如下结构的文件夹：
     - `DATA_DIRECTORY/cats/*.jpg`
     - `DATA_DIRECTORY/dogs/*.png`
   - 父目录名（例如 `cats`, `dogs`）会成为 class label。

3. **提取 features 并训练 classifier head（transfer learning）**  
   - 对每张训练图片，使用 MobileNet 计算 embedding vector。
   - 创建一个小型 `tf.sequential()` model，包含：
     - Dense layer（例如 128 units, ReLU）
     - Dropout
     - 对所有类别做 softmax 的 final dense layer
   - 该 head 使用 categorical cross-entropy 和 Adam optimizer 进行训练。
   - 这是一个经典的 **transfer learning** 设置：base network（MobileNet）被冻结，只在你的数据上训练 classifier head。

4. **在测试图片上运行预测**  
   - 单独的测试图片会经过 MobileNet，得到 embedding。
   - Head model 预测 class probabilities。
   - UI 显示：
     - Top predicted label
     - 以百分比表示的 Top-k class confidences

由于 base model 被冻结，并且只有 classifier head 在浏览器中训练，所以训练是

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

在 Web 上，你选择顶层文件夹（例如 DATA_DIRECTORY）。
App 会遍历该树，从文件夹名推断标签，并统计每个标签的图片数量。

---

## 推荐环境

- 现代桌面浏览器（Chrome、Edge 或 Firefox）
- 启用 WebGL（用于 TensorFlow.js GPU 加速）。
- 可从你的文件系统访问的本地图片文件夹。

Note: 某些移动浏览器可能不支持文件夹上传（webkitdirectory），或体验会有所降级。

---

## 🚀 入门

### 1. 前提条件
- [Docker Compose](https://docs.docker.com/compose/)

### 2. 构建并启动所有服务：

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
