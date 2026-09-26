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

Um playground de transferência de aprendizado para IA de imagens, baseado no navegador e construído sobre TensorFlow.js MobileNet.

Client Side AI Training Environment é um **playground baseado no navegador** para experimentar
classificação de imagens sobre **TensorFlow.js MobileNet**.  
Você pode importar suas próprias pastas de imagens rotuladas, treinar um pequeno classifier head inteiramente
no navegador e executar predições — **sem enviar suas imagens para nenhum servidor**.

---

## ✨ Recursos

- **Treinamento somente no client-side**  
  - Usa TensorFlow.js no navegador. Nenhum backend ou APIs externas são necessários para treinar.

- **Transfer learning sobre MobileNet**  
  - Carrega um modelo MobileNet pré-treinado e extrai embeddings.
  - Treina um pequeno dense classifier head (camadas fully connected com softmax) no seu dataset.

- **Importação de dataset baseada em pastas**  
  - Importe uma árvore de pastas como `/DATA_DIRECTORY/<class_name>/<images...>`.
  - Os nomes das classes são derivados automaticamente dos nomes dos diretórios.
  - Mostra contagens de imagens por classe e miniaturas de pré-visualização.

- **UI passo a passo**  
  - **1. Import training data** (pasta de imagens rotuladas)  
  - **2. Train head model** sobre as features do MobileNet  
  - **3. Test & Predict** com uma imagem de teste separada

- **Privacidade por padrão**  
  - Todo o processamento e os dados permanecem dentro da aba do navegador.  
  - Não há I/O de rede para imagens do usuário (a menos que você compartilhe manualmente screenshots ou logs).

- **Funciona como um app Expo Web**  
  - Implementado como um app Expo / React Native que é exportado para a web  
    e servido via GitHub Pages em `/client_side_ai_training_environment`.

---

## 🧰 Como funciona

Por baixo dos panos, o app segue este fluxo:

1. **Carregar MobileNet (base model)**  
   - Na inicialização, o app carrega o MobileNet v2 via TensorFlow.js e prepara o backend WebGL se estiver disponível.

2. **Importar imagens rotuladas como uma pasta**  
   - Na web, você pode selecionar uma pasta com esta aparência:
     - `DATA_DIRECTORY/cats/*.jpg`
     - `DATA_DIRECTORY/dogs/*.png`
   - O nome do diretório pai (por exemplo, `cats`, `dogs`) se torna a class label.

3. **Extrair features e treinar um classifier head (transfer learning)**  
   - Para cada imagem de treinamento, o MobileNet é usado para calcular um embedding vector.
   - Um pequeno modelo `tf.sequential()` é criado com:
     - Dense layer (por exemplo, 128 units, ReLU)
     - Dropout
     - Final dense layer com softmax sobre todas as classes
   - O head é treinado com categorical cross-entropy e Adam optimizer.
   - Esta é uma configuração clássica de **transfer learning**: a base network (MobileNet) fica congelada, e apenas o classifier head é treinado com seus dados.

4. **Executar predições em uma imagem de teste**  
   - Uma imagem de teste separada passa pelo MobileNet para obter um embedding.
   - O head model prediz class probabilities.
   - A UI exibe:
     - Top predicted label
     - Top-k class confidences como porcentagens

Como o base model fica congelado e apenas o classifier head é treinado no navegador, o treinamento é

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

Na web, você seleciona a pasta de nível superior (por exemplo, DATA_DIRECTORY).
O app percorre a árvore, infere labels a partir dos nomes das pastas e conta as imagens por label.

---

## Ambiente recomendado

- Um navegador desktop moderno (Chrome, Edge ou Firefox)
- WebGL habilitado (para aceleração GPU do TensorFlow.js).
- Pastas de imagens locais acessíveis pelo seu sistema de arquivos.

Note: Alguns navegadores mobile podem não oferecer suporte a upload de pastas (webkitdirectory) ou podem oferecer uma experiência degradada.

---

## 🚀 Primeiros passos

### 1. Prerequisites
- [Docker Compose](https://docs.docker.com/compose/)

### 2. Build e inicialização de todos os serviços:

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
