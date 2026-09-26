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

Un playground de transferencia de aprendizaje para IA de imágenes, basado en navegador y construido sobre TensorFlow.js MobileNet.

Client Side AI Training Environment es un **playground basado en navegador** para experimentar con
clasificación de imágenes sobre **TensorFlow.js MobileNet**.  
Puedes importar tus propias carpetas de imágenes etiquetadas, entrenar un pequeño classifier head completamente
en el navegador y ejecutar predicciones — **sin enviar tus imágenes a ningún servidor**.

---

## ✨ Funciones

- **Entrenamiento solo del lado del cliente**  
  - Usa TensorFlow.js en el navegador. No se requiere backend ni APIs externas para entrenar.

- **Transfer learning sobre MobileNet**  
  - Carga un modelo MobileNet preentrenado y extrae embeddings.
  - Entrena un pequeño dense classifier head (capas fully connected con softmax) sobre tu dataset.

- **Importación de dataset basada en carpetas**  
  - Importa un árbol de carpetas como `/DATA_DIRECTORY/<class_name>/<images...>`.
  - Los nombres de clase se derivan automáticamente de los nombres de los directorios.
  - Muestra conteos de imágenes por clase y miniaturas de vista previa.

- **UI paso a paso**  
  - **1. Import training data** (carpeta de imágenes etiquetadas)  
  - **2. Train head model** sobre las features de MobileNet  
  - **3. Test & Predict** con una imagen de prueba separada

- **Privacidad por defecto**  
  - Todo el cómputo y los datos permanecen dentro de la pestaña del navegador.  
  - No hay I/O de red para las imágenes del usuario (a menos que compartas manualmente capturas o logs).

- **Funciona como app Expo Web**  
  - Implementada como una app Expo / React Native que se exporta a la web  
    y se sirve mediante GitHub Pages en `/client_side_ai_training_environment`.

---

## 🧰 Cómo funciona

Internamente, la app sigue este flujo:

1. **Cargar MobileNet (base model)**  
   - Al iniciar, la app carga MobileNet v2 mediante TensorFlow.js y prepara el backend WebGL si está disponible.

2. **Importar imágenes etiquetadas como una carpeta**  
   - En la web, puedes seleccionar una carpeta con una estructura como:
     - `DATA_DIRECTORY/cats/*.jpg`
     - `DATA_DIRECTORY/dogs/*.png`
   - El nombre del directorio padre (por ejemplo, `cats`, `dogs`) se convierte en la class label.

3. **Extraer features y entrenar un classifier head (transfer learning)**  
   - Para cada imagen de entrenamiento, MobileNet se usa para calcular un embedding vector.
   - Se crea un pequeño modelo `tf.sequential()` con:
     - Dense layer (por ejemplo, 128 units, ReLU)
     - Dropout
     - Final dense layer con softmax sobre todas las clases
   - El head se entrena con categorical cross-entropy y Adam optimizer.
   - Esta es una configuración clásica de **transfer learning**: la base network (MobileNet) queda congelada, y solo el classifier head se entrena con tus datos.

4. **Ejecutar predicciones en una imagen de prueba**  
   - Una imagen de prueba separada se pasa por MobileNet para obtener un embedding.
   - El head model predice class probabilities.
   - La UI muestra:
     - Top predicted label
     - Top-k class confidences como porcentajes

Como el base model está congelado y solo el classifier head se entrena en el navegador, el entrenamiento es

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

En la web, seleccionas la carpeta de nivel superior (por ejemplo, DATA_DIRECTORY).
La app recorre el árbol, infiere etiquetas a partir de los nombres de carpeta y cuenta las imágenes por etiqueta.

---

## Entorno recomendado

- Un navegador de escritorio moderno (Chrome, Edge o Firefox)
- WebGL habilitado (para la aceleración GPU de TensorFlow.js).
- Carpetas de imágenes locales accesibles desde tu sistema de archivos.

Note: Algunos navegadores móviles pueden no admitir la subida de carpetas (webkitdirectory) o pueden ofrecer una experiencia degradada.

---

## 🚀 Primeros pasos

### 1. Prerequisites
- [Docker Compose](https://docs.docker.com/compose/)

### 2. Compilar e iniciar todos los servicios:

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
