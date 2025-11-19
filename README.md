# [Client Side AI Training](https://github.com/europanite/client_side_ai_training "Client Side AI Training")

[![CI](https://github.com/europanite/client_side_ai_training/actions/workflows/ci.yml/badge.svg)](https://github.com/europanite/client_side_ai_training/actions/workflows/ci.yml)
[![docker](https://github.com/europanite/client_side_ai_training/actions/workflows/docker.yml/badge.svg)](https://github.com/europanite/client_side_ai_training/actions/workflows/docker.yml)
[![GitHub Pages](https://github.com/europanite/client_side_ai_training/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/europanite/client_side_ai_training/actions/workflows/deploy-pages.yml)

A Browser-Based AI Training Playground. 

!["web_ui"](./assets/images/web_ui.png)

##  Demo
 [Client Side AI Training](https://europanite.github.io/client_side_ai_training/)

---

## 🚀 Getting Started

### 1. Prerequisites
- [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/)

### 2. Build and start all services:

```bash
# set environment variables:
export REACT_NATIVE_PACKAGER_HOSTNAME=${YOUR_HOST}

# Build the image
docker compose build

# Run the container
docker compose up
```

---

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