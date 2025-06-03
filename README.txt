# 🅿️ Detector de Espacios de Estacionamiento con IA (CNN + PyTorch)

Este proyecto permite detectar espacios de estacionamiento **ocupados y vacíos** usando un modelo de red neuronal convolucional (CNN) entrenado con PyTorch y aplicado a video de aéreas de estacionamientos.

---

## 📁 Estructura del Proyecto

```
proyecto/
├── entrenamiento_cnn_ligero.py         # Entrena el modelo CNN personalizado
├── modelo_cnn_ligero.pth               # Modelo guardado tras entrenamiento
├── detector_en_video_cnn.py            # Detección en video usando el modelo
├── zonas_estacionamiento.py            # Coordenadas extraídas desde máscara
├── mask_1920_1080.png                  # Imagen binaria con zonas reales
```

---

## ✅ Requisitos

Instala las dependencias necesarias:

```bash
pip install torch torchvision opencv-python numpy
```

Descargar el dataset de https://www.kaggle.com/datasets/iasadpanwhar/parking-lot-detection-counter?resource=download

---

## 🧠 Entrenamiento del modelo

El modelo se entrena con imágenes de 68x29 píxeles, organizadas en:

```
clf-data/
├── empty/
│   ├── vacio1.jpg
│   └── ...
└── not_empty/
    ├── lleno1.jpg
    └── ...
```

Ejecuta el entrenamiento:

```bash
python entrenamiento_cnn_ligero.py
```

Esto generará `modelo_cnn_ligero.pth`.

---

## Detección en video

### Zonas válidas desde máscara

Usa `mask_1920_1080.png` para generar automáticamente las coordenadas válidas.

```bash
python detector_en_video_cnn.py
```

El script usa:

```python
from zonas_estacionamiento import zonas
```

---

## 📦 Imports utilizados

```python
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
```

---

## 💡 Observaciones

- El modelo fue optimizado para imágenes pequeñas (68x29).
- El uso de máscara visual elimina zonas irrelevantes.
- La detección funciona en tiempo real sobre video local.

---

## ✨ Autor

Desarrollado con el apoyo de IA (ChatGPT) y Hernan Martínez