#!/bin/bash

# Script de setup para el proyecto CEIA-VPC3
# Clasificación de Plantas con Vision Transformers

echo "🌱 Creando estructura del proyecto CEIA-VPC3..."

# Crear directorios principales
mkdir -p config
mkdir -p data/{raw,processed,splits}
mkdir -p notebooks
mkdir -p src/{data,models,training,evaluation,utils}
mkdir -p experiments/configs
mkdir -p scripts
mkdir -p outputs/{models,figures,results}
mkdir -p mlruns
mkdir -p docs
mkdir -p assets/{images,diagrams,presentation}

# Crear archivos __init__.py para los paquetes Python
touch config/__init__.py
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/training/__init__.py
touch src/evaluation/__init__.py
touch src/utils/__init__.py

echo "📁 Estructura de carpetas creada"

# Crear .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Data
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep

# Models
outputs/models/*.pth
outputs/models/*.pt
*.pth
*.pt

# MLflow
mlruns/
mlartifacts/

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
*.tmp
EOF

echo "✅ .gitignore creado"

# Crear archivos .gitkeep para mantener carpetas vacías en git
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/splits/.gitkeep
touch outputs/models/.gitkeep
touch outputs/figures/.gitkeep
touch outputs/results/.gitkeep

# Crear README.md inicial
cat > README.md << 'EOF'
# 🌱 Clasificación de Plantas con Vision Transformers

Proyecto final para la materia **Visión por Computadora III** - CEIA

## 📋 Descripción

Sistema de clasificación de plantas utilizando Vision Transformers (ViT), comparando diferentes arquitecturas como DeiT y MobileViT.

## 🎯 Objetivos

- Implementar y comparar diferentes arquitecturas de Vision Transformers
- Realizar experimentación sistemática con hiperparámetros
- Evaluar el rendimiento con métricas apropiadas
- Desarrollar un pipeline de entrenamiento reproducible

## 🏗️ Estructura del Proyecto

```
CEIA-VPC3/
├── config/              # Configuraciones del proyecto
├── data/                # Datasets (raw, processed, splits)
├── notebooks/           # Jupyter notebooks para EDA
├── src/                 # Código fuente modular
│   ├── data/           # Dataset, transforms, dataloaders
│   ├── models/         # Arquitecturas de modelos
│   ├── training/       # Lógica de entrenamiento
│   ├── evaluation/     # Métricas y visualizaciones
│   └── utils/          # Utilidades generales
├── experiments/         # Scripts de experimentación
├── scripts/            # Scripts de ejecución
├── outputs/            # Modelos, figuras, resultados
├── assets/             # Imágenes, diagramas, presentación
└── docs/               # Documentación e informe técnico
```

## 🚀 Setup

### Requisitos

- Python 3.8+
- CUDA 11.8+ (para entrenamiento en GPU)

### Instalación

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## 📊 Dataset

**Flowers Recognition Dataset**
- 5 clases: daisy, dandelion, roses, sunflowers, tulips
- ~4,000 imágenes
- Fuente: Kaggle

## 🔬 Experimentación

Los experimentos están organizados en:
1. Baseline models
2. Arquitecturas (DeiT vs MobileViT)
3. Hiperparámetros (learning rate, optimizers, schedulers)
4. Data augmentation
5. Regularización

Ver `experiments/` para configuraciones detalladas.

## 📈 Resultados

[Por completar después de experimentación]

## 👥 Equipo

- [Nombre del equipo]
- [Integrantes]

## 📝 Licencia

Este proyecto es parte del programa de CEIA y está destinado únicamente para fines educativos.
EOF

echo "📄 README.md creado"

# Crear requirements.txt
cat > requirements.txt << 'EOF'
# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
transformers>=4.30.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
pillow>=9.5.0
albumentations>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# MLOps
mlflow>=2.3.0
tensorboard>=2.13.0

# Metrics & Evaluation
scikit-learn>=1.3.0
torchmetrics>=1.0.0

# Utils
pyyaml>=6.0
tqdm>=4.65.0
python-dotenv>=1.0.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.23.0
ipywidgets>=8.0.0
EOF

echo "📦 requirements.txt creado"

# Crear setup.py
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="ceia-vpc3",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pillow>=9.5.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "mlflow>=2.3.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)
EOF

echo "⚙️  setup.py creado"

# Crear config.yaml base
cat > config/config.yaml << 'EOF'
# Configuración General del Proyecto

# Dataset
data:
  dataset_name: "flowers"
  num_classes: 5
  classes: ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
  img_size: 224
  batch_size: 32
  num_workers: 4
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  seed: 42

# Model
model:
  architecture: "deit_tiny_patch16_224"
  pretrained: true
  num_classes: 5
  dropout: 0.1
  drop_path_rate: 0.0

# Training
training:
  epochs: 50
  learning_rate: 5e-5
  weight_decay: 1e-4
  optimizer: "adamw"
  scheduler: "cosine"
  warmup_epochs: 5
  label_smoothing: 0.1
  gradient_clip: 1.0
  
# Augmentation
augmentation:
  train:
    horizontal_flip: 0.5
    rotation: 15
    color_jitter:
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1
    random_resized_crop: true
  val:
    center_crop: true

# MLflow
mlflow:
  experiment_name: "flowers_classification"
  tracking_uri: "./mlruns"
  
# Paths
paths:
  data_dir: "./data"
  output_dir: "./outputs"
  models_dir: "./outputs/models"
  figures_dir: "./outputs/figures"
  results_dir: "./outputs/results"
EOF

echo "📝 config.yaml creado"

# Crear estructura básica de notebooks
cat > notebooks/01_eda.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🌱 Exploratory Data Analysis - Flowers Dataset\n",
    "\n",
    "Este notebook realiza el análisis exploratorio del dataset de flores."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('..')\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from pathlib import Path\n",
    "\n",
    "# Configuración\n",
    "plt.style.use('seaborn-v0_8-darkgrid')\n",
    "sns.set_palette('husl')\n",
    "\n",
    "%matplotlib inline\n",
    "%load_ext autoreload\n",
    "%autoreload 2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Carga de Datos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO: Implementar carga de datos"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Distribución de Clases"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO: Análisis de distribución"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Análisis de Imágenes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO: Visualización de muestras"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo "📓 Notebook EDA creado"

echo ""
echo "✨ ¡Proyecto CEIA-VPC3 configurado exitosamente!"
echo ""
echo "📋 Próximos pasos:"
echo "   1. cd /home/martin/Documents/CEIA/CEIA-VPC3"
echo "   2. python -m venv venv"
echo "   3. source venv/bin/activate"
echo "   4. pip install -r requirements.txt"
echo "   5. git init"
echo "   6. git add ."
echo "   7. git commit -m 'Initial project structure'"
echo ""
echo "🚀 ¡Listo para empezar a codear!"
