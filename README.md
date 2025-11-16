#  Clasificación de Ingredientes de Cocina con Vision Transformers

Proyecto final para la materia **Visión por Computadora III** - CEIA

##  Descripción

Sistema de clasificación automática de ingredientes de cocina utilizando Vision Transformers (ViT), comparando diferentes arquitecturas como DeiT y MobileViT. El sistema puede identificar 40 tipos diferentes de ingredientes comunes, con aplicaciones en asistentes de cocina inteligentes, gestión de inventario y recomendación de recetas.

##  Objetivos

- Implementar y comparar diferentes arquitecturas de Vision Transformers (DeiT-tiny, MobileViT)
- Realizar experimentación sistemática con hiperparámetros y técnicas de optimización
- Evaluar el rendimiento con métricas apropiadas para clasificación multi-clase
- Desarrollar un pipeline de entrenamiento reproducible con tracking MLflow
- Analizar el comportamiento del modelo en clases visualmente similares

##  Dataset

**Food-Ingredient-Dataset-51** (Kaggle)
- **51 clases** de frutas y vegetales
- **~5,000+ imágenes** totales
- **Clases incluidas**:
  - Frutas: apple, banana, grapes, kiwi, lemon, mango, orange, pear, pineapple, pomegranate, watermelon, grapefruit, lime, peach, plum, strawberry
  - Vegetales: beetroot, bell_pepper, cabbage, capsicum, carrot, cauliflower, chilli_pepper, corn, cucumber, eggplant, garlic, ginger, lettuce, onion, paprika, peas, potato, radish, soy_beans, spinach, sweetcorn, sweetpotato, tomato, turnip, broccoli, green_beans, mushroom, okra, pumpkin, zucchini, asparagus, avocado, celery, jalepeno, red_chilli
- **Fuente**: [Kaggle - Food-Ingredient-Dataset-51](https://www.kaggle.com/datasets/sunnyagarwal427444/food-ingredient-dataset-51)
- **Alternativa**: [HuggingFace Mirror](https://huggingface.co/datasets/SunnyAgarwal4274/Food_Ingredients)

##  Estructura del Proyecto

```
CEIA-VPC3/
├── config/              # Configuraciones del proyecto
│   ├── config.yaml     # Configuración principal
│   └── mlflow_config.py
├── data/                # Datasets (raw, processed, splits)
│   ├── raw/            # Dataset original de Roboflow
│   ├── processed/      # Imágenes preprocesadas
│   └── splits/         # Train/val/test splits
├── notebooks/           # Jupyter notebooks para EDA
│   ├── 01_eda.ipynb
│   ├── 02_data_exploration.ipynb
│   └── 03_results_analysis.ipynb
├── src/                 # Código fuente modular
│   ├── data/           # Dataset, transforms, dataloaders
│   ├── models/         # Arquitecturas de modelos (DeiT, MobileViT)
│   ├── training/       # Lógica de entrenamiento
│   ├── evaluation/     # Métricas y visualizaciones
│   └── utils/          # Utilidades generales
├── experiments/         # Scripts de experimentación
│   ├── baseline.py
│   ├── experiment_runner.py
│   └── configs/        # Configuraciones de experimentos
├── scripts/            # Scripts de ejecución
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
├── outputs/            # Modelos, figuras, resultados
│   ├── models/         # Checkpoints de modelos
│   ├── figures/        # Gráficos y visualizaciones
│   └── results/        # Métricas y reportes
├── mlruns/             # MLflow tracking
├── assets/             # Recursos adicionales
│   ├── images/         # Imágenes para README
│   ├── diagrams/       # Diagramas de arquitectura
│   └── presentation/   # Slides de presentación
└── docs/               # Documentación e informe técnico
```

##  Setup

### Requisitos Previos

- Python 3.10+
- CUDA 11.8+ (para entrenamiento en GPU)
- [uv](https://github.com/astral-sh/uv) - Gestor de paquetes ultrarrápido

### Instalación de uv

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setup del Proyecto

```bash
# Clonar/navegar al proyecto
cd /home/martin/Documents/CEIA/CEIA-VPC3

# Crear entorno virtual con uv
uv venv

# Activar entorno virtual
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Instalar dependencias
uv pip install -r requirements.txt

# Instalar el proyecto en modo desarrollo
uv pip install -e .
```

### Descargar Dataset

```bash
# Opción 1: Usar script de descarga automática (Kaggle API)
python scripts/download_kaggle.py

# Opción 2: Manual desde Kaggle
# 1. Ir a: https://www.kaggle.com/datasets/sunnyagarwal427444/food-ingredient-dataset-51
# 2. Click "Download" (requiere cuenta de Kaggle)
# 3. Extraer ZIP en data/raw/

# Opción 3: Desde HuggingFace
# https://huggingface.co/datasets/SunnyAgarwal4274/Food_Ingredients
```

**Setup de Kaggle API:**
```bash
# 1. Obtener credentials:
#    - Ir a https://www.kaggle.com/settings/account
#    - Scroll a 'API' → 'Create New Token'
#    - Descargar kaggle.json

# 2. Agregar al .env:
#    KAGGLE_USERNAME=your_username
#    KAGGLE_KEY=your_key_from_kaggle_json

# 3. Ejecutar descarga:
python scripts/download_kaggle.py
```

### Configurar Databricks MLflow

```bash
# 1. Crear archivo .env en la raíz del proyecto
cp .env.example .env

# 2. Editar .env con tus credenciales de Databricks
# DATABRICKS_HOST=https://tu-workspace.cloud.databricks.com
# DATABRICKS_TOKEN=tu_personal_access_token

# 3. Obtener tu PAT (Personal Access Token):
#    - Ir a Databricks workspace
#    - User Settings → Developer → Access Tokens
#    - Generate New Token
#    - Copiar token al .env

# 4. Probar conexión
python config/mlflow_config.py
```

**Nota**: El archivo `.env` está en `.gitignore` y NUNCA debe subirse a git.

## 🔬 Experimentación

### Experimentos Planificados

1. **Baseline Models**
   - DeiT-tiny con capas congeladas
   - MobileViT-small baseline

2. **Arquitecturas**
   - Comparación DeiT-tiny vs MobileViT-small vs MobileViT-xx-small
   - Fine-tuning completo vs parcial

3. **Optimización**
   - Learning rates: [1e-5, 5e-5, 1e-4, 5e-4]
   - Optimizers: Adam, AdamW, SGD
   - Schedulers: Cosine, Step Decay, ReduceLROnPlateau

4. **Data Augmentation**
   - Baseline: Resize + Normalize
   - Medium: + HorizontalFlip + Rotation + ColorJitter
   - Heavy: + RandomResizedCrop + Mixup

5. **Regularización**
   - Dropout: [0.1, 0.3, 0.5]
   - Weight decay: [1e-5, 1e-4, 1e-3]
   - Label smoothing: [0, 0.1]

### Ejecutar Experimentos

```bash
# Baseline
python experiments/baseline.py

# Experimento específico
python scripts/train.py --config experiments/configs/exp_augmentation.yaml

# Runner de múltiples experimentos
python experiments/experiment_runner.py
```

### Monitoreo con MLflow

```bash
# Opción 1: Ver en Databricks UI
# - Ir a tu workspace de Databricks
# - Machine Learning → Experiments
# - Buscar "ingredients_classification"

# Opción 2: MLflow UI local (alternativo)
mlflow ui --backend-store-uri databricks --port 5000
# Abrir en navegador: http://localhost:5000
```

##  Métricas de Evaluación

- **Accuracy** (overall y por clase)
- **Balanced Accuracy** (para manejar desbalance)
- **Precision, Recall, F1-score** (macro y weighted)
- **Top-3 y Top-5 Accuracy**
- **Confusion Matrix** con análisis de clases confundidas
- **Learning Curves** (train/val loss y accuracy)
- **Inference Time** y eficiencia computacional

##  Aplicaciones Potenciales

-  **App móvil**: Reconocimiento de ingredientes en tiempo real
-  **Gestión de inventario**: Control automático de stock en cocinas
-  **Recomendación de recetas**: Sugerir recetas basadas en ingredientes disponibles
-  **Asistente de compras**: Lista inteligente de compras
-  **Nutrición**: Análisis nutricional automático de comidas

##  Resultados Preliminares

[Por completar después de experimentación]

##  Equipo

- Martín Brocca
- Ariadna Garmendia
- Carina Roldan

##  Tecnologías Utilizadas

- **Deep Learning**: PyTorch, Timm, Transformers
- **MLOps**: MLflow, TensorBoard
- **Data**: Albumentations, Torchvision
- **Visualización**: Matplotlib, Seaborn, Plotly
- **Gestión de Paquetes**: uv

##  Referencias

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Training data-efficient image transformers](https://arxiv.org/abs/2012.12877) (DeiT)
- [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178)

##  Licencia

Este proyecto es parte del programa de CEIA (Especialización en Inteligencia Artificial) y está destinado únicamente para fines educativos y académicos.

---

**Materia**: Visión por Computadora III - Vision Transformers  
**Institución**: CEIA  
**Año**: 2025