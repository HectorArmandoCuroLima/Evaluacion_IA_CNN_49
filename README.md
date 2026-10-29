
@'
import json
from pathlib import Path

notebook_path = Path("notebooks/experiment.ipynb")
notebook_path.parent.mkdir(parents=True, exist_ok=True)

def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True)
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True)
    }

cells = [
    markdown_cell("""# Experimento: clasificación CIFAR-10 mediante CNN

**Proyecto:** Evaluación de Inteligencia Artificial  
**Semilla:** 49  
**Framework:** PyTorch  
**Objetivo:** entrenar y evaluar una red neuronal convolucional para clasificar imágenes en diez categorías.
"""),
    markdown_cell("""## 1. Configuración y reproducibilidad

Se fijará la semilla y se utilizará la función de carga de datos desarrollada en `src/dataset.py`.
"""),
    code_cell("""import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import matplotlib.pyplot as plt
import pandas as pd

from src.dataset import SEED, set_seed, load_datasets
from src.model import CNN

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Directorio del proyecto:", ROOT)
print("Semilla:", SEED)
print("Dispositivo:", DEVICE)
print("PyTorch:", torch.__version__)
"""),
    markdown_cell("""## 2. Carga y distribución de los datos

CIFAR-10 contiene diez categorías. Se utiliza una partición de entrenamiento, validación y prueba independiente.
"""),
    code_cell("""(
    train_dataset,
    val_dataset,
    test_dataset,
    train_loader,
    val_loader,
    test_loader,
    base_dataset,
) = load_datasets(root=str(ROOT / "data"), batch_size=128)

print(f"Entrenamiento: {len(train_dataset):,} imágenes")
print(f"Validación:    {len(val_dataset):,} imágenes")
print(f"Prueba:        {len(test_dataset):,} imágenes")

images, labels = next(iter(train_loader))
print("Forma del lote de imágenes:", tuple(images.shape))
print("Forma del lote de etiquetas:", tuple(labels.shape))
"""),
    markdown_cell("""## 3. Arquitectura del modelo

La CNN emplea tres bloques convolucionales con ReLU y MaxPooling, seguidos de una capa clasificadora con Dropout.
"""),
    code_cell("""model = CNN().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(model)
print(f"Parámetros entrenables: {total_params:,}")

with torch.no_grad():
    sample_output = model(images[:8].to(DEVICE))
print("Forma de salida para 8 imágenes:", tuple(sample_output.shape))
"""),
    markdown_cell("""## 4. Historial de entrenamiento

El entrenamiento se ejecuta con `src/train.py`. Esta sección visualiza el historial generado por dicho script.
"""),
    code_cell("""history_path = ROOT / "outputs" / "training_history.csv"

if history_path.exists():
    history = pd.read_csv(history_path)
    display(history)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="Entrenamiento")
    axes[0].plot(history["epoch"], history["val_loss"], marker="o", label="Validación")
    axes[0].set_title("Pérdida por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["epoch"], history["train_accuracy"], marker="o", label="Entrenamiento")
    axes[1].plot(history["epoch"], history["val_accuracy"], marker="o", label="Validación")
    axes[1].set_title("Exactitud por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
else:
    print("No se encontró el historial. Ejecuta primero: python -m src.train")
"""),
    markdown_cell("""## 5. Evaluación en el conjunto de prueba

La evaluación final se realiza con `src/evaluate.py`. El conjunto de prueba no debe utilizarse para ajustar los hiperparámetros.
"""),
    code_cell("""metrics_path = ROOT / "outputs" / "test_metrics.txt"

if metrics_path.exists():
    print(metrics_path.read_text(encoding="utf-8"))
else:
    print("No se encontraron métricas. Ejecuta primero: python -m src.evaluate")
"""),
    code_cell("""matrix_path = ROOT / "outputs" / "confusion_matrix.csv"

if matrix_path.exists():
    matrix = pd.read_csv(matrix_path, index_col=0)
    display(matrix)

    plt.figure(figsize=(9, 7))
    plt.imshow(matrix.values, cmap="Blues")
    plt.title("Matriz de confusión — CIFAR-10")
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.colorbar(label="Cantidad de imágenes")
    plt.tight_layout()
    plt.show()
else:
    print("No se encontró la matriz. Ejecuta primero: python -m src.evaluate")
"""),
    markdown_cell("""## 6. Conclusiones

Completa esta sección después de ejecutar todas las celdas y revisar las gráficas.

- **Desempeño:** registra la exactitud obtenida en el conjunto de prueba.
- **Generalización:** compara las curvas de entrenamiento y validación.
- **Errores:** identifica las categorías con menor exactitud y los pares de clases que se confunden con mayor frecuencia.
- **Mejoras futuras:** considera aumentar las épocas, ajustar la arquitectura o aplicar técnicas de regularización, comparando los resultados de forma controlada.
""")
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": sys.version.split()[0]
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

notebook_path.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print(f"Notebook creado: {notebook_path}")
print(f"Celdas creadas: {len(cells)}")
print(f"Tamaño: {notebook_path.stat().st_size:,} bytes")
'@ | Set-Content crear_notebook.py -Encoding UTF8

python crear_notebook.py
