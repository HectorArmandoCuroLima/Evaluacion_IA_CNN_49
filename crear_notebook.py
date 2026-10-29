import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
notebook_path = root / "notebooks" / "experiment.ipynb"
notebook_path.parent.mkdir(parents=True, exist_ok=True)

def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True)
    }

def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True)
    }

cells = [
    md("""# Experimento de clasificación CIFAR-10 con CNN

**Proyecto:** Evaluación de Inteligencia Artificial  
**Semilla:** 49  
**Framework:** PyTorch

Este notebook documenta la carga de datos, arquitectura, entrenamiento y evaluación de una red neuronal convolucional.
"""),
    md("""## 1. Configuración del entorno"""),
    code("""import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import pandas as pd
import matplotlib.pyplot as plt

from src.dataset import SEED, set_seed, load_datasets
from src.model import CNN

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Proyecto:", ROOT)
print("Semilla:", SEED)
print("Dispositivo:", DEVICE)
print("PyTorch:", torch.__version__)
"""),
    md("""## 2. Carga y distribución de los datos"""),
    code("""(
    train_dataset,
    val_dataset,
    test_dataset,
    train_loader,
    val_loader,
    test_loader,
    base_dataset,
) = load_datasets(root=str(ROOT / "data"), batch_size=128)

print("Entrenamiento:", len(train_dataset))
print("Validación:", len(val_dataset))
print("Prueba:", len(test_dataset))

images, labels = next(iter(train_loader))
print("Forma de imágenes:", tuple(images.shape))
print("Forma de etiquetas:", tuple(labels.shape))
"""),
    md("""## 3. Arquitectura de la CNN"""),
    code("""model = CNN().to(DEVICE)
parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(model)
print(f"Parámetros entrenables: {parameters:,}")

with torch.no_grad():
    output = model(images[:8].to(DEVICE))

print("Forma de salida:", tuple(output.shape))
"""),
    md("""## 4. Historial de entrenamiento

El historial se genera mediante `src/train.py`.
"""),
    code("""history_path = ROOT / "outputs" / "training_history.csv"

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
    print("No existe training_history.csv. Ejecuta python -m src.train")
"""),
    md("""## 5. Resultados de prueba"""),
    code("""metrics_path = ROOT / "outputs" / "test_metrics.txt"

if metrics_path.exists():
    print(metrics_path.read_text(encoding="utf-8"))
else:
    print("No existen métricas. Ejecuta python -m src.evaluate")
"""),
    md("""## 6. Matriz de confusión"""),
    code("""matrix_path = ROOT / "outputs" / "confusion_matrix.csv"

if matrix_path.exists():
    matrix = pd.read_csv(matrix_path, index_col=0)
    display(matrix)

    plt.figure(figsize=(9, 7))
    plt.imshow(matrix.values, cmap="Blues")
    plt.title("Matriz de confusión CIFAR-10")
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.colorbar(label="Cantidad")
    plt.tight_layout()
    plt.show()
else:
    print("No existe confusion_matrix.csv. Ejecuta python -m src.evaluate")
"""),
    md("""## 7. Conclusiones

- Registrar la exactitud final obtenida en el conjunto de prueba.
- Comparar las curvas de entrenamiento y validación para estudiar la generalización.
- Identificar las clases con menor exactitud y los errores más frecuentes.
- Proponer mejoras futuras, como ajustar las épocas, la arquitectura o la regularización.
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

print("Notebook creado correctamente.")
print("Archivo:", notebook_path)
print("Celdas:", len(cells))
print("Tamaño:", notebook_path.stat().st_size, "bytes")
