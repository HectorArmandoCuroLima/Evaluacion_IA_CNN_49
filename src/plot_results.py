
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"

HISTORY_PATH = OUTPUT_DIR / "training_history.csv"
CONFUSION_PATH = OUTPUT_DIR / "confusion_matrix.csv"

CLASSES = [
    "avion", "automovil", "ave", "gato", "ciervo",
    "perro", "rana", "caballo", "barco", "camion"
]


def plot_training_history():
    with open(HISTORY_PATH, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    train_acc = [float(row["train_accuracy"]) for row in rows]
    val_acc = [float(row["val_accuracy"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, train_loss, marker="o", label="Entrenamiento")
    axes[0].plot(epochs, val_loss, marker="o", label="Validación")
    axes[0].set_title("Pérdida por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="Entrenamiento")
    axes[1].plot(epochs, val_acc, marker="o", label="Validación")
    axes[1].set_title("Precisión por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Precisión (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Entrenamiento de CNN - CIFAR-10 (semilla 49)")
    fig.tight_layout()

    path = OUTPUT_DIR / "training_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráficas de entrenamiento guardadas: {path}")


def plot_confusion_matrix():
    with open(CONFUSION_PATH, newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    matrix = np.array(
        [[int(value) for value in row[1:]] for row in rows[1:]],
        dtype=int,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Número de imágenes")

    ax.set(
        xticks=np.arange(len(CLASSES)),
        yticks=np.arange(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Clase predicha",
        ylabel="Clase real",
        title="Matriz de confusión - CIFAR-10",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    threshold = matrix.max() / 2

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()

    path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Matriz de confusión guardada: {path}")


if __name__ == "__main__":
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"No existe {HISTORY_PATH}")

    if not CONFUSION_PATH.exists():
        raise FileNotFoundError(f"No existe {CONFUSION_PATH}")

    plot_training_history()
    plot_confusion_matrix()
