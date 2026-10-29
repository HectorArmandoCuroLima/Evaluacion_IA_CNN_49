# ============================================================
# PREGUNTA 1: CARGA, VISUALIZACIÓN Y PREPROCESAMIENTO CIFAR-10
# Archivo: src/dataset.py
# Semilla individual: 49
# ============================================================

import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA SEMILLA PARA REPRODUCIBILIDAD
# ------------------------------------------------------------

SEED = 49


def set_seed(seed=SEED):
    """Fija las semillas para favorecer la reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# 2. CLASES DEL DATASET CIFAR-10
# ------------------------------------------------------------

CLASSES = (
    "avion",
    "automovil",
    "ave",
    "gato",
    "ciervo",
    "perro",
    "rana",
    "caballo",
    "barco",
    "camion",
)


# ------------------------------------------------------------
# 3. PIPELINE DE TRANSFORMACIONES
# ------------------------------------------------------------

# Transformaciones para las imágenes de entrenamiento:
# - RandomCrop: recorte aleatorio con padding.
# - RandomHorizontalFlip: volteo horizontal aleatorio.
# - ToTensor: convierte la imagen en un tensor.
# - Normalize: normaliza los tres canales RGB.

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])


# Para validación y prueba no se aplican aumentos aleatorios.
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])


# ------------------------------------------------------------
# 4. VISUALIZACIÓN DE 16 IMÁGENES CON SUS ETIQUETAS REALES
# ------------------------------------------------------------

def show_samples(dataset, number=16):
    """Muestra una cuadrícula de imágenes y sus clases reales."""

    number = min(number, len(dataset))
    rows, cols = 4, 4

    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(number):
        image, label = dataset[i]

        # CIFAR-10 sin transformaciones entrega imágenes PIL.
        axes[i].imshow(image)
        axes[i].set_title(
            f"Etiqueta: {CLASSES[label]} ({label})"
        )
        axes[i].axis("off")

    for i in range(number, len(axes)):
        axes[i].axis("off")

    plt.suptitle(
        "CIFAR-10: 16 imágenes de ejemplo y sus etiquetas reales",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 5. CARGA DE CIFAR-10 Y PARTICIÓN ESTRATIFICADA
# ------------------------------------------------------------

def load_datasets(
    root="./data",
    batch_size=128,
    val_size=0.20,
    seed=SEED
):
    """Descarga CIFAR-10 y genera conjuntos de entrenamiento,
    validación y prueba.
    """

    set_seed(seed)

    # Dataset original sin transformaciones para inspeccionar
    # las imágenes y obtener las etiquetas reales.
    base_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=None
    )

    labels = np.array(base_dataset.targets)
    indices = np.arange(len(base_dataset))

    # División estratificada: mantiene aproximadamente la misma
    # proporción de cada clase en entrenamiento y validación.
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_size,
        random_state=seed,
        stratify=labels
    )

    # Se crean dos instancias del mismo conjunto original para
    # aplicar transformaciones diferentes a cada partición.
    train_full = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=train_transform
    )

    val_full = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=eval_transform
    )

    train_dataset = Subset(train_full, train_indices.tolist())
    val_dataset = Subset(val_full, val_indices.tolist())

    # El conjunto de prueba oficial se mantiene separado.
    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=eval_transform
    )

    # Generador para reproducir el orden de los datos al entrenar.
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # 6. IMPRESIÓN DE LAS DIMENSIONES Y COMPROBACIONES
    # --------------------------------------------------------

    print("\n========== DATASET CIFAR-10 ==========")
    print(f"Semilla utilizada: {seed}")
    print(f"Total original de entrenamiento: {len(base_dataset)}")
    print(f"Imágenes para entrenamiento: {len(train_dataset)}")   
    print(f"Imágenes para validación: {len(val_dataset)}")
    print(f"Imágenes para prueba: {len(test_dataset)}")

    print("\n========== DIMENSIONES DE LOS TENSORES ==========")

    images, batch_labels = next(iter(train_loader))

    print(f"Forma del lote de imágenes: {images.shape}")
    print(f"Forma del lote de etiquetas: {batch_labels.shape}")
    print(f"Tipo de dato de las imágenes: {images.dtype}")
    print(f"Tipo de dato de las etiquetas: {batch_labels.dtype}")

    print("\nForma esperada de las imágenes: [lote, 3, 32, 32]")
    print("Canales RGB: 3")
    print("Número de clases: 10")

    # Verifica que la partición contiene todas las clases.
    print("\n========== DISTRIBUCIÓN POR CLASE ==========")

    train_labels = labels[train_indices]
    val_labels = labels[val_indices]

    for class_id, class_name in enumerate(CLASSES):
        train_count = int(np.sum(train_labels == class_id))
        val_count = int(np.sum(val_labels == class_id))

        print(
            f"{class_name:12s} | "
            f"Entrenamiento: {train_count:4d} | "
            f"Validación: {val_count:4d}"
        )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        base_dataset
    )


# ------------------------------------------------------------
# 7. EJECUCIÓN DIRECTA DEL ARCHIVO
# ------------------------------------------------------------

if __name__ == "__main__":

    set_seed(SEED)

    # Carga los datos y realiza la partición estratificada.
    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        base_dataset
    ) = load_datasets()

    # Visualiza las primeras 16 imágenes originales.
    show_samples(base_dataset, number=16)