
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Permite ejecutar el archivo directamente o como módulo.
try:
    from .dataset import SEED, set_seed, load_datasets
    from .model import CNN
except ImportError:
    from dataset import SEED, set_seed, load_datasets
    from model import CNN


# ------------------------------------------------------------
# 1. CONFIGURACIÓN
# ------------------------------------------------------------

EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"

MODEL_PATH = OUTPUT_DIR / "model_best.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"


# ------------------------------------------------------------
# 2. ENTRENAMIENTO DE UNA ÉPOCA
# ------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


# ------------------------------------------------------------
# 3. VALIDACIÓN
# ------------------------------------------------------------

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


# ------------------------------------------------------------
# 4. PROGRAMA PRINCIPAL
# ------------------------------------------------------------

def main():
    # Inicializar reproducibilidad.
    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("ENTRENAMIENTO DE CNN - CIFAR-10")
    print("=" * 55)
    print(f"Semilla: {SEED}")
    print(f"Dispositivo: {device}")
    print(f"Épocas: {EPOCHS}")
    print(f"Tasa de aprendizaje: {LEARNING_RATE}")

    # Cargar conjuntos y DataLoaders.
    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        base_dataset,
    ) = load_datasets()

    print("\nDatos preparados correctamente.")
    print(f"Imágenes de entrenamiento: {len(train_dataset)}")
    print(f"Imágenes de validación: {len(val_dataset)}")
    print(f"Imágenes de prueba: {len(test_dataset)}")

    # Crear el modelo.
    model = CNN(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_accuracy = -1.0

    # Reiniciar el historial de este experimento.
    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        ])

    print("\nComenzando entrenamiento...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        # Registrar resultados de cada época.
        with open(
            HISTORY_PATH, "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_accuracy:.4f}",
                f"{val_loss:.6f}",
                f"{val_accuracy:.4f}",
            ])

        print(
            f"Época [{epoch:02d}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

        # Guardar únicamente cuando mejora la validación.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                    "seed": SEED,
                },
                MODEL_PATH,
            )

            print("  -> Mejor modelo guardado.")

    print("\n" + "=" * 55)
    print("ENTRENAMIENTO FINALIZADO")
    print("=" * 55)
    print(f"Mejor precisión de validación: {best_val_accuracy:.2f}%")
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Historial guardado en: {HISTORY_PATH}")
    print("\nEl conjunto de prueba queda reservado para la")
    print("evaluación final del mejor modelo.")


if __name__ == "__main__":
    main()
