
import csv
from pathlib import Path

import torch
import torch.nn as nn

try:
    from .dataset import SEED, set_seed, load_datasets, CLASSES
    from .model import CNN
except ImportError:
    from dataset import SEED, set_seed, load_datasets, CLASSES
    from model import CNN


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "model_best.pth"


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 55)
    print("EVALUACIÓN FINAL - CNN CIFAR-10")
    print("=" * 55)
    print(f"Dispositivo: {device}")

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        base_dataset,
    ) = load_datasets()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {MODEL_PATH}"
        )

    model = CNN(num_classes=10).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total = 0

    num_classes = len(CLASSES)
    confusion = [
        [0 for _ in range(num_classes)]
        for _ in range(num_classes)
    ]

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total += batch_size

            for actual, predicted in zip(
                labels.cpu().tolist(),
                predictions.cpu().tolist(),
            ):
                confusion[actual][predicted] += 1

    test_loss = total_loss / total
    test_accuracy = 100.0 * total_correct / total

    print("\n========== RESULTADOS DE TEST ==========")
    print(f"Modelo seleccionado en época: {checkpoint['epoch']}")
    print(f"Precisión de validación guardada: "
          f"{checkpoint['val_accuracy']:.2f}%")
    print(f"Pérdida de test: {test_loss:.4f}")
    print(f"Precisión de test: {test_accuracy:.2f}%")

    print("\n========== PRECISIÓN POR CLASE ==========")

    class_accuracies = []

    for i, class_name in enumerate(CLASSES):
        class_total = sum(confusion[i])
        class_correct = confusion[i][i]

        accuracy = (
            100.0 * class_correct / class_total
            if class_total else 0.0
        )

        class_accuracies.append(accuracy)

        print(
            f"{class_name:12s}: {accuracy:6.2f}% "
            f"({class_correct}/{class_total})"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = OUTPUT_DIR / "test_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write("EVALUACIÓN FINAL CNN - CIFAR-10\n")
        file.write(f"Semilla: {SEED}\n")
        file.write(f"Época seleccionada: {checkpoint['epoch']}\n")
        file.write(
            f"Precisión de validación: "
            f"{checkpoint['val_accuracy']:.2f}%\n"
        )
        file.write(f"Pérdida de test: {test_loss:.6f}\n")
        file.write(f"Precisión de test: {test_accuracy:.4f}%\n")
        file.write("\nPrecisión por clase:\n")

        for name, accuracy in zip(CLASSES, class_accuracies):
            file.write(f"{name}: {accuracy:.2f}%\n")

    confusion_path = OUTPUT_DIR / "confusion_matrix.csv"
    with open(
        confusion_path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Real/Predicho"] + list(CLASSES))

        for class_name, row in zip(CLASSES, confusion):
            writer.writerow([class_name] + row)

    print("\nArchivos guardados:")
    print(f"- {metrics_path}")
    print(f"- {confusion_path}")


if __name__ == "__main__":
    main()
