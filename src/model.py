
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Bloque 1: 3 x 32 x 32 -> 32 x 16 x 16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Bloque 2: 32 x 16 x 16 -> 64 x 8 x 8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Bloque 3: 64 x 8 x 8 -> 128 x 4 x 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=0.30),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = CNN(num_classes=10)
    x = torch.randn(8, 3, 32, 32)

    with torch.no_grad():
        features = model.features(x)
        output = model(x)

    print("========== PRUEBA DE LA CNN ==========")
    print("Entrada:", x.shape)
    print("Salida convolucional:", features.shape)
    print("Salida final:", output.shape)
    print("Parámetros entrenables:",
          sum(p.numel() for p in model.parameters()
              if p.requires_grad))

    assert features.shape == (8, 128, 4, 4)
    assert output.shape == (8, 10)

    print("Prueba completada correctamente.")
