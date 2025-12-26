import math

import matplotlib.pyplot as plt
import torch
from torch import nn


def make_dataset(num_samples: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-2 * math.pi, 2 * math.pi, num_samples).unsqueeze(1)
    y = torch.sin(x)
    return x, y


class SineRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    losses = []

    for epoch in range(2000):
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 400 == 0:
            print(f"Epoch {epoch + 1:4d} | Loss: {loss.item():.6f}")

    return losses


def plot_results(x: torch.Tensor, y: torch.Tensor, preds: torch.Tensor) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(x.numpy(), y.numpy(), label="True", linewidth=2)
    plt.plot(x.numpy(), preds.detach().numpy(), label="Predicted", linestyle="--")
    plt.legend()
    plt.title("Sine Wave Regression with PyTorch")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.tight_layout()
    plt.show()


def main() -> None:
    torch.manual_seed(42)

    x, y = make_dataset()
    model = SineRegressor()

    train(model, x, y)

    with torch.no_grad():
        preds = model(x)

    plot_results(x, y, preds)


if __name__ == "__main__":
    main()
