import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}")
    return device


def main():
    device = get_device()
    tf = (
        transforms.ToTensor()
    )  # turns a 28x28 image into a float tensor in [0,1], shape (1,28,28)

    train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False)

    model = nn.Sequential(
        nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_num = 0
    running_loss = 0
    for epoch in range(10):
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()

            loss.backward()

            opt.step()
            batch_num += 1

            running_loss += loss.item()

            if batch_num % 20 == 0:
                print(batch_num, running_loss / 20)
                running_loss = 0

        print(f"epoch {epoch + 1} finish")
        model.eval()
        with torch.no_grad():
            test_profit = 0
            test_loss = 0
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                test_profit += (pred == yb).sum()
                test_loss += loss_fn(logits, yb).item() * len(yb)

            test_acc = test_profit.item() / len(test_ds)
            print(f"test acc {test_acc:.2%}, test loss {test_loss / len(test_ds)}")

        model.train()

    mistakes = []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)

            wrong = pred != yb
            wrong_idx = torch.where(wrong)[0]
            for idx in wrong_idx:
                mistakes.append((xb[idx].cpu(), pred[idx].item(), yb[idx].item()))

    fig, axes = plt.subplots(3, 5, figsize=(10, 4))
    for ax, mistake in zip(axes.flat, mistakes):
        img, p, t = mistake
        ax.imshow(img[0].cpu(), cmap="gray")
        ax.set_title(f"pred {p}, true {t}")
        ax.axis("off")

    fig.savefig("mistakes.png")


if __name__ == "__main__":
    main()
