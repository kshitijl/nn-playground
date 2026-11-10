import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn


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
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=True)

    model = nn.Sequential(
        nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_num = 0
    for epoch in range(3):
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()

            loss.backward()

            opt.step()
            batch_num += 1

            if batch_num % 20 == 0:
                print(batch_num, loss.item())

        print(f"epoch {epoch} finish")
        model.eval()
        with torch.no_grad():
            test_profit = 0
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                test_profit += (pred == yb).sum()

            print("test profit", test_profit.item())

        model.train()


if __name__ == "__main__":
    main()
