from itertools import chain
import argparse
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


def get_model(variant: str, device: str):
    should_conv_relu = True
    should_pool = True
    should_stride = False
    linear_head = False

    if variant == "no_conv_relu":
        should_conv_relu = False
    if variant == "no_pooling":
        should_pool = False
    if variant == "stride_instead_of_pool":
        should_stride = True
        should_pool = False
    if variant == "linear_head":
        linear_head = True

    def conv_layer(in_channels: int, out_channels: int):
        if should_stride:
            assert not should_pool
            conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2)
        else:
            conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        relu = [nn.ReLU()] if should_conv_relu else []
        pool = [nn.MaxPool2d(2)] if should_pool else []
        args = chain.from_iterable([[conv], relu, pool])
        return nn.Sequential(*args)

    # (1,28,28) -> (32, 14, 14)
    l1 = conv_layer(1, 32)

    # (32, 14, 14) -> (64, 7, 7)
    l2 = conv_layer(32, 64)

    # (64,7,7) -> (10)
    if should_pool or should_stride:
        d = 7
    else:
        d = 28
    if linear_head:
        l3 = nn.Sequential(nn.Flatten(), nn.Linear(64 * d * d, 10))
    else:
        l3 = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * d * d, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    model = nn.Sequential(l1, l2, l3).to(device)

    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"num parameters: {num_parameters:,}")
    return model


def main():
    torch.manual_seed(1)
    p = argparse.ArgumentParser()
    p.add_argument(
        "variant",
        choices=[
            "baseline",
            "no_conv_relu",
            "no_pooling",
            "stride_instead_of_pool",
            "linear_head",
        ],
    )
    args = p.parse_args()

    device = get_device()
    tf = (
        transforms.ToTensor()
    )  # turns a 28x28 image into a float tensor in [0,1], shape (1,28,28)

    train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False)

    model = get_model(args.variant, device)
    loss_fn = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_num = 0
    running_loss = 0
    for epoch in range(5):
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()

            loss.backward()

            opt.step()
            batch_num += 1

            running_loss += loss.item()

            if batch_num % 200 == 0:
                print(f"{batch_num} {running_loss / 200:.2}")
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
            print(f"test acc {test_acc:.2%}, test loss {test_loss / len(test_ds):.2}")

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
