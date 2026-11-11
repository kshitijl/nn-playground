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


def get_model(image_dim: int, image_channels: int, variant: str, device: str):
    def conv_layer(in_channels: int, out_channels: int):
        conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        relu = [nn.ReLU()]
        pool = [nn.MaxPool2d(2)]
        args = chain.from_iterable([[conv], [nn.BatchNorm2d(out_channels)], relu, pool])
        return nn.Sequential(*args)

    # (image_channels,id,id) -> (32, id/2, id/2)
    l1 = conv_layer(image_channels, 32)

    # (32, id/2, id/2) -> (64, id/4, id/4)
    l2 = conv_layer(32, 64)

    # (64,id/4,id/4) -> (10)
    d = image_dim // 4
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

    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_tf = (
        transforms.ToTensor()
    )  # turns a NxN image into a float tensor in [0,1], shape (channels,N,N)

    train_ds = datasets.CIFAR10("data", train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10("data", train=False, download=True, transform=test_tf)
    train_dl = DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=8, pin_memory=True
    )
    test_dl = DataLoader(
        test_ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=True
    )

    model = get_model(32, 3, args.variant, device)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 30
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_dl)
    )

    batch_num = 0
    running_loss = 0
    for epoch in range(epochs):
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()

            loss.backward()

            opt.step()
            sched.step()
            batch_num += 1

            running_loss += loss.item()

            if batch_num % 200 == 0:
                print(f"{batch_num} {running_loss / 200:.4}")
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
            print(f"test acc {test_acc:.2%}, test loss {test_loss / len(test_ds):.4}")

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

    fig, axes = plt.subplots(3, 5, figsize=(10, 7))
    for ax, mistake in zip(axes.flat, mistakes):
        img, p, t = mistake
        img = img.permute(1, 2, 0)
        ax.imshow(img)
        ax.set_title(
            f"pred {test_ds.classes[p]}, true {test_ds.classes[t]}", fontsize=8
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig("mistakes.png")


if __name__ == "__main__":
    main()
