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

    return device


def main():
    tf = (
        transforms.ToTensor()
    )  # turns a 28x28 image into a float tensor in [0,1], shape (1,28,28)

    train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)

    xb, yb = next(iter(train_dl))

    print(xb.shape, yb.shape, yb[:10])


if __name__ == "__main__":
    main()
