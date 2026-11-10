import torch
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
    device = get_device()
    print(f"Device is {device}")

    # fake data: y = 3x + 2 plus noise
    x = torch.randn(1000, 1, device=device)
    y = 3 * x + 2 + 0.1 * torch.randn_like(x)

    model = nn.Linear(1, 1).to(device)  # one weight, one bias
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    for step in range(200):
        # forward pass
        pred = model(x)
        loss = loss_fn(pred, y)

        # clear old gradients
        opt.zero_grad()

        # compute gradients
        loss.backward()

        opt.step()
        if step % 20 == 0:
            print(step, loss.item())

    print(f"Learned: {model.weight.item()}, {model.bias.item()}")


if __name__ == "__main__":
    main()
