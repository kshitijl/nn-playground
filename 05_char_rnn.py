import torch
from torch import nn
import math


def get_data() -> str:
    with open("data/input.txt", "r") as f:
        contents = f.read()

    print("-" * 80)
    print(contents[:500])
    print("-" * 80)
    print(f"The dataset has {len(contents):,} chars")

    return contents


def get_batch(data: torch.Tensor, B: int, T: int) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - T, size=(B,))
    offsets = torch.arange(T)  # [0, 1, ..., T-1]

    starts = starts.reshape(B, 1)
    offsets = offsets.reshape(1, T)

    idx = starts + offsets
    assert idx.shape == (B, T)

    x = data[idx]
    y = data[idx + 1]
    assert x.shape == (B, T)
    assert y.shape == (B, T)

    return (x, y)


class MyModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _hc = self.lstm(embedded)
        return self.linear(output)


def main():
    embed_dim = 128
    hidden_dim = 256
    contents = get_data()

    chars = sorted(set(contents))
    stoi = {c: idx for (idx, c) in enumerate(chars)}
    itos = {idx: c for (idx, c) in enumerate(chars)}
    vocab_size = len(chars)

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]

    def decode(ids: list[int]) -> str:
        return "".join([itos[idx] for idx in ids])

    print("-" * 80)
    print("Testing encode and decode")
    print(encode("hello"))
    print(decode(encode("hello")))
    print("-" * 80)

    data = torch.tensor(encode(contents))
    print(data.shape)

    train_length = int(0.9 * len(data))
    train, val = data[:train_length], data[train_length:]
    print("-" * 80)
    print(f"Size of training set: {len(train):,}. Val size: {len(val):,}")
    print("-" * 80)

    print("Building batches")
    B, T = 64, 128

    x, y = get_batch(train, B, T)
    print(decode(x[0].tolist()))
    print(decode(y[0].tolist()))
    print("Done building batches")

    model = MyModel(vocab_size, embed_dim, hidden_dim)

    logits = model(x)
    print(logits.shape)
    num_parameters = sum(x.numel() for x in model.parameters())
    print(f"The model has {num_parameters:,} parameters")

    loss_fn = nn.CrossEntropyLoss()
    initial_loss = loss_fn(logits.view(-1, vocab_size), y.view(-1)).item()
    expected_initial_loss = math.log(vocab_size)
    print(f"Initial loss is {initial_loss:.4}, expected {expected_initial_loss:.4}")


if __name__ == "__main__":
    main()
