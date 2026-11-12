import torch
from torch import nn
import math


def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}")
    return device


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


class Bigram(nn.Module):
    # V: vocab size, C: embedding dim
    def __init__(self, V: int, C: int):
        super().__init__()
        self.tok_emb = nn.Embedding(V, C)
        self.head = nn.Linear(C, V)

    def forward(self, x):  # (B, T)
        h = self.tok_emb(x)  # (B, T, C)
        return self.head(h)  # (B, T, V)


def main():
    torch.manual_seed(0)

    embed_dim = 128
    hidden_dim = 256
    num_steps = 5000
    B, T = 64, 128

    device = get_device()

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

    x, y = get_batch(train, B, T)
    x, y = x.to(device), y.to(device)
    print(decode(x[0].tolist()))
    print(decode(y[0].tolist()))
    print("Done building batches")

    # model = MyModel(vocab_size, embed_dim, hidden_dim).to(device)
    model = Bigram(vocab_size, 64).to(device)

    logits = model(x)
    print(logits.shape)
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"The model has {num_parameters:,} parameters")

    loss_fn = nn.CrossEntropyLoss()
    initial_loss = loss_fn(logits.view(-1, vocab_size), y.view(-1)).item()
    expected_initial_loss = math.log(vocab_size)
    print(f"Initial loss is {initial_loss:.4}, expected {expected_initial_loss:.4}")

    def estimate_loss(data):
        num_batches = 20
        model.eval()

        with torch.no_grad():
            total_loss = 0

            for idx in range(num_batches):
                x, y = get_batch(data, B, T)
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
                total_loss += loss.item()

        model.train()

        return total_loss / num_batches

    def generate(n: int, temperature: float):
        model.eval()
        ids = torch.tensor(encode("\n")).reshape(1, 1).to(device)

        assert ids.shape == (1, 1)

        with torch.no_grad():
            for idx in range(n):
                assert ids.shape == (1, idx + 1)

                logits = model(ids)
                assert logits.shape == (1, idx + 1, vocab_size)

                last_token_logits = logits[:, -1, :]
                assert last_token_logits.shape == (1, vocab_size)

                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                assert probs.shape == (1, vocab_size)

                token = torch.multinomial(probs, 1)
                assert token.shape == (1, 1)

                ids = torch.cat([ids, token], dim=1)
                assert ids.shape == (1, idx + 2)

        model.train()
        ids = ids.reshape(-1)
        return decode(ids.tolist())

    print(f"Initial generation is {generate(100, 1.0)}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(num_steps):
        x, y = get_batch(train, B, T)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()

        if step % 250 == 0 or step == num_steps - 1:
            train_loss = estimate_loss(train)
            val_loss = estimate_loss(val)
            print(f"Step {step}, train loss {train_loss:.4}, val loss {val_loss:.4}")
            print(f"Sample text:{generate(100, 1.0)}")
            print("")

    torch.save(model.state_dict(), "char_lstm.pt")


if __name__ == "__main__":
    main()
