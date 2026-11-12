import torch
from torch import nn
import torch.nn.functional as F
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


def attention(q, k, v):
    T, d = q.shape[-2], q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)

    future = torch.ones((T, T), dtype=torch.bool, device=q.device).triu(diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    weights = torch.softmax(scores, dim=-1)

    answer = weights @ v
    return answer, weights


class MLP(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.fc1 = nn.Linear(C, 4 * C)
        self.fc2 = nn.Linear(4 * C, C)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Head(nn.Module):
    def __init__(self, C: int, head_size: int):
        super().__init__()
        self.query = nn.Linear(C, head_size)
        self.key = nn.Linear(C, head_size)
        self.value = nn.Linear(C, head_size)

    def forward(self, x):  # (B,T)
        q = self.query(x)  # (B, head_size)
        k = self.key(x)  # (B, head_size)
        v = self.value(x)  # (B, head_size)

        return attention(q, k, v)[0]


class MultiHead(nn.Module):
    def __init__(self, C: int, num_heads: int):
        super().__init__()
        head_size = C // num_heads
        self.heads = nn.ModuleList([Head(C, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(C, C)

    def forward(self, x):  # (B, T, C)
        # 4 x (B, T, 16) -> (B, T, 64)
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, C: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.attn = MultiHead(C, num_heads=num_heads)
        self.ln2 = nn.LayerNorm(C)
        self.mlp = MLP(C)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    # V: vocab size, C: embedding dim
    def __init__(self, V: int, C: int, block_size: int, n_layers: int, num_heads: int):
        super().__init__()
        self.tok_emb = nn.Embedding(V, C)
        self.pos_emb = nn.Embedding(block_size, C)
        self.blocks = nn.Sequential(*[Block(C, num_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(C)
        self.head = nn.Linear(C, V)

    def forward(self, x):  # (B, T)
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)  # (B, T, C) + (T, C)

        h = self.blocks(h)
        h = self.ln_f(h)
        return self.head(h)  # (B, T, V)


def main():
    torch.manual_seed(0)
    block_size = 128
    # C is the embed dim
    C = 128
    num_steps = 5000
    B, T = 64, 128
    num_layers = 4
    num_heads = 4

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

    print("\n\n\n")

    # model = MyModel(vocab_size, C, hidden_dim).to(device)
    model = Transformer(vocab_size, C, block_size, num_layers, num_heads).to(device)

    logits = model(x)
    print(logits.shape)
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"The model has {num_parameters:,} parameters")

    print("\n\n\n")

    loss_fn = nn.CrossEntropyLoss()
    initial_loss = loss_fn(logits.view(-1, vocab_size), y.view(-1)).item()
    expected_initial_loss = math.log(vocab_size)
    print(f"Initial loss is {initial_loss:.4}, expected {expected_initial_loss:.4}")

    print("\n\n")
    print("Causality test")
    model.eval()

    x, _ = get_batch(train, 4, 128)
    x = x.to(device)
    x2 = x.clone().to(device)
    x2[:, 60] = (x2[:, 60] + 1) % 65  # change one character
    with torch.no_grad():
        a, b = model(x), model(x2)
    assert torch.equal(a[:, :60], b[:, :60])  # should be True: past ignores future
    assert not torch.equal(a[:, 60:], b[:, 60:])  # should be False

    model.train()
    print("\n\n")

    def estimate_loss(data, num_batches: int):
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

                logits = model(ids[:, -block_size:])
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

    def lr_at(step, max_lr=1e-3, warmup=100, min_lr=1e-4):
        if step < warmup:
            return max_lr * (step + 1) / warmup
        progress = (step - warmup) / (num_steps - warmup)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(num_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(train, B, T)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()

        if step % 250 == 0 or step == num_steps - 1:
            num_batches = 20
            if step == num_steps - 1:
                num_batches = 200
            train_loss = estimate_loss(train, num_batches)
            val_loss = estimate_loss(val, num_batches)
            print(f"Step {step}, train loss {train_loss:.4}, val loss {val_loss:.4}")
            print(f"Sample text:{generate(100, 1.0)}")
            print("")

    torch.save(model.state_dict(), "char_lstm.pt")


if __name__ == "__main__":
    main()
