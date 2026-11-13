import torch
from torch import nn
import torch.nn.functional as F
import math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

dropout_p = 0.2


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

    print(f"The dataset has {len(contents):,} chars")

    return contents


def attention(q, k, v, dropout):
    T, d = q.shape[-2], q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)

    future = torch.ones((T, T), dtype=torch.bool, device=q.device).triu(diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    weights = torch.softmax(scores, dim=-1)

    weights = dropout(weights)

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
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):  # (B,T)
        q = self.query(x)  # (B, head_size)
        k = self.key(x)  # (B, head_size)
        v = self.value(x)  # (B, head_size)

        answer, weights = attention(q, k, v, self.dropout)
        self.last_weights = weights.detach()
        return answer


class MultiHead(nn.Module):
    def __init__(self, C: int, num_heads: int):
        super().__init__()
        assert C % num_heads == 0
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
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class Transformer(nn.Module):
    # V: vocab size, C: embedding dim
    def __init__(self, V: int, C: int, block_size: int, n_layers: int, num_heads: int):
        super().__init__()
        self.tok_emb = nn.Embedding(V, C)
        self.pos_emb = nn.Embedding(block_size, C)
        self.blocks = nn.Sequential(*[Block(C, num_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, V)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):  # (B, T)
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)  # (B, T, C) + (T, C)
        h = self.dropout(h)
        h = self.blocks(h)
        h = self.ln_f(h)
        return self.lm_head(h)  # (B, T, V)


def main():
    torch.manual_seed(0)
    # C is the embed dim
    C = 384
    T = 256
    num_layers = 6
    num_heads = 6

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

    data = torch.tensor(encode(contents))
    print(data.shape)

    print("\n\n\n")

    model = Transformer(vocab_size, C, T, num_layers, num_heads).to(device)
    model.load_state_dict(torch.load("char_transformer.pt", weights_only=True))
    model.eval()

    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"The model has {num_parameters:,} parameters")

    text = """GONZALO:
I do well believe your highness; and
did it to minister occasion to these gentlemen,
who are of such sensible and nimble lungs that
    """

    encoded_text = torch.tensor(encode(text)).reshape(1, -1).to(device)

    with torch.no_grad():
        model(encoded_text)

    print(model.blocks[0].attn.heads[0].last_weights.shape, len(text))

    last_weights = []

    for layer_idx in range(num_layers):
        this_layer = []
        for head_idx in range(num_heads):
            weights = model.blocks[layer_idx].attn.heads[head_idx].last_weights
            assert torch.allclose(torch.sum(weights, dim=-1), torch.tensor(1.0))
            assert torch.allclose(torch.triu(weights, diagonal=1), torch.tensor(0.0))
            weights = weights[0]
            assert weights.shape == (len(text), len(text))
            this_layer.append(weights.cpu())
        last_weights.append(this_layer)

    char_label_dict = {"\n": "\\n", " ": "_"}
    char_labels = [char_label_dict.get(c, c) for c in list(text)]

    fig, axes = plt.subplots(
        num_layers, num_heads, figsize=(num_layers * 6, num_heads * 6)
    )

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            head_weights = last_weights[layer_idx][head_idx]
            ax = axes[layer_idx, head_idx]

            ax.imshow(head_weights, cmap="Blues", vmin=0, vmax=1.0)
            ax.tick_params(axis="x", labeltop=True)
            ax.set_xticks(range(len(text)), char_labels, fontsize=5)
            ax.set_yticks(range(len(text)), char_labels, fontsize=5)

            if head_idx == 0:
                ax.set_ylabel(f"Layer {layer_idx + 1}")

            if layer_idx == 0:
                ax.set_title(f"Head {head_idx + 1}")

    fig.tight_layout()
    fig.savefig("activations.png", dpi=300)


if __name__ == "__main__":
    main()
