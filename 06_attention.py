import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, T, C = 2, 5, 8
q, k, v = torch.randn(B, T, C), torch.randn(B, T, C), torch.randn(B, T, C)


def attention(q, k, v):
    assert q.shape == (B, T, C)
    assert k.shape == (B, T, C)
    assert v.shape == (B, T, C)

    scores = q @ k.transpose(-2, -1) / math.sqrt(C)
    assert scores.shape == (B, T, T)

    future = torch.ones((T, T), dtype=torch.bool).triu(diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    weights = torch.softmax(scores, dim=-1)

    answer = weights @ v
    assert answer.shape == (B, T, C)
    return answer, weights


out, weights = attention(q, k, v)
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(torch.allclose(out, ref, atol=1e-6))
print(weights[0])
