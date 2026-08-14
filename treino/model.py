"""GPT char-level minúsculo (estilo nanoGPT), pra treinar em CPU.
Não é o modelo final do projeto — é o brinquedo que prova a esteira toda de graça.
O modelo "de verdade" (BPE, maior) roda na GPU alugada depois."""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 192
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Atencao causal com SDPA e QKV fundido.

    Substitui nn.MultiheadAttention, que desliga o fast-path assim que recebe um
    attn_mask explicito e cai no caminho lento — parte do motivo da MFU ter ficado em
    ~11% na H200. F.scaled_dot_product_attention com is_causal=True usa o kernel
    fundido e dispensa materializar a mascara TxT.

    Os PARAMETROS mantem de proposito os nomes de nn.MultiheadAttention
    (in_proj_weight, in_proj_bias, out_proj.weight, out_proj.bias): o state_dict fica
    identico, entao checkpoint antigo — inclusive o de 219M do run pocket-h200-v2 —
    carrega sem conversao nenhuma.

    Bonus pra fase 3: o export pra NPU ja exigia SDPA decomposto no lugar do
    nn.MultiheadAttention, entao treino e export passam a usar o mesmo caminho.
    """

    def __init__(self, c: GPTConfig):
        super().__init__()
        assert c.n_embd % c.n_head == 0
        self.n_head = c.n_head
        self.n_embd = c.n_embd
        self.p_drop = c.dropout
        self.in_proj_weight = nn.Parameter(torch.empty(3 * c.n_embd, c.n_embd))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * c.n_embd))
        self.out_proj = nn.Linear(c.n_embd, c.n_embd)
        nn.init.xavier_uniform_(self.in_proj_weight)   # mesmo init do MultiheadAttention

    def forward(self, x):
        B, T, C = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.p_drop if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.attn = CausalSelfAttention(c)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(c.n_embd, 4 * c.n_embd),
            nn.GELU(),
            nn.Linear(4 * c.n_embd, c.n_embd),
            nn.Dropout(c.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos = nn.Embedding(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.lnf = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.tok.weight = self.head.weight  # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        T = idx.size(1)
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        for b in self.blocks:
            x = b(x)
        x = self.lnf(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            # reshape, nao view: o chamador costuma passar uma FATIA (lote[:, 1:]),
            # que nao e contigua, e view falha nesse caso
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            cond = idx[:, -self.c.block_size:]
            logits, _ = self(cond)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx
