"""Exporta o Pocket-LM para GGUF, o formato do llama.cpp.

Por que GGUF e nao .tflite pra chegar no celular HOJE: o .tflite exige um aplicativo
Android proprio pra carregar, e construir isso leva dias. GGUF e lido por aplicativos que
ja existem na Play Store (PocketPal, ChatterUI, Layla) — voce instala o app, copia o
arquivo e usa offline. A NPU continua sendo o destino final; isto e o caminho curto.

A conversao e direta porque o Pocket-LM E arquitetura GPT-2:
- embedding de posicao APRENDIDA (nao RoPE)      -> GPT-2 `wpe`
- pre-norm com LayerNorm                          -> GPT-2 `ln_1`/`ln_2`
- MLP GELU 4x                                     -> GPT-2 `mlp`
- embeddings amarrados (tok.weight == head.weight)-> GPT-2 amarra tambem
- QKV fundido num tensor (3E, E)                  -> GPT-2 `c_attn`

Uma armadilha: o GPT-2 original usa `Conv1D`, que guarda os pesos TRANSPOSTOS em relacao
a `nn.Linear`. O gguf espera a convencao do llama.cpp, que e a de `nn.Linear` — entao os
pesos vao como estao. Se sair texto sem sentido no aplicativo, e aqui que se olha
primeiro.

  python exporta_gguf.py --ckpt gpu_out/pocket-estilo-v1/ckpt.pt \
                         --tokenizer gpu_out/pocket-estilo-v1/tokenizer.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

from model import GPTConfig  # noqa: E402


def tokens_do_tokenizer(caminho: str):
    """Extrai vocabulario e merges do tokenizer.json (ByteLevelBPE).

    INCLUI `added_tokens`, e isso nao e detalhe. Os tokens acrescentados depois do
    treinamento do BPE — `<|im_start|>`, `<|im_end|>` e os reservados `<|unused_N|>` —
    NAO ficam em `model.vocab`; ficam numa secao separada `added_tokens` do tokenizer.json.
    Lendo so `model.vocab`, o export escrevia 16.384 nomes para uma matriz de 16.512 linhas,
    e o llama.cpp recusava o arquivo com "tensor 'token_embd.weight' has wrong shape;
    expected 1024,16384, got 1024,16512" — depois do treino inteiro ja gasto.

    Devolve tambem quais ids sao especiais, para marcar como CONTROL em vez de NORMAL:
    token de controle marcado como normal pode ser emitido como texto pelo modelo.
    """
    d = json.load(open(caminho, encoding="utf-8"))
    vocab = dict(d["model"]["vocab"])
    merges = d["model"].get("merges") or []
    especiais = set()
    for a in (d.get("added_tokens") or []):
        vocab[a["content"]] = a["id"]
        if a.get("special"):
            especiais.add(a["id"])
    n = max(vocab.values()) + 1
    palavras = [None] * n
    for tok, i in vocab.items():
        palavras[i] = tok
    buracos = [i for i, p in enumerate(palavras) if p is None]
    if buracos:
        raise SystemExit(f"vocabulario com buraco: {len(buracos)} ids sem token, "
                         f"primeiros {buracos[:5]}")
    merges_txt = [" ".join(m) if isinstance(m, (list, tuple)) else m for m in merges]
    return palavras, merges_txt, especiais


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--saida", default=None)
    ap.add_argument("--nome", default="pocket-lm")
    a = ap.parse_args()

    from gguf import GGUFWriter, TokenType

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfgd = ck["cfg"]
    cfg = cfgd if isinstance(cfgd, GPTConfig) else GPTConfig(**dict(cfgd))
    sd = ck["model"]
    n_params = sum(v.numel() for v in sd.values())
    saida = a.saida or os.path.join(DIR, "gguf", f"{a.nome}-f16.gguf")
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    print(json.dumps({"fase": "carregado", "params": n_params,
                      "n_layer": cfg.n_layer, "n_embd": cfg.n_embd,
                      "n_head": cfg.n_head, "block": cfg.block_size,
                      "vocab": cfg.vocab_size}), flush=True)

    w = GGUFWriter(saida, "gpt2")
    w.add_name(a.nome)
    w.add_context_length(cfg.block_size)
    w.add_embedding_length(cfg.n_embd)
    w.add_feed_forward_length(4 * cfg.n_embd)
    w.add_block_count(cfg.n_layer)
    w.add_head_count(cfg.n_head)
    w.add_layer_norm_eps(1e-5)
    w.add_file_type(1)                       # F16

    palavras, merges, especiais = tokens_do_tokenizer(a.tokenizer)
    if len(palavras) != cfg.vocab_size:
        raise SystemExit(f"TAMANHO NAO BATE: tokenizer tem {len(palavras)} tokens e a "
                         f"matriz do modelo tem {cfg.vocab_size}. Corrija antes de exportar "
                         f"— o llama.cpp so recusaria o arquivo no fim.")
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("gpt-2")
    w.add_token_list(palavras)
    w.add_token_types([TokenType.CONTROL if (p == "<|endoftext|>" or i in especiais)
                       else TokenType.NORMAL
                       for i, p in enumerate(palavras)])
    if merges:
        w.add_token_merges(merges)
    w.add_bos_token_id(0)
    w.add_eos_token_id(0)                    # <|endoftext|> e o id 0 no nosso BPE
    w.add_add_bos_token(True)                # todo documento do treino vem apos um EOT

    # Template de conversa DENTRO do arquivo: sem isto o aplicativo manda o texto cru e
    # o modelo volta a ser continuador — foi o que aconteceu no primeiro teste no
    # celular ("Oi" devolveu um trecho sobre media ponderada). Com o template embutido,
    # PocketPal/llama.cpp aplicam sozinhos.
    w.add_chat_template(
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "Usuário: {{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}"
        "Assistente: {{ message['content'] }}\n"
        "{% elif message['role'] == 'system' %}"
        "{{ message['content'] }}\n"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}Assistente:{% endif %}"
    )

    def po(t: torch.Tensor) -> np.ndarray:
        """2-D vai em F16; 1-D (normas e vieses) fica em F32.

        Convencao do llama.cpp: os conversores oficiais so quantizam tensores 2-D. Pôr
        norma em F16 faz o ggml_mul do LayerNorm abortar em tempo de inferencia — o
        modelo CARREGA normalmente e so quebra na primeira geracao.
        """
        t = t.detach()
        return t.to(torch.float16 if t.ndim >= 2 else torch.float32).numpy()

    # embeddings e saida (amarrados: o mesmo tensor serve aos dois)
    w.add_tensor("token_embd.weight", po(sd["tok.weight"]))
    w.add_tensor("position_embd.weight", po(sd["pos.weight"]))
    w.add_tensor("output_norm.weight", po(sd["lnf.weight"]))
    w.add_tensor("output_norm.bias", po(sd["lnf.bias"]))
    # NAO gravar `output.weight`: o modelo usa weight tying, entao ele seria byte por byte
    # o mesmo conteudo de `token_embd.weight`. O llama.cpp cai de volta no `token_embd`
    # quando `output` esta ausente, e o `llama-quantize` PROMOVE o `token_embd` para Q6_K
    # ao perceber que ele passou a servir tambem de cabeca de saida.
    #
    # Medido em 03/08/2026, modelo ext3, Q4_K_M:
    #   com duplicata:  177,2 MiB | perplexidade 20,3032 +/- 0,248
    #   sem duplicata:  168,2 MiB | perplexidade 20,3024 +/- 0,248
    # 5,1% menor, diferenca de qualidade mil vezes menor que a margem de erro. Como
    # geracao em celular e limitada por banda de memoria, 5,1% de arquivo e ~5% de
    # velocidade — de graca.
    #
    # A cabeca NAO perde precisao: antes eram `output` Q6_K (13,76 MB) mais `token_embd`
    # Q4_K (9,44); agora e so `token_embd` Q6_K (13,76). A entrada ate melhorou.
    #
    # COM_OUTPUT_DUPLICADO=1 volta ao comportamento antigo, se algum consumidor de GGUF
    # exigir o tensor explicito.
    if os.environ.get("COM_OUTPUT_DUPLICADO") == "1":
        w.add_tensor("output.weight", po(sd["head.weight"]))

    for i in range(cfg.n_layer):
        p = f"blocks.{i}."
        b = f"blk.{i}."
        w.add_tensor(b + "attn_norm.weight", po(sd[p + "ln1.weight"]))
        w.add_tensor(b + "attn_norm.bias", po(sd[p + "ln1.bias"]))
        # QKV ja vem fundido em (3E, E) — e exatamente o c_attn do GPT-2
        w.add_tensor(b + "attn_qkv.weight", po(sd[p + "attn.in_proj_weight"]))
        w.add_tensor(b + "attn_qkv.bias", po(sd[p + "attn.in_proj_bias"]))
        w.add_tensor(b + "attn_output.weight", po(sd[p + "attn.out_proj.weight"]))
        w.add_tensor(b + "attn_output.bias", po(sd[p + "attn.out_proj.bias"]))
        w.add_tensor(b + "ffn_norm.weight", po(sd[p + "ln2.weight"]))
        w.add_tensor(b + "ffn_norm.bias", po(sd[p + "ln2.bias"]))
        w.add_tensor(b + "ffn_up.weight", po(sd[p + "mlp.0.weight"]))
        w.add_tensor(b + "ffn_up.bias", po(sd[p + "mlp.0.bias"]))
        w.add_tensor(b + "ffn_down.weight", po(sd[p + "mlp.2.weight"]))
        w.add_tensor(b + "ffn_down.bias", po(sd[p + "mlp.2.bias"]))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    mb = os.path.getsize(saida) / 1e6
    print(json.dumps({"fase": "pronto", "arquivo": os.path.relpath(saida, DIR),
                      "mb": round(mb, 1), "params": n_params,
                      "estimativa_q4_mb": round(n_params * 0.55 / 1e6, 1)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
