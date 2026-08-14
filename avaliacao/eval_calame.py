"""CALAME-PT: o modelo acerta a ULTIMA palavra da frase?

Mede esquecimento. O SFT reescreve os pesos para uma tarefa nova (QA extrativo em ChatML);
a pergunta e quanto o modelo perdeu do que sabia antes. Por isso este script roda o MESMO
codigo no modelo base e no ajustado — comparar numero proprio com numero copiado de paper
nao mede nada, porque implementacao diferente da resultado diferente (ACL C4).

DUAS COISAS QUE MUDAM O NUMERO E QUASE NINGUEM REPORTA:

1. **Reportar `handwritten` e `generated` separados.** O CALAME-PT tem 2.076 exemplos, mas
   1.670 deles foram GERADOS por GPT-3.5 e so 406 sao escritos a mao. A media agregada
   fica ~80% dominada por dado sintetico — e um modelo pode ir bem no sintetico por imitar
   o estilo do gerador, sem ler melhor. O numero honesto e a tabela, nao a media.

2. **O tokenizador do modelo ajustado tem 16.512 tokens** (ChatML + reservados) contra
   16.384 do base. Avaliar os dois com o mesmo arquivo de tokenizador produziria
   desalinhamento silencioso de ids. Cada modelo usa o seu.

A tarefa e continuacao crua: nada de ChatML aqui. E exatamente isso que testa se a
capacidade de modelagem de linguagem sobreviveu ao ajuste.

  ./venv/bin/python eval_calame.py gpu_out/ext3/slim.pt:data_big/tokenizer.json \
                                   gpu_out/sft-qa-v1/slim.pt:gpu_out/sft-qa-v1/tokenizer_chat.json
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


def normaliza(t: str) -> str:
    """ignore_case + ignore_punctuation, como a implementacao de referencia do CALAME.
    Acento NAO e removido: em portugues ele distingue palavra."""
    t = t.lower().strip()
    t = "".join(c for c in t if not unicodedata.category(c).startswith("P"))
    return " ".join(t.split())


def carrega(caminho: str, tok_path: str):
    d = torch.load(caminho, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**d["cfg"])
    m = GPT(cfg)
    m.load_state_dict(d["model"], strict=True)
    m.eval()
    return m, cfg, Tokenizer.from_file(tok_path)


@torch.no_grad()
def prediz(m, cfg, tok, contexto: str, max_novos: int = 6) -> str:
    """Geracao gulosa ate o primeiro separador de palavra. O CALAME pede UMA palavra;
    gerar mais e depois cortar e o que a implementacao de referencia faz."""
    ids = tok.encode(contexto, add_special_tokens=False).ids[-(cfg.block_size - max_novos):]
    saida: list[int] = []
    for _ in range(max_novos):
        x = torch.tensor([ids], dtype=torch.long)
        logits, _ = m(x)
        prox = int(torch.argmax(logits[0, -1]))
        saida.append(prox)
        ids.append(prox)
        texto = tok.decode(saida)
        # parou de escrever a palavra: espaco ou pontuacao forte depois de ter algo
        if texto.strip() and re.search(r"[\s.,;:!?)\]]", texto[len(texto.strip()) and 1:] or texto):
            break
    bruto = tok.decode(saida).strip()
    return bruto.split()[0] if bruto.split() else ""


def main() -> int:
    alvos = []
    for arg in sys.argv[1:]:
        if ":" in arg:
            c, t = arg.rsplit(":", 1)
        else:
            c, t = arg, "data_big/tokenizer.json"
        alvos.append((c, t))
    if not alvos:
        print("uso: eval_calame.py ckpt.pt:tokenizer.json [outro.pt:tok.json ...]")
        return 1

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    base = "datasets/NOVA-vision-language/calame-pt@refs%2Fconvert%2Fparquet"
    linhas = []
    for cfg_dir in fs.ls(base, detail=False):
        nome = cfg_dir.rstrip("/").split("/")[-1]
        # PULA o subset `all`: ele NAO e um terceiro conjunto, e a uniao dos outros dois
        # (2.076 = 1.670 gerados + 406 escritos a mao). Iterar sobre os tres contaria cada
        # item duas vezes e produziria um "n" inflado com a mesma acuracia — erro que passa
        # despercebido porque a porcentagem nao muda.
        if nome in (".gitattributes", "all"):
            continue
        for split in fs.ls(cfg_dir, detail=False):
            for arq in fs.ls(split, detail=False):
                if not arq.endswith(".parquet"):
                    continue
                with fs.open(arq, "rb") as fp:
                    for r in pq.ParquetFile(fp).read().to_pylist():
                        r["_subset"] = nome
                        linhas.append(r)
    if not linhas:
        print("nao consegui carregar o CALAME-PT")
        return 1
    cols = sorted(linhas[0].keys())
    print(json.dumps({"itens": len(linhas), "colunas": cols,
                      "subsets": sorted({l["_subset"] for l in linhas})}, ensure_ascii=False),
          flush=True)

    col_ctx = "sentence" if "sentence" in cols else ("context" if "context" in cols else cols[0])
    col_alvo = ("last_word" if "last_word" in cols
                else ("target" if "target" in cols else cols[1]))

    resultado = {}
    for caminho, tok_path in alvos:
        m, cfg, tok = carrega(caminho, tok_path)
        por = {}
        for k, r in enumerate(linhas):
            sub = r["_subset"]
            d = por.setdefault(sub, [0, 0])
            acerto = normaliza(prediz(m, cfg, tok, r[col_ctx])) == normaliza(str(r[col_alvo]))
            d[0] += acerto
            d[1] += 1
            if (k + 1) % 250 == 0:
                print(json.dumps({"modelo": caminho, "feitos": k + 1}), flush=True)
        tot_a = sum(v[0] for v in por.values())
        tot_n = sum(v[1] for v in por.values())
        resultado[caminho] = {
            "all": round(100 * tot_a / max(1, tot_n), 2),
            **{s: round(100 * v[0] / max(1, v[1]), 2) for s, v in sorted(por.items())},
            "n": tot_n,
            "n_por_subset": {s: v[1] for s, v in sorted(por.items())},
        }
        print(json.dumps({caminho: resultado[caminho]}, ensure_ascii=False), flush=True)
        del m

    print()
    subs = sorted({s for v in resultado.values() for s in v if s not in ("n", "n_por_subset")})
    print(f"{'modelo':<34}" + "".join(f"{s:>14}" for s in subs))
    for c, v in resultado.items():
        print(f"{c.split('/')[-2] if '/' in c else c:<34}" +
              "".join(f"{v.get(s, float('nan')):>14.2f}" for s in subs))
    with open("avaliacao/calame.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
