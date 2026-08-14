"""Fase de dados: baixa texto pt-BR em pedaços (streaming da Wikipedia pt), limpa e
acumula em data/corpus.txt até atingir a meta. Roda um PEDAÇO por noite — idempotente
e retomável. Quando a meta é atingida, constrói o vocabulário char-level e os splits
train/val em .npy, e sinaliza que o treino pode começar.

Escolha consciente: char-level (sem tokenizer BPE) porque é robusto pra rodar sozinho e
já mostra o modelo aprendendo estrutura do português. O tokenizer BPE de verdade entra
na fase de GPU."""
from __future__ import annotations
import json
import os
import re
import sys
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "data")
CORPUS = os.path.join(DATA, "corpus.txt")
META_MB = float(os.environ.get("CORPUS_META_MB", "40"))   # alvo total
CHUNK_DOCS = int(os.environ.get("PREP_CHUNK_DOCS", "4000"))  # artigos por noite


def limpa(txt: str) -> str:
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    return txt.strip()


def tamanho_mb(path: str) -> float:
    return os.path.getsize(path) / 1e6 if os.path.exists(path) else 0.0


def baixa_pedaco() -> dict:
    """Puxa CHUNK_DOCS artigos novos da Wikipedia pt e acrescenta ao corpus.
    Usa streaming e pula os que já foram vistos (via contador salvo)."""
    from datasets import load_dataset

    state_f = os.path.join(DATA, "prep_state.json")
    st = json.load(open(state_f)) if os.path.exists(state_f) else {"pulados": 0}
    ja = st.get("pulados", 0)

    ds = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)
    escritos = 0
    with open(CORPUS, "a", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if i < ja:
                continue
            t = limpa(ex.get("text", ""))
            if len(t) > 200:
                f.write(t + "\n\n")
                escritos += 1
            if escritos >= CHUNK_DOCS:
                break
    st["pulados"] = ja + CHUNK_DOCS
    json.dump(st, open(state_f, "w"))
    return {"artigos_add": escritos, "corpus_mb": round(tamanho_mb(CORPUS), 1)}


def constroi_dataset() -> dict:
    """Vocabulário char-level + train/val em uint16 .npy. Só quando a meta é atingida."""
    texto = open(CORPUS, encoding="utf-8").read()
    chars = sorted(set(texto))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in texto], dtype=np.uint16)
    n = len(ids)
    corte = int(n * 0.95)
    ids[:corte].tofile(os.path.join(DATA, "train.bin"))
    ids[corte:].tofile(os.path.join(DATA, "val.bin"))
    json.dump({"chars": chars, "vocab_size": len(chars)},
              open(os.path.join(DATA, "vocab.json"), "w"), ensure_ascii=False)
    return {"vocab_size": len(chars), "tokens": int(n),
            "train_tokens": corte, "val_tokens": int(n - corte)}


def main():
    os.makedirs(DATA, exist_ok=True)
    mb = tamanho_mb(CORPUS)
    if mb >= META_MB:
        info = constroi_dataset()
        print(json.dumps({"fase": "dados_prontos", **info}, ensure_ascii=False))
        return 0  # 0 = prep concluída
    r = baixa_pedaco()
    falta = round(META_MB - r["corpus_mb"], 1)
    print(json.dumps({"fase": "coletando", "meta_mb": META_MB, "falta_mb": max(0, falta), **r},
                     ensure_ascii=False))
    return 1  # 1 = ainda coletando


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    # os._exit: o streaming do HF deixa threads penduradas que travam a saída normal.
    # Tudo já foi gravado em disco (arquivos fechados), então sair "à força" é seguro.
    os._exit(rc)
