"""Tokeniza acoes/executor_treino.jsonl em .bin uint16 pro pré-treino from-scratch do EXECUTOR.

Cada exemplo é `entrada + alvo` = "frase: <fala>\\nacao: <intent>;<campos>" — exatamente o
que o modelo vê na inferência. Tokeniza com data_big/tokenizer.json (vocab 16384) e grava
.bin uint16 no MESMO formato que o train_gpu.py lê (sequência plana, <|endoftext|> na
fronteira de CADA exemplo).

O EOS na fronteira é o que ensina onde um comando termina e o próximo começa — sem ele o
formato do rótulo não "gruda" no fim (mesma decisão do tokeniza_conversa.py).

Sobe pro R2 em executor_premium/v1/ sob doppler:
  doppler run --project omnigen-ia --config prd -- ./venv/bin/python tokeniza_executor.py
"""
from __future__ import annotations
import json
import os
import sys
from array import array

import numpy as np
from tokenizers import Tokenizer

DIR = os.path.dirname(os.path.abspath(__file__))
FONTE = os.path.join(DIR, "acoes", "executor_treino.jsonl")
TOKJSON = os.path.join(DIR, "data_big", "tokenizer.json")
R2_VARS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
PREFIXO_R2 = "executor_premium/v1"
OUT_DIR = os.environ.get("EXEC_OUT_DIR", os.path.join(DIR, "dados", "executor_bin"))


def diz(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def textos():
    n = 0
    with open(FONTE, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            o = json.loads(linha)
            entrada = (o.get("entrada") or "").rstrip()
            alvo = o.get("alvo") or ""
            texto = (entrada + alvo).strip()
            if texto:
                yield texto
                n += 1
    diz(fonte=os.path.basename(FONTE), exemplos=n)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    tok = Tokenizer.from_file(TOKJSON)
    if tok.get_vocab_size() > 65535:
        raise SystemExit(f"vocab {tok.get_vocab_size()} nao cabe em uint16")
    eot = tok.encode("<|endoftext|>").ids or [0]

    buf = array("H")
    tot_doc = 0
    lote: list[str] = []
    BATCH = 2000

    def flush():
        nonlocal tot_doc
        if not lote:
            return
        for cod in tok.encode_batch(lote):
            buf.extend(cod.ids)
            buf.extend(eot)   # fronteira de exemplo
            tot_doc += 1
        lote.clear()

    for texto in textos():
        lote.append(texto)
        if len(lote) >= BATCH:
            flush()
    flush()

    tot_tok = len(buf)
    if tot_tok == 0:
        raise SystemExit("nenhum token gerado — fonte vazia?")

    nome = "executor-00000.bin"
    cam = os.path.join(OUT_DIR, nome)
    np.frombuffer(buf, dtype=np.uint16).tofile(cam)
    mb = os.path.getsize(cam) / 1e6

    meta = {"tokenizer": "bpe-16384", "exemplos": tot_doc, "tokens": tot_tok,
            "mb": round(mb, 2), "prefixo_r2": PREFIXO_R2,
            "tokens_por_exemplo": round(tot_tok / max(1, tot_doc), 2)}
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    diz(gravou=nome, **meta)

    if os.environ.get("SEM_R2") == "1":
        diz(aviso="SEM_R2=1 — nao subiu pro R2")
        return 0
    faltando = [v for v in R2_VARS if not os.environ.get(v)]
    if faltando:
        diz(ALERTA="sem credenciais R2, nao subiu", faltando=faltando)
        return 0

    import boto3
    from botocore.config import Config
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto", config=Config(signature_version="s3v4", retries={"max_attempts": 5}))
    bucket = os.environ.get("POCKET_LM_BUCKET", "pocket-lm")
    s3.upload_file(cam, bucket, f"{PREFIXO_R2}/{nome}")
    s3.put_object(Bucket=bucket, Key=f"{PREFIXO_R2}/meta.json",
                  Body=json.dumps(meta, ensure_ascii=False, indent=1).encode())
    diz(subiu=f"{PREFIXO_R2}/{nome}", bucket=bucket)
    return 0


if __name__ == "__main__":
    sys.exit(main())
