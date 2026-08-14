"""Tokeniza o corpus de conversa telegrafica em .bin uint16 pro pre-treino from-scratch.

Junta corpus_shard_*.jsonl + sft_treino.jsonl + dialogos.jsonl de dados/conversa/,
tokeniza com data_big/tokenizer.json (vocab 16384) e grava .bin uint16 no MESMO formato
que o train_gpu.py le (sequencia plana, <|endoftext|> na fronteira de cada CONVERSA).

Cada conversa e um documento: sem o EOS na fronteira o modelo nao ve onde um dialogo
termina e o proximo comeca (mesma decisao do prepara_fonte_limpa.py).

Sobe pro R2 em conversa_premium/v1/ sob doppler:
  doppler run --project omnigen-ia --config prd -- ./venv/bin/python tokeniza_conversa.py
"""
from __future__ import annotations
import glob
import json
import os
import sys
from array import array

import numpy as np
from tokenizers import Tokenizer

DIR = os.path.dirname(os.path.abspath(__file__))
CONV = os.path.join(DIR, "dados", "conversa")
TOKJSON = os.path.join(DIR, "data_big", "tokenizer.json")
R2_VARS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
# v1 = corpus antigo (deepseek, licenca suja). v2 = corpus Qwen (Apache 2.0, licenca limpa).
PREFIXO_R2 = os.environ.get("CONV_PREFIXO_R2", "conversa_premium/v1")
OUT_DIR = os.environ.get("CONV_OUT_DIR", os.path.join(DIR, "dados", "conversa_bin"))
TERM = "<|endoftext|>"
# CONV_SO_QWEN=1 tokeniza SO o qwen_conversa.jsonl (licenca limpa), ignorando as fontes
# antigas de licenca suja. O flush() poe o <|endoftext|> na fronteira; o alvo do Qwen ja
# traz o marcador literal no fim -> tira aqui pra nao duplicar (EOS-duplo).
SO_QWEN = os.environ.get("CONV_SO_QWEN") == "1"


def diz(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def textos():
    """Gera o texto de cada conversa, uma por vez, das tres fontes."""
    # 1) corpus_shard_*.jsonl e sft_treino.jsonl: campos entrada + alvo.
    #    "entrada" ja vem como "usuário: ...\nassistente:" e "alvo" comeca com " ..."
    qwen = os.path.join(CONV, "qwen_conversa.jsonl")
    if SO_QWEN:
        fontes_ea = [qwen]
    else:
        fontes_ea = sorted(glob.glob(os.path.join(CONV, "corpus_shard_*.jsonl")))
        fontes_ea.append(os.path.join(CONV, "sft_treino.jsonl"))
        fontes_ea.append(qwen)
    for cam in fontes_ea:
        if not os.path.exists(cam):
            diz(aviso="fonte ausente", arquivo=os.path.basename(cam))
            continue
        n = 0
        with open(cam, encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if not linha:
                    continue
                o = json.loads(linha)
                entrada = (o.get("entrada") or "").rstrip()
                alvo = o.get("alvo") or ""
                texto = (entrada + alvo).strip()
                # tira o marcador literal do fim: o flush() poe o EOS na fronteira
                if texto.endswith(TERM):
                    texto = texto[: -len(TERM)].rstrip()
                if texto:
                    yield texto
                    n += 1
        diz(fonte=os.path.basename(cam), conversas=n)

    # 2) dialogos.jsonl: multi-turno em "turnos"=[{u,a}, ...] -> mesmo formato textual
    cam = os.path.join(CONV, "dialogos.jsonl")
    if SO_QWEN:
        pass  # licenca limpa: so o corpus Qwen
    elif os.path.exists(cam):
        n = 0
        with open(cam, encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if not linha:
                    continue
                o = json.loads(linha)
                partes = []
                for t in o.get("turnos", []):
                    u = (t.get("u") or "").strip()
                    a = (t.get("a") or "").strip()
                    if not (u or a):
                        continue
                    partes.append(f"usuário: {u}\nassistente: {a}")
                texto = "\n".join(partes).strip()
                if texto:
                    yield texto
                    n += 1
        diz(fonte="dialogos.jsonl", conversas=n)
    else:
        diz(aviso="fonte ausente", arquivo="dialogos.jsonl")


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
            buf.extend(eot)   # fronteira de conversa
            tot_doc += 1
        lote.clear()

    for texto in textos():
        lote.append(texto)
        if len(lote) >= BATCH:
            flush()
    flush()

    tot_tok = len(buf)
    if tot_tok == 0:
        raise SystemExit("nenhum token gerado — fontes vazias?")

    # um shard unico: o volume e pequeno (dezenas de MB). Nome casa com o glob
    # premium/*.bin usado pelo job (conversa_premium/v1/*.bin no R2).
    nome = "conversa-00000.bin"
    cam = os.path.join(OUT_DIR, nome)
    np.frombuffer(buf, dtype=np.uint16).tofile(cam)
    mb = os.path.getsize(cam) / 1e6

    meta = {"tokenizer": "bpe-16384", "conversas": tot_doc, "tokens": tot_tok,
            "mb": round(mb, 2), "prefixo_r2": PREFIXO_R2,
            "bytes_por_token": round(os.path.getsize(cam) / max(1, tot_tok), 3)}
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
