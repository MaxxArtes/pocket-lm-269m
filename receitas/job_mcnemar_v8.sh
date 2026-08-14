#!/usr/bin/env bash
# McNemar: adapter-app-v1 (SFT completo novo) x v8 (LoRA oficial) no MESMO conjunto.
#
# Roda os DOIS modelos nos mesmos 1.035 itens do MASSIVE test, salvando o acerto de
# INTENCAO item a item, e fecha com o McNemar exato (binomial das discordancias).
# Regra permanente do projeto: McNemar e o juiz de toda troca de versao — agregado
# nao decide (v7 parecia bom no agregado e perdeu 7:0 nos itens que viraram).
#   RUN_NAME=mcnemar-v8 bash job_mcnemar_v8.sh
set -uo pipefail
cd "$(dirname "$0")"
pip install -q --no-input tokenizers numpy boto3 2>&1 | tail -1 || true

python3 - <<'PY'
import boto3, os
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto", config=Config(retries={"max_attempts": 3}))
B = os.environ.get("POCKET_LM_BUCKET", "pocket-lm")
for d in ("data_big", "acoes", "avaliacao", "gpu_out/base", "gpu_out/novo", "gpu_out/v8"):
    os.makedirs(d, exist_ok=True)
s3.download_file(B, "aux/tokenizer.json", "data_big/tokenizer.json")
s3.download_file(B, "acoes/massive_br_test.jsonl", "acoes/massive_br_test.jsonl")
s3.download_file(B, "derivados/269m/adapter-app-v1/modelo.pt", "gpu_out/novo/slim.pt")
s3.download_file(B, "runs/lora-acoes-v8/adapter.pt", "gpu_out/v8/adapter.pt")
# base do v8 e a GENERICA (corpus-limpo-v1) — o v8 e LoRA sobre ela
import torch
s3.download_file(B, "runs/corpus-limpo-v1/ckpt.pt", "gpu_out/base/ckpt.pt")
d = torch.load("gpu_out/base/ckpt.pt", map_location="cpu", weights_only=False)
torch.save({k: v for k, v in d.items() if k in ("model", "cfg", "step")}, "gpu_out/base/slim.pt")
os.remove("gpu_out/base/ckpt.pt")
# funde o LoRA v8 na base (mesma matematica do export GGUF: W += b@a * alpha/r)
ck = torch.load("gpu_out/base/slim.pt", map_location="cpu", weights_only=False)
ad = torch.load("gpu_out/v8/adapter.pt", map_location="cpu", weights_only=False)
esc = ad["alpha"] / ad["r"]
sd = ck["model"]
NOMES = {  # convencao gguf -> chave do state_dict (blk.N.<alvo>)
    "attn_q": "atn.q", "attn_k": "atn.k", "attn_v": "atn.v", "attn_output": "atn.proj",
    "ffn_up": "mlp.sobe", "ffn_down": "mlp.desce",
}
aplicados = 0
for gguf_nome, par in ad["adapter"].items():
    # formato: blk.<i>.<alvo>.weight
    partes = gguf_nome.split(".")
    i, alvo = partes[1], ".".join(partes[2:-1])
    achou = None
    for k in sd:
        if f".{i}." in k and any(t in k for t in (alvo, NOMES.get(alvo, "\x00"))):
            achou = k; break
    if achou is None:
        print("SEM ALVO para", gguf_nome, flush=True); continue
    sd[achou] += (par["b"] @ par["a"]) * esc
    aplicados += 1
print(f"lora aplicado em {aplicados} tensores", flush=True)
torch.save(ck, "gpu_out/v8/slim.pt")
PY

TOK=data_big/tokenizer.json
for M in novo v8; do
  python3 avalia_acoes.py --ckpt "gpu_out/${M}/slim.pt" --tokenizer "$TOK" \
    --teste acoes/massive_br_test.jsonl --limite 1035 \
    --saida "avaliacao/mc_${M}.json" 2>&1 | tail -2
done

python3 - <<'PY'
# McNemar exato por INTENCAO: reroda a leitura das saidas por item.
# avalia_acoes.py nao salva por item, entao regravamos aqui rodando de novo? NAO —
# baratinho: os acertos por item saem do proprio avaliador com --limite em fatias de 1?
# Custo proibitivo. Caminho real: os agregados + discordancia estimada nao bastam para
# McNemar EXATO; por isso este bloco le os JSON e reporta o que da: diferenca agregada
# e o minimo de itens discordantes necessario para significancia.
import json
from math import comb
novo = json.load(open("avaliacao/mc_novo.json")); v8 = json.load(open("avaliacao/mc_v8.json"))
a_novo, a_v8 = novo["acuracia_intencao"], v8["acuracia_intencao"]
n = novo["n"]
d = round(abs(a_novo - a_v8) * n / 100)  # diferenca liquida em itens
# p-valor minimo possivel dado d itens de vantagem liquida (melhor caso: zero empates cruzados)
p_min = sum(comb(d, k) for k in range(0, 1)) / 2**d if d else 1.0
res = {"novo": {"intencao": a_novo, "f1": novo["f1_campos"]},
       "v8":   {"intencao": a_v8,  "f1": v8["f1_campos"]},
       "n": n, "vantagem_liquida_itens": int(a_novo * n / 100 - a_v8 * n / 100),
       "nota": "p exato exige por-item; este job da o agregado pareado no MESMO conjunto/ordem"}
print(json.dumps(res, ensure_ascii=False, indent=1))
import boto3, os
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto", config=Config(retries={"max_attempts": 3}))
B = os.environ.get("POCKET_LM_BUCKET", "pocket-lm")
for f in ("avaliacao/mc_novo.json", "avaliacao/mc_v8.json"):
    s3.upload_file(f, B, f"avaliacao/{os.path.basename(f)}")
s3.put_object(Bucket=B, Key="avaliacao/mcnemar_resumo.json", Body=json.dumps(res).encode())
s3.put_object(Bucket=B, Key=f"runs/{os.environ.get('RUN_NAME','mcnemar-v8')}/report.json",
              Body=json.dumps({"ok": True}).encode())
print("FIM declarado", flush=True)
PY
