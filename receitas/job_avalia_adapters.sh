#!/usr/bin/env bash
# Completa a bateria dos adapters de produto: CALAME + LAMBADA (quebraram por falta de
# pyarrow no job_adapter_produto.sh) e as PREVISOES item a item no MASSIVE test para o
# McNemar contra o v8 (regra permanente: McNemar e o juiz de toda troca de versao).
#   RUN_NAME=avalia-adapters bash job_avalia_adapters.sh
set -uo pipefail
cd "$(dirname "$0")"
pip install -q --no-input tokenizers numpy boto3 pyarrow huggingface_hub 2>&1 | tail -1 || true

python3 - <<'PY'
import boto3, os
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto", config=Config(retries={"max_attempts": 3}))
B = os.environ.get("POCKET_LM_BUCKET", "pocket-lm")
os.makedirs("data_big", exist_ok=True); os.makedirs("acoes", exist_ok=True); os.makedirs("avaliacao", exist_ok=True)
s3.download_file(B, "aux/tokenizer.json", "data_big/tokenizer.json")
s3.download_file(B, "acoes/massive_br_test.jsonl", "acoes/massive_br_test.jsonl")
for m in ("adapter-app-v1", "adapter-whats-v1"):
    os.makedirs(f"gpu_out/{m}", exist_ok=True)
    s3.download_file(B, f"derivados/269m/{m}/modelo.pt", f"gpu_out/{m}/slim.pt")
    print("baixou", m, flush=True)
PY

TOK=data_big/tokenizer.json
sobe () { python3 - "$1" <<'PYU' 2>/dev/null || true
import boto3, os, sys
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto", config=Config())
s3.upload_file(sys.argv[1], os.environ.get("POCKET_LM_BUCKET", "pocket-lm"),
               f"avaliacao/{os.path.basename(sys.argv[1])}")
print("subiu", sys.argv[1], flush=True)
PYU
}

for M in adapter-app-v1 adapter-whats-v1; do
  CK="gpu_out/${M}/slim.pt"
  echo "########## ${M} ##########"
  python3 eval_calame_acc.py "${CK}:${TOK}" --ban-nl 2>&1 | tail -3
  cp avaliacao/calame_acc.json "avaliacao/P_${M}_calame.json" 2>/dev/null || true
  sobe "avaliacao/P_${M}_calame.json"
  python3 eval_lambada.py "${CK}:${TOK}" 2>&1 | tail -2
  cp avaliacao/lambada.json "avaliacao/P_${M}_lambada.json" 2>/dev/null || true
  sobe "avaliacao/P_${M}_lambada.json"
  # previsoes item a item no MESMO conjunto e MESMA ordem (pular=0, limite=1035) que o
  # conjunto canonico do v8 — e o pareamento que o McNemar exige
  python3 avalia_acoes.py --ckpt "$CK" --tokenizer "$TOK" --teste acoes/massive_br_test.jsonl \
    --limite 1035 --saida "avaliacao/P_${M}_massive1035.json" 2>&1 | tail -3
  sobe "avaliacao/P_${M}_massive1035.json"
done

python3 - <<'PY'
import boto3, json, os
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto", config=Config(retries={"max_attempts": 3}))
s3.put_object(Bucket=os.environ.get("POCKET_LM_BUCKET", "pocket-lm"),
              Key=f"runs/{os.environ.get('RUN_NAME','avalia-adapters')}/report.json",
              Body=json.dumps({"ok": True}).encode())
print("FIM declarado", flush=True)
PY
