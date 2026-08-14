#!/usr/bin/env bash
# Fecha a bateria dos adapters de produto: SO o que falta — massive_fewshot (n=300,
# 5 shots) por modelo. O resto (CALAME, LAMBADA, massive1035, acoes n=500, parada
# n=200, portoes+juiz) ja esta no R2 em avaliacao/P_* (runs de 14/08 madrugada/manha).
#   RUN_NAME=bateria-adapters bash job_bateria_adapters.sh
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
os.makedirs("data_big", exist_ok=True); os.makedirs("avaliacao", exist_ok=True)
s3.download_file(B, "aux/tokenizer.json", "data_big/tokenizer.json")
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
  python3 massive_fewshot.py --ckpt "$CK" --tokenizer "$TOK" --n 300 --shots 5 \
    --saida "avaliacao/P_${M}_fewshot.json" 2>&1 | tail -3
  sobe "avaliacao/P_${M}_fewshot.json"
done

python3 - <<'PY'
import boto3, json, os
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto", config=Config(retries={"max_attempts": 3}))
s3.put_object(Bucket=os.environ.get("POCKET_LM_BUCKET", "pocket-lm"),
              Key=f"runs/{os.environ.get('RUN_NAME','bateria-adapters')}/report.json",
              Body=json.dumps({"ok": True}).encode())
print("FIM declarado", flush=True)
PY
