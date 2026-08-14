#!/usr/bin/env bash
# COBERTURA 100%: uma base + suas 3 variantes de SFT, todas medidas em TODOS os testes.
#
#   BASE=corpus-limpo-v1 bash job_completo.sh
#
# Produz 4 modelos e roda 7 testes em cada:
#   modelos:  base | +SFT acoes | +SFT conversa | +SFT multitarefa
#   testes:   CALAME, CALAME sem \n, LAMBADA, acoes, F1 com parada, portoes, 5-shot
#             (+ gera o arquivo do juiz cego, que roda depois na bancada por usar API paga)
#
# TODOS os modelos sao salvos em R2. Ate 13/08 o train_acoes.py nao salvava, e por isso
# 9 dos 12 modelos de hoje viraram so numero — nao dava para medir mais nada neles.
set -uo pipefail
cd "$(dirname "$0")"
BASE="${BASE:?defina BASE}"
pip install -q --no-input tokenizers numpy boto3 pyarrow huggingface_hub 2>&1 | tail -1 || true

python3 - <<PY
import boto3, json, os, random, torch
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto", config=Config(retries={"max_attempts": 3}))
B = os.environ.get("POCKET_LM_BUCKET", "pocket-lm"); base = "${BASE}"
for d in ("data_big", "acoes", "dados/conversa", "avaliacao", f"gpu_out/{base}"):
    os.makedirs(d, exist_ok=True)
for k, dest in (("aux/tokenizer.json", "data_big/tokenizer.json"),
                ("acoes/sft_acoes.jsonl", "acoes/sft_acoes.jsonl"),
                ("acoes/massive_br_test.jsonl", "acoes/massive_br_test.jsonl"),
                ("acoes/massive_br_validation.jsonl", "acoes/massive_br_validation.jsonl"),
                ("dados/conversa/sft_treino.jsonl", "dados/conversa/sft_treino.jsonl")):
    s3.download_file(B, k, dest)
s3.download_file(B, f"runs/{base}/ckpt.pt", f"gpu_out/{base}/ckpt.pt")
d = torch.load(f"gpu_out/{base}/ckpt.pt", map_location="cpu", weights_only=False)
torch.save({k2: v for k2, v in d.items() if k2 in ("model","cfg","step")}, f"gpu_out/{base}/slim.pt")
os.remove(f"gpu_out/{base}/ckpt.pt")
# mix multitarefa equilibrado (cortar o maior, nao repetir o menor)
conv = [json.loads(l) for l in open("dados/conversa/sft_treino.jsonl", encoding="utf-8")]
acao = [json.loads(l) for l in open("acoes/sft_acoes.jsonl", encoding="utf-8")]
n = min(len(conv), len(acao)); random.seed(7)
mix = random.sample(conv, n) + random.sample(acao, n); random.shuffle(mix)
with open("dados/sft_generico.jsonl", "w", encoding="utf-8") as f:
    for r in mix: f.write(json.dumps({"entrada": r["entrada"], "alvo": r["alvo"]}, ensure_ascii=False) + "\n")
print(f"base {base} pronta | mix {len(mix)}", flush=True)
PY

TOK=data_big/tokenizer.json
treina () {   # $1=sufixo  $2=dados
  echo "===== SFT ${1} sobre ${BASE} ====="
  python3 train_acoes.py --ckpt "gpu_out/${BASE}/slim.pt" --dados "$2" --tokenizer "$TOK" \
    --epocas 2 --batch 32 --max-len 256 --lr 1e-4 --saida "gpu_out/${BASE}-${1}" 2>&1 | tail -6
}
treina acoes  acoes/sft_acoes.jsonl
treina conversa dados/conversa/sft_treino.jsonl
treina multi  dados/sft_generico.jsonl

# 7 testes em cada um dos 4 modelos
for M in "${BASE}" "${BASE}-acoes" "${BASE}-conversa" "${BASE}-multi"; do
  CK="gpu_out/${M}/slim.pt"
  [ -f "$CK" ] || CK="gpu_out/${BASE}/slim.pt"
  echo "########## ${M} ##########"
  python3 eval_calame_acc.py "${CK}:${TOK}" --ban-nl 2>&1 | tail -3
  cp avaliacao/calame_acc.json "avaliacao/T_${M}_calame.json" 2>/dev/null || true
  python3 eval_lambada.py "${CK}:${TOK}" 2>&1 | tail -2
  cp avaliacao/lambada.json "avaliacao/T_${M}_lambada.json" 2>/dev/null || true
  python3 avalia_acoes.py --ckpt "$CK" --tokenizer "$TOK" --teste acoes/massive_br_test.jsonl \
    --limite 500 --saida "avaliacao/T_${M}_acoes.json" 2>&1 | tail -3
  python3 avalia_acoes_parada.py --ckpt "$CK" --tokenizer "$TOK" --limite 200 \
    --saida "avaliacao/T_${M}_parada.json" 2>&1 | tail -2
  python3 avalia_conversa.py --ckpt "$CK" --prompts-juiz "avaliacao/T_${M}_juiz.jsonl" \
    --saida "avaliacao/T_${M}_portoes.json" 2>&1 | tail -3
  python3 massive_fewshot.py --ckpt "$CK" --tokenizer "$TOK" --n 300 --shots 5 \
    --saida "avaliacao/T_${M}_fewshot.json" 2>&1 | tail -3
done

python3 - <<PY
import boto3, json, os, glob
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto", config=Config(retries={"max_attempts": 3}))
B = os.environ.get("POCKET_LM_BUCKET", "pocket-lm"); base = "${BASE}"
for f in glob.glob("avaliacao/T_*"):
    s3.upload_file(f, B, f"avaliacao/{os.path.basename(f)}")
# SALVA OS MODELOS (a lacuna que custou 9 modelos hoje)
for suf in ("acoes", "conversa", "multi"):
    p = f"gpu_out/{base}-{suf}/slim.pt"
    if os.path.exists(p):
        s3.upload_file(p, B, f"derivados/269m/{base}-{suf}/modelo.pt")
        print("modelo salvo:", f"derivados/269m/{base}-{suf}", flush=True)
s3.put_object(Bucket=B, Key=f"runs/{os.environ.get('RUN_NAME','completo')}/report.json",
              Body=json.dumps({"ok": True, "base": base}).encode())
print("FIM declarado", flush=True)
PY
