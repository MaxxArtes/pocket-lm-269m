#!/bin/bash
# "full-conversa-v1": FULL RUN do conversacional do zero (269M, init aleatorio), backbone
# limpo na fase estavel + corpus de conversa QWEN (licenca-limpa, conversa_premium/v2,
# 6,64M tokens unicos) dominando o decay. Escala o pilot (que deu 94,6% in-escopo truncado)
# pra producao.
#
# Config ancorada no HISTORICO (corpus-limpo-v1, o run base de 13h que funcionou: DECAY 0.2)
# + evidencia do pilot (~400 epocas de premium estreito generalizaram a 94,6%, nao decoraram):
#  - PREMIUM_BIN = conversa_premium/v2 (Qwen 14B Apache 2.0, ja com <|endoftext|> por conversa,
#    SEM EOS-duplo). 6x mais dado unico que o pilot (6,64M vs 1,07M).
#  - DECAY_FRAC 0.20: valor do corpus-limpo-v1; agy concorda (literatura 10-20%). Mais backbone
#    na fase estavel = base melhor.
#  - PREMIUM_FRAC 0.35 -> ~73 epocas sobre o premium (BEM abaixo das ~400 do pilot que funcionou;
#    o alarme do agy de "overfit" era exagerado pra conversa). Mantem annealing de tarefa forte.
#  - TRAIN_BUDGET_S 43200 (12h ~ 7B tokens a 161k tok/s do pilot): over-train do backbone.
#  - SFT de polimento (terminador/recusa oos) vem DEPOIS, num job proprio; aqui e pre-treino.
set -uo pipefail
cd /work
L=/work/treino.log
export PYTHONUNBUFFERED=1

python3 - <<'PY' >> $L 2>&1
import boto3, os, json
from botocore.config import Config
e = 'https://' + os.environ['R2_ACCOUNT_ID'] + '.r2.cloudflarestorage.com'
s3 = boto3.client('s3', endpoint_url=e,
    aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
    region_name='auto', config=Config(signature_version='s3v4', retries={'max_attempts': 5}))
B = os.environ.get('POCKET_LM_BUCKET', 'pocket-lm')
for d in ('treino', 'val', 'premium'):
    os.makedirs(f'/work/dados/{d}', exist_ok=True)
os.makedirs('/work/out', exist_ok=True)
p = s3.get_paginator('list_objects_v2')
def baixa(prefixo, destino):
    n = 0
    for pag in p.paginate(Bucket=B, Prefix=prefixo):
        for o in pag.get('Contents', []):
            if not o['Key'].endswith('.bin'):
                continue
            alvo = os.path.join(destino, o['Key'].rsplit('/', 1)[-1])
            if not (os.path.exists(alvo) and os.path.getsize(alvo) == o['Size']):
                s3.download_file(B, o['Key'], alvo)
            n += 1
    print(f'{prefixo}: {n} arquivos', flush=True)
for pref in ('finepdfs/v2', 'fineweb2hq/v2', 'hplt/v2'):
    baixa(pref, '/work/dados/treino')
for pref in ('validacao/finepdfs', 'validacao/fineweb2hq', 'validacao/hplt'):
    baixa(pref, '/work/dados/val')
baixa('conversa_premium/v2', '/work/dados/premium')   # <- corpus limpo Qwen
json.dump({'vocab_size': 16384}, open('/work/dados/meta.json', 'w'))
print('sem ckpt semeado: init aleatorio (from-scratch)', flush=True)
PY
[ $? -ne 0 ] && { echo "PREPARO FALHOU" >> $L; exit 1; }

export DATA_DIR=/work/dados OUT_DIR=/work/out \
  TRAIN_BIN="treino/*.bin" VAL_BIN="val/*.bin" PREMIUM_BIN="premium/*.bin" \
  N_LAYER=20 N_HEAD=16 N_EMBD=1024 BLOCK=512 \
  BATCH=64 GRAD_ACCUM=4 \
  LR=6e-4 OTIM=muon MUON_MULT='{"matriz":13.5,"embed":11.5,"head":3.5,"escalar":12.5}' \
  WARMUP=500 DECAY_FRAC=0.20 PREMIUM_FRAC=0.35 \
  MAX_STEPS=1000000 TRAIN_BUDGET_S="${TRAIN_BUDGET_S:-43200}" \
  EVAL_EVERY=500 CKPT_EVERY=1000 CKPT_R2_EVERY=3000 \
  RUN_NAME=full-conversa-v1 R2_RUN_PREFIX=runs/full-conversa-v1 \
  GPU_TIPO=H100

NPROC=$(nvidia-smi -L | wc -l)
[ "$NPROC" -lt 1 ] && NPROC=1
echo "== FULL pre-treino from-scratch, $NPROC GPU(s), budget ${TRAIN_BUDGET_S:-43200}s ==" >> $L
if [ "$NPROC" -gt 1 ]; then
  torchrun --nproc_per_node "$NPROC" --master_port 29542 train_gpu.py >> $L 2>&1
else
  python3 train_gpu.py >> $L 2>&1
fi
cod=$?
echo "treino saiu com codigo $cod" >> $L

python3 - <<'PY' >> $L 2>&1
import boto3, os, json, time
from botocore.config import Config
e = 'https://' + os.environ['R2_ACCOUNT_ID'] + '.r2.cloudflarestorage.com'
s3 = boto3.client('s3', endpoint_url=e,
    aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
    region_name='auto', config=Config(signature_version='s3v4'))
B = os.environ.get('POCKET_LM_BUCKET', 'pocket-lm')
pref = 'runs/full-conversa-v1'
try:
    if os.path.exists('/work/treino.log'):
        s3.upload_file('/work/treino.log', B, f'{pref}/log.txt'); print('subiu log', flush=True)
except Exception as _e:
    print('falha log:', _e, flush=True)
for nome in ('ckpt.pt', 'report.json', 'progress.json'):
    cam = f'/work/out/{nome}'
    if os.path.exists(cam):
        s3.upload_file(cam, B, f'{pref}/{nome}'); print('subiu', nome, flush=True)
s3.put_object(Bucket=B, Key=f'{pref}/done.json',
              Body=json.dumps({'run': 'full-conversa-v1', 'fim_utc': time.time()}).encode())
print('subiu done.json', flush=True)
PY
echo "== FIM JOB ==" >> $L
