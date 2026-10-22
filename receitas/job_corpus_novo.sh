#!/bin/bash
# Rodada "corpus-limpo-v1": continua o 269M (pesos do ckpt_pre_decay do ext3) sobre o
# corpus de licenca limpa (12,31 bi tokens), em 4 GPUs com DDP.
#
# Decisoes e por que (analise completa em docs/plano-treino-corpus-novo.md):
#  - pesos do PRE-decay: o decaimento final colapsa o LR; retomar do decaido perde
#    plasticidade. So os PESOS viajam — otimizador e passo comecam do zero, porque o
#    momento acumulado apontava para a distribuicao antiga (76% trocada).
#  - LR 1e-4 com warmup 500: veredito do agy (6e-5 era LR de fine-tuning, nao de
#    adaptacao a distribuicao nova), moderado pela nossa medicao de expoente 0,16.
#  - lote global 524.288 tok (64 x 4 acum x 512 x 4 GPUs): 4x o antigo; medido que
#    32k deixava a H100 em MFU 18,8%.
#  - premium = finepdfs na fase de decay (PREMIUM_FRAC 0.5): a unica fonte de PDF
#    (o card cita sobretudo dominios juridicos e educacionais; os PDFs tambem vem
#    do Common Crawl)
#    como dado curado do fim, que e a alavanca de maior ganho medido no nosso porte.
#    NAO sobre-amostrado no resto do treino (veto do agy: 269M decora dialeto de PDF).
#  - val = shards reservados de validacao (movidos para fora do treino em 04/08).
#  - SCHEDULE_POR_PASSO=1: o WSD e por PASSO, nao por relogio. Sem isso, se o treino
#    cumprir os 23.480 passos antes do orcamento de tempo, o decaimento nunca roda —
#    o erro que ja aconteceu no pocket-h200-v2 (LR em 99,8% do pico no fim).
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
os.makedirs('/work/dados/treino', exist_ok=True)
os.makedirs('/work/dados/val', exist_ok=True)
os.makedirs('/work/dados/premium', exist_ok=True)
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
# premium = so o finepdfs, por hardlink (zero disco extra)
for f in os.listdir('/work/dados/treino'):
    if f.startswith('finepdfs'):
        alvo = f'/work/dados/premium/{f}'
        os.path.exists(alvo) or os.link(f'/work/dados/treino/{f}', alvo)
json.dump({'vocab_size': 16384}, open('/work/dados/meta.json', 'w'))

# pesos do pre-decay, SEM otimizador nem passo: chega como "otimizador mudou, estado
# descartado" no load do train_gpu, que e exatamente a intencao
s3.download_file(B, 'runs/pocket-269m-ext3/ckpt_pre_decay.pt', '/work/ck_full.pt')
import torch
ck = torch.load('/work/ck_full.pt', map_location='cpu', weights_only=False)
torch.save({'model': ck['model'], 'cfg': ck['cfg']}, '/work/out/ckpt.pt')
os.remove('/work/ck_full.pt')
print('checkpoint preparado: so pesos, passo 0', flush=True)
PY
[ $? -ne 0 ] && { echo "PREPARO FALHOU" >> $L; exit 1; }

export DATA_DIR=/work/dados OUT_DIR=/work/out \
  TRAIN_BIN="treino/*.bin" VAL_BIN="val/*.bin" PREMIUM_BIN="premium/*.bin" \
  N_LAYER=20 N_HEAD=16 N_EMBD=1024 BLOCK=512 \
  BATCH=64 GRAD_ACCUM=4 \
  LR=1e-4 OTIM=muon MUON_MULT='{"matriz":13.5,"embed":11.5,"head":3.5,"escalar":12.5}' \
  WARMUP=500 DECAY_FRAC=0.2 MAX_STEPS=23480 TRAIN_BUDGET_S=46800 \
  SCHEDULE_POR_PASSO=1 \
  EVAL_EVERY=250 CKPT_EVERY=500 CKPT_R2_EVERY=2000 \
  RUN_NAME=corpus-limpo-v1 R2_RUN_PREFIX=runs/corpus-limpo-v1 \
  GPU_TIPO=H100

echo "== treino DDP 4 GPUs ==" >> $L
torchrun --nproc_per_node 4 --master_port 29541 train_gpu.py >> $L 2>&1
cod=$?
echo "treino saiu com codigo $cod" >> $L

# sobe o resultado final independente do codigo de saida
python3 - <<'PY' >> $L 2>&1
import boto3, os, json
from botocore.config import Config
e = 'https://' + os.environ['R2_ACCOUNT_ID'] + '.r2.cloudflarestorage.com'
s3 = boto3.client('s3', endpoint_url=e,
    aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
    region_name='auto', config=Config(signature_version='s3v4'))
B = os.environ.get('POCKET_LM_BUCKET', 'pocket-lm')
for nome in ('ckpt.pt', 'report.json', 'progress.json'):
    cam = f'/work/out/{nome}'
    if os.path.exists(cam):
        s3.upload_file(cam, B, f'runs/corpus-limpo-v1/{nome}')
        print('subiu', nome, flush=True)
PY
echo "== FIM JOB ==" >> $L
