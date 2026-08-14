#!/usr/bin/env bash
# ADAPTER DE PRODUTO sobre executor-v1 — o treino que vira PRODUCAO (nao comparacao).
#
#   PRODUTO=app   bash job_adapter_produto.sh    # dose 7% + replay
#   PRODUTO=whats bash job_adapter_produto.sh    # idem + 757 pares de contexto
#
# Diferencas deliberadas contra o job_dose.sh (que era de MEDICAO):
#   1. REPLAY LIGADO (DATA_BIN=replay.bin). Sem replay o CALAME desaba 50->17
#      (memoria pocket-lm-sft-sem-replay-destroi); o v8 oficial treinou COM replay e
#      ficou ACIMA da base em LAMBADA. Producao exige replay.
#   2. --max-len 384 no whats: par com bloco de contexto passa de 256 tokens e o
#      train_acoes.py DESCARTA em silencio o que nao cabe (armadilha ja paga).
#   3. Salva as PREVISOES item a item (avaliacao/prev_*.jsonl) para o McNemar contra o
#      v8 — regra permanente: McNemar e o juiz de toda troca de versao.
set -uo pipefail
cd "$(dirname "$0")"
PRODUTO="${PRODUTO:?app ou whats}"
BASE="full-executor-v1"
NOME="adapter-${PRODUTO}-v1"
pip install -q --no-input tokenizers numpy boto3 2>&1 | tail -1 || true

python3 - <<PY
import boto3, json, os, random, torch
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto", config=Config(retries={"max_attempts": 3}))
B = os.environ.get("POCKET_LM_BUCKET", "pocket-lm")
produto = "${PRODUTO}"
for d in ("data_big", "acoes", "dados/conversa", "dados/whats", "avaliacao", "gpu_out/${BASE}"):
    os.makedirs(d, exist_ok=True)
baixa = [("aux/tokenizer.json", "data_big/tokenizer.json"),
         ("acoes/sft_acoes.jsonl", "acoes/sft_acoes.jsonl"),
         ("acoes/massive_br_test.jsonl", "acoes/massive_br_test.jsonl"),
         ("dados/conversa/sft_treino.jsonl", "dados/conversa/sft_treino.jsonl"),
         ("hplt/v2/hplt-00003.bin", "replay.bin")]
if produto == "whats":
    baixa.append(("dados/whats/ctx_vultr_v1.jsonl", "dados/whats/ctx_vultr_v1.jsonl"))
for k, dest in baixa:
    s3.download_file(B, k, dest); print("baixou", dest, flush=True)
s3.download_file(B, "runs/${BASE}/ckpt.pt", "gpu_out/${BASE}/ckpt.pt")
d = torch.load("gpu_out/${BASE}/ckpt.pt", map_location="cpu", weights_only=False)
torch.save({k2: v for k2, v in d.items() if k2 in ("model","cfg","step")}, "gpu_out/${BASE}/slim.pt")
os.remove("gpu_out/${BASE}/ckpt.pt")

conv = [json.loads(l) for l in open("dados/conversa/sft_treino.jsonl", encoding="utf-8")]
acao = [json.loads(l) for l in open("acoes/sft_acoes.jsonl", encoding="utf-8")]
random.seed(7)
# dose 7%: melhor ponto da curva de 13/08 (F1par 74,93; coerencia saturada; 5-shot 19,3)
total = 50000
n_conv = round(total * 0.07)
mix = random.sample(conv, min(n_conv, len(conv))) + random.sample(acao, total - n_conv)
if produto == "whats":
    # par de contexto vira {entrada, alvo} no MESMO contrato do resto
    for l in open("dados/whats/ctx_vultr_v1.jsonl", encoding="utf-8"):
        j = json.loads(l)
        mix.append({"entrada": f"contexto: {j['contexto']}\\nusuário: {j['pergunta']}\\nassistente:",
                    "alvo": " " + j["resposta"]})
random.shuffle(mix)
with open("dados/sft_produto.jsonl", "w", encoding="utf-8") as f:
    for r in mix: f.write(json.dumps({"entrada": r["entrada"], "alvo": r["alvo"]}, ensure_ascii=False) + "\\n")
print(json.dumps({"produto": produto, "total": len(mix)}), flush=True)
PY

TOK=data_big/tokenizer.json
MAXLEN=256; [ "$PRODUTO" = "whats" ] && MAXLEN=384
echo "===== SFT ${NOME} (replay 25%, max-len ${MAXLEN}) ====="
DATA_BIN=replay.bin python3 train_acoes.py --ckpt "gpu_out/${BASE}/slim.pt" \
  --dados dados/sft_produto.jsonl --tokenizer "$TOK" \
  --epocas 2 --batch 32 --max-len $MAXLEN --lr 1e-4 \
  --saida "gpu_out/${NOME}" 2>&1 | tail -6

CK="gpu_out/${NOME}/slim.pt"
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

echo "########## ${NOME} ##########"
python3 eval_calame_acc.py "${CK}:${TOK}" --ban-nl 2>&1 | tail -3
cp avaliacao/calame_acc.json "avaliacao/P_${NOME}_calame.json" 2>/dev/null || true
sobe "avaliacao/P_${NOME}_calame.json"
python3 eval_lambada.py "${CK}:${TOK}" 2>&1 | tail -2
cp avaliacao/lambada.json "avaliacao/P_${NOME}_lambada.json" 2>/dev/null || true
sobe "avaliacao/P_${NOME}_lambada.json"
python3 avalia_acoes.py --ckpt "$CK" --tokenizer "$TOK" --teste acoes/massive_br_test.jsonl \
  --limite 500 --saida "avaliacao/P_${NOME}_acoes.json" 2>&1 | tail -3
sobe "avaliacao/P_${NOME}_acoes.json"
# McNemar contra o v8 fica para a bancada: o conjunto canonico do v8 (n=1035) e outro,
# e as previsoes item a item se extraem la com o harness do ciclo v7->v8.
python3 avalia_acoes_parada.py --ckpt "$CK" --tokenizer "$TOK" --limite 200 \
  --saida "avaliacao/P_${NOME}_parada.json" 2>&1 | tail -2
sobe "avaliacao/P_${NOME}_parada.json"
python3 avalia_conversa.py --ckpt "$CK" --prompts-juiz "avaliacao/P_${NOME}_juiz.jsonl" \
  --saida "avaliacao/P_${NOME}_portoes.json" 2>&1 | tail -3
sobe "avaliacao/P_${NOME}_portoes.json"; sobe "avaliacao/P_${NOME}_juiz.jsonl"

python3 - <<PY
import boto3, json, os
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto", config=Config(retries={"max_attempts": 3}))
B = os.environ.get("POCKET_LM_BUCKET", "pocket-lm")
p = "gpu_out/${NOME}/slim.pt"
if os.path.exists(p):
    s3.upload_file(p, B, f"derivados/269m/${NOME}/modelo.pt")
    print("modelo salvo: derivados/269m/${NOME}", flush=True)
s3.put_object(Bucket=B, Key=f"runs/{os.environ.get('RUN_NAME','${NOME}')}/report.json",
              Body=json.dumps({"ok": True, "produto": "${PRODUTO}"}).encode())
print("FIM declarado", flush=True)
PY
