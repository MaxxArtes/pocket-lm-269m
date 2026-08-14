#!/usr/bin/env bash
# CURVA DE DOSE: quanta CONVERSA no SFT multitarefa, com o TOTAL FIXO.
#
#   DOSE=33 BASE=full-executor-v1 bash job_dose.sh
#
# Uma instancia por dose; todas com o mesmo tamanho de treino, entao terminam juntas.
#
# POR QUE TOTAL FIXO (50.000): no `job_completo.sh` o mix era `n = min(conversa, acoes)`,
# ou seja 3.317 + 3.317 — jogava fora 43.441 dos 46.758 exemplos de acao (93%). Os 11 pp de
# intencao que o multitarefa "perdia" para o SFT dedicado estavam confundidos entre PROPORCAO
# e VOLUME. Com o total fixo sobra so a proporcao, que e a pergunta.
#
# POR QUE CALAME/LAMBADA FICAM: na tabela de 13/08 o mix quase DOBROU o CALAME nas tres bases
# (+16,6 / +18,4 / +22,2 pp sobre o SFT dedicado) — dado de conversa e lingua natural e age
# como replay parcial. Como todas as doses tem a mesma receita, a coluna compara direito
# entre elas. Cheguei a propor cortar os dois por serem lentos; teria jogado fora o eixo mais
# interessante do experimento.
set -uo pipefail
cd "$(dirname "$0")"
BASE="${BASE:-full-executor-v1}"
DOSE="${DOSE:?defina DOSE (0,7,18,33,50 = % de conversa no mix)}"
TOTAL="${TOTAL:-50000}"
NOME="dose${DOSE}-${BASE}"
pip install -q --no-input tokenizers numpy boto3 pyarrow huggingface_hub 2>&1 | tail -1 || true

python3 - <<PY
import boto3, json, os, random, torch
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto", config=Config(retries={"max_attempts": 3}))
B = os.environ.get("POCKET_LM_BUCKET", "pocket-lm")
base, dose, total = "${BASE}", int("${DOSE}"), int("${TOTAL}")
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

conv = [json.loads(l) for l in open("dados/conversa/sft_treino.jsonl", encoding="utf-8")]
acao = [json.loads(l) for l in open("acoes/sft_acoes.jsonl", encoding="utf-8")]
n_conv = round(total * dose / 100)
n_acao = total - n_conv
random.seed(7)
# conversa e o conjunto pequeno (3.317): acima de ~1x precisa repetir. Ate ~7,5x, dentro do
# que a literatura considera seguro para SFT (o desastre das 148 epocas foi no PRE-TREINO).
def tira(pool, n):
    if n <= len(pool): return random.sample(pool, n)
    saida = []
    while len(saida) < n:
        saida += random.sample(pool, min(len(pool), n - len(saida)))
    return saida
mix = tira(conv, n_conv) + tira(acao, n_acao)
random.shuffle(mix)
with open("dados/sft_dose.jsonl", "w", encoding="utf-8") as f:
    for r in mix: f.write(json.dumps({"entrada": r["entrada"], "alvo": r["alvo"]}, ensure_ascii=False) + "\n")
print(json.dumps({"base": base, "dose_pct": dose, "conversa": n_conv, "acoes": n_acao,
                  "repeticoes_conversa": round(n_conv / len(conv), 2), "total": len(mix)}), flush=True)
PY


sobe () {  # sobe um arquivo para o R2 na hora; perder a instancia deixa de custar o run inteiro
  python3 - "$1" <<'PYU' 2>/dev/null || true
import boto3, os, sys
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto", config=Config())
f = sys.argv[1]
s3.upload_file(f, os.environ.get("POCKET_LM_BUCKET", "pocket-lm"), f"avaliacao/{os.path.basename(f)}")
print("subiu", f, flush=True)
PYU
}

TOK=data_big/tokenizer.json
CK="gpu_out/${NOME}/slim.pt"
echo "===== SFT dose ${DOSE}% sobre ${BASE} ====="
# --max-len 256: o mesmo da rodada de cobertura, para a comparacao valer. O default do
# train_acoes.py e 160 e descarta par maior EM SILENCIO.
python3 train_acoes.py --ckpt "gpu_out/${BASE}/slim.pt" --dados dados/sft_dose.jsonl \
  --tokenizer "$TOK" --epocas 2 --batch 32 --max-len 256 --lr 1e-4 \
  --saida "gpu_out/${NOME}" 2>&1 | tail -6

echo "########## ${NOME} ##########"
python3 eval_calame_acc.py "${CK}:${TOK}" --ban-nl 2>&1 | tail -3
cp avaliacao/calame_acc.json "avaliacao/D_${NOME}_calame.json" 2>/dev/null || true
sobe "avaliacao/D_${NOME}_calame.json"
python3 eval_lambada.py "${CK}:${TOK}" 2>&1 | tail -2
cp avaliacao/lambada.json "avaliacao/D_${NOME}_lambada.json" 2>/dev/null || true
sobe "avaliacao/D_${NOME}_lambada.json"
python3 avalia_acoes.py --ckpt "$CK" --tokenizer "$TOK" --teste acoes/massive_br_test.jsonl \
  --limite 500 --saida "avaliacao/D_${NOME}_acoes.json" 2>&1 | tail -3
sobe "avaliacao/D_${NOME}_acoes.json"
python3 avalia_acoes_parada.py --ckpt "$CK" --tokenizer "$TOK" --limite 200 \
  --saida "avaliacao/D_${NOME}_parada.json" 2>&1 | tail -2
sobe "avaliacao/D_${NOME}_parada.json"
python3 avalia_conversa.py --ckpt "$CK" --prompts-juiz "avaliacao/D_${NOME}_juiz.jsonl" \
  --saida "avaliacao/D_${NOME}_portoes.json" 2>&1 | tail -3
sobe "avaliacao/D_${NOME}_portoes.json"
sobe "avaliacao/D_${NOME}_juiz.jsonl"
python3 massive_fewshot.py --ckpt "$CK" --tokenizer "$TOK" --n 300 --shots 5 \
  --saida "avaliacao/D_${NOME}_fewshot.json" 2>&1 | tail -3
sobe "avaliacao/D_${NOME}_fewshot.json"

python3 - <<PY
import boto3, json, os, glob
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto", config=Config(retries={"max_attempts": 3}))
B = os.environ.get("POCKET_LM_BUCKET", "pocket-lm")
for f in glob.glob("avaliacao/D_*"):
    s3.upload_file(f, B, f"avaliacao/{os.path.basename(f)}")
p = "gpu_out/${NOME}/slim.pt"
if os.path.exists(p):
    s3.upload_file(p, B, f"derivados/269m/${NOME}/modelo.pt")
    print("modelo salvo: derivados/269m/${NOME}", flush=True)
s3.put_object(Bucket=B, Key=f"runs/{os.environ.get('RUN_NAME','${NOME}')}/report.json",
              Body=json.dumps({"ok": True, "dose_pct": int("${DOSE}"), "base": "${BASE}"}).encode())
print("FIM declarado", flush=True)
PY
