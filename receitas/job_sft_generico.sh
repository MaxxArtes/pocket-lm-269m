#!/usr/bin/env bash
# SFT GENÉRICO (multitarefa): conversa + execução no MESMO treino, sem viés de produto.
#
# Objetivo: ver até onde cada base chega quando ensinada nas duas habilidades ao mesmo
# tempo — em vez de uma SFT por especialidade. É o "generalista" que faltava na coleção.
#
# Roda nas TRÊS bases, com a mesma receita, para a comparação valer:
#   corpus-limpo-v1  (genérica)   -> o generalista de referência
#   full-conversa-v1 (especialista conversa)
#   full-executor-v1 (especialista executor)
#
# Os dois conjuntos têm o MESMO formato ({entrada, alvo}), então dá para concatenar e
# embaralhar — o modelo aprende as duas tarefas juntas, não em sequência (SFT em sequência
# faz a segunda tarefa apagar a primeira).
set -uo pipefail
cd "$(dirname "$0")"
pip install -q --no-input tokenizers numpy boto3 2>&1 | tail -1 || true

python3 - <<'PY'
import boto3, json, os, random
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto", config=Config(retries={"max_attempts": 3}))
B = os.environ.get("POCKET_LM_BUCKET", "pocket-lm")
for d in ("data_big", "dados/conversa", "acoes", "avaliacao"):
    os.makedirs(d, exist_ok=True)
for k, dest in (("aux/tokenizer.json", "data_big/tokenizer.json"),
                ("dados/conversa/sft_treino.jsonl", "dados/conversa/sft_treino.jsonl"),
                ("acoes/sft_acoes.jsonl", "acoes/sft_acoes.jsonl"),
                ("acoes/massive_br_test.jsonl", "acoes/massive_br_test.jsonl")):
    s3.download_file(B, k, dest); print("baixou", dest, flush=True)

conv = [json.loads(l) for l in open("dados/conversa/sft_treino.jsonl", encoding="utf-8")]
acao = [json.loads(l) for l in open("acoes/sft_acoes.jsonl", encoding="utf-8")]
print(f"conversa={len(conv)} acoes={len(acao)}", flush=True)
# Equilibra: a tarefa com mais exemplos domina o gradiente e a outra vira ruído.
# Corta a maior no tamanho da menor, em vez de repetir a menor (repetir foi o erro do
# pré-treino: 148 épocas sobre 1,9M de tokens).
n = min(len(conv), len(acao))
random.seed(7)
mix = random.sample(conv, n) + random.sample(acao, n)
random.shuffle(mix)
with open("dados/sft_generico.jsonl", "w", encoding="utf-8") as f:
    for r in mix:
        f.write(json.dumps({"entrada": r["entrada"], "alvo": r["alvo"]}, ensure_ascii=False) + "\n")
print(f"mix multitarefa: {len(mix)} exemplos ({n} de cada)", flush=True)
import torch
for base in ("corpus-limpo-v1", "full-conversa-v1", "full-executor-v1"):
    os.makedirs(f"gpu_out/{base}", exist_ok=True)
    s3.download_file(B, f"runs/{base}/ckpt.pt", f"gpu_out/{base}/ckpt.pt")
    d2 = torch.load(f"gpu_out/{base}/ckpt.pt", map_location="cpu", weights_only=False)
    torch.save({k2: v for k2, v in d2.items() if k2 in ("model","cfg","step")}, f"gpu_out/{base}/slim.pt")
    os.remove(f"gpu_out/{base}/ckpt.pt")
    print("base pronta:", base, flush=True)
PY

for BASE in corpus-limpo-v1 full-conversa-v1 full-executor-v1; do
  echo "===== SFT GENERICO sobre ${BASE} ====="
  python3 train_acoes.py --ckpt "gpu_out/${BASE}/slim.pt" \
    --dados dados/sft_generico.jsonl --tokenizer data_big/tokenizer.json \
    --epocas 2 --batch 32 --max-len 256 --lr 1e-4 \
    --saida "gpu_out/gen-${BASE}" 2>&1 | tail -10
  CK="gpu_out/gen-${BASE}/slim.pt"

  echo "===== ${BASE}: EXECUTA? ====="
  python3 avalia_acoes.py --ckpt "$CK" --tokenizer data_big/tokenizer.json \
    --teste acoes/massive_br_test.jsonl --limite 300 \
    --saida "avaliacao/gen_${BASE}_acoes.json" 2>&1 | tail -6
  echo "===== ${BASE}: CONVERSA? ====="
  python3 avalia_conversa.py --ckpt "$CK" \
    --prompts-juiz "avaliacao/gen_${BASE}_juiz.jsonl" \
    --saida "avaliacao/gen_${BASE}_portoes.json" 2>&1 | tail -10
done

python3 - <<'PY'
import boto3, json, os, glob
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto", config=Config(retries={"max_attempts": 3}))
B = os.environ.get("POCKET_LM_BUCKET", "pocket-lm")
for f in glob.glob("avaliacao/gen_*"):
    s3.upload_file(f, B, f"avaliacao/{os.path.basename(f)}"); print("subiu", f, flush=True)
s3.put_object(Bucket=B, Key=f"runs/{os.environ.get('RUN_NAME','sft-generico')}/report.json",
              Body=json.dumps({"ok": True, "tipo": "sft_generico"}).encode())
print("FIM declarado", flush=True)
PY
