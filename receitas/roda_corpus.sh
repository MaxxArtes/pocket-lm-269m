#!/bin/bash
# Reconstroi o corpus com fontes de licenca limpa. Retomavel: cada shard concluido
# grava marcador no R2, entao pode ser morto e reiniciado sem perder trabalho.
cd /opt/pocket-lm
L=/opt/pocket-lm/avaliacao/corpus_limpo.log
roda () {
  local fonte=$1 shards=$2 alvo=$3
  echo "[$(date +%H:%M)] === $fonte shards $shards alvo $alvo ===" >> $L
  MEMCAP=1700M ./heavy.sh doppler run --project omnigen-ia --config prd -- \
    ./venv/bin/python prepara_fonte_limpa.py --fonte $fonte --shards $shards \
    --alvo-tokens $alvo --out-shard-tokens 40000000 >> $L 2>&1
  # captura o codigo ANTES de qualquer outro comando: `$(date)` dentro do proprio echo
  # executa primeiro e zera o $?, o que escondeu a falha do finepdfs em 04/08
  local cod=$?
  echo "[$(date +%H:%M)] $fonte terminou codigo $cod" >> $L
}
# finepdfs primeiro: e a unica fonte NAO-crawl, e falta de diversidade de pipeline
# e o defeito conhecido da mistura atual
roda finepdfs 0-5 3000000000
roda fineweb2hq 0-9 2000000000
roda hplt 1-40 1000000000
echo "[$(date +%H:%M)] FIM CORPUS" >> $L
