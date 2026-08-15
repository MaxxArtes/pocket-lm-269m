# Análise — adapters de produto (adapter-app-v1 / adapter-whats-v1)

Data: 14/08/2026, horários em hora de Cuiabá. Bateria fechada em três rodadas:
parada + portões + arquivos do juiz na noite de 13/08 (~21:30), CALAME / LAMBADA /
MASSIVE-1035 de manhã (07:36–07:48, run `avalia-adapters`), e o 5-shot no run
`bateria-adapters` (lançado 08:26, RTX 4090 fracionada a US$ 0,27/h; app pousou 09:11,
whats 09:49; duração total ~85 min, instância destruída pelo vigia). Juiz cego rodado
na bancada ~08:35 sobre os `P_*_juiz.jsonl` (52 coerência + 15 honestidade por modelo);
resultado em `avaliacao/juiz_adapters.json`. Custo da rodada: ~US$ 0,40 de GPU +
~US$ 0,05 de juiz.

Receita dos adapters: SFT completo sobre `full-executor-v1`, dose 7% de conversa,
**replay 25% ligado** (`DATA_BIN=replay.bin`), max-len 384; o whats soma +757 pares
com bloco de contexto (`ctx_vultr_v1.jsonl`).

## 1. Tabela completa

Intenção/F1 dos adapters em dois conjuntos: n=500 (comparável com a cobertura) e
n=1035 (conjunto canônico do v8, mesma ordem — é o que o McNemar usa).

| modelo | CALAME | LAMBADA | intenção | F1 | F1-parada | 5-shot | distinct-2 | coerência | honest. |
|---|---|---|---|---|---|---|---|---|---|
| **adapter-app-v1** | **42,97** | **24,01** | 87,4 · **87,92** | 73,08 · 74,84 | **75,47** | 41,33 | 0,668 | 78,85 | **92,86** |
| **adapter-whats-v1** | 42,77 | 23,79 | 86,2 · 86,96 | 72,75 · 74,53 | 75,32 | **43,0** | 0,604 | 78,85 | 66,67 |
| full-executor-v1 (base, sem SFT) | 49,86 | 35,01 | 86,4 | 37,96 | 73,65 | 35,67 | 0,682 | 34,62 | 13,33 |
| full-executor-v1-multi (SFT sem replay) | 33,67 | 19,37 | 83,0 | 69,16 | 72,50 | 44,33 | 0,625 | 92,16 | 93,33 |
| dose7-full-executor-v1 (7%, sem replay) | 18,11 | 7,84 | 87,2 | 72,07 | 74,93 | 19,33 | 0,637 | 65,38 | 66,67 |
| dose18-full-executor-v1 (18%, sem replay) | 19,12 | 8,09 | 87,2 | 71,30 | 74,22 | 10,67 | 0,722 | 65,38 | 80,0 |
| v8 (oficial, produto atual) | 48,36 | 38,39 | 89,7 (conjunto 68,3) | — | — | — | — | — | — |

Avisos que viajam com a tabela:
- **CALAME/LAMBADA das linhas de cobertura/dose com SFT foram treinados SEM replay**;
  os adapters COM replay 25%. Essa é a comparação central da rodada, não um vício.
- **Coerência/honestidade têm ~9 pp de ruído do próprio juiz** (mesma entrada, temp 0);
  diferença < 10 pp não significa nada. Honestidade tem n=15 — granularidade de 6,7 pp
  por item.
- Distinct-2 ≥ 0,70 é o portão do produto de jogo.
- `ganho_do_terminador` = 0,0 nos dois adapters (o terminador foi aprendido; F1-parada
  e F1 cru coincidem).
- 5-shot: rótulo válido caiu para 57,7% (app) / 62,0% (whats) contra 84,7% do dose7 —
  o acerto maior vem com menos "obediência de formato" no ICL.

## 2. Conclusões

1. **O replay pagou — e não custou nada.** Mesma dose de 7%, mesmo SFT: sem replay
   (dose7) CALAME 18,11 / LAMBADA 7,84; com replay 25% (adapter-app) 42,97 / 24,01
   (+24,9 / +16,2 pp). Execução intacta (F1-parada 74,93 → 75,47; intenção 87,2 → 87,4)
   e o 5-shot saltou de 19,33 para 41,33 — acima até da base sem SFT (35,67). Replay em
   produção fica confirmado como regra, não opção.
2. **O contexto do whats não custou nada mensurável nas métricas duras.** Deltas
   whats−app: −0,20 CALAME, −0,22 LAMBADA, −0,96 intenção (n=1035), −0,31 F1, +1,67 no
   5-shot — tudo dentro do ruído. Os +757 pares de contexto entraram de graça; se o
   bloco de contexto funcionar no produto, é ganho líquido.
3. **A única diferença fora do ruído entre os dois é honestidade (66,67 vs 92,86) — e
   ainda assim é sinal fraco.** São 26 pp, acima dos ~9 pp do juiz, mas com n=15 é
   diferença de 3 aceites (10 vs 13), e a contagem automática "disse não sei" deu 15/15
   nos DOIS. Antes de concluir que o contexto degradou a recusa, re-julgar as mesmas 15
   armadilhas 2–3 vezes; não iterar treino por causa disso ainda.
4. **Contra o v8: ameaça na execução e em conversa, perde em língua.** A decisão formal
   saiu no McNemar pareado de 14/08 (ver §4 abaixo): no run pareado o v8 fez 87,73 de
   intenção contra 87,92 do candidato — 1 item líquido em 1.035, ruído — e ficou 3,1 pp
   acima em F1 de campos. O candidato traz o que o v8 não tem: coerência de conversa ~79 e
   honestidade 93 (app). Mas em língua o v8 segue claramente acima: CALAME 48,36 vs
   42,97 e sobretudo LAMBADA 38,39 vs 24,01 — o replay de 25% recuperou 2/3 do CALAME
   da base (49,86) e só ~55% do LAMBADA (35,01). O v8 fechou ACIMA da base em LAMBADA;
   os adapters não chegam perto disso.

## 3. O que ficou pior que o esperado (candidatos a iteração)

1. **Distinct-2 reprova o portão do jogo nos dois** (0,668 app / 0,604 whats < 0,70) —
   é o ÚNICO portão automático reprovado (`VEREDITO_AUTOMATICO` dos portões). O dose18
   sem replay fez 0,722, então não é teto do modelo: é dose/dado. Iteração barata antes
   de retreinar: decoding (penalidade de repetição / temperatura) na inferência do
   produto; se não bastar, mais variedade no dado de conversa.
2. **Desvio factual disparou só em 4/10** nos dois adapters (o desenho pede desviar de
   pergunta factual/numérica), e as respostas às contas básicas saem tortas ("conta
   simples, e mais dois"). Faltam pares desse tipo no dado — é o mesmo remédio do item 1.
3. **LAMBADA 24 é o número mais distante do v8** (38,39). Se o McNemar não reprovar a
   execução, é aqui que o candidato ainda não justifica a troca: testar replay maior que
   25% ou uma fase curta de anneal com dado premium antes do SFT.
4. Menor: o 5-shot só emite rótulo válido em ~58–62% das saídas (dose7: 84,7%). O acerto
   subiu mesmo assim, mas se ICL virar feature de produto, vale um olhar no formato.

## 4. Veredito do McNemar pareado (14/08, n=1035)

| modelo | intenção | F1 campos |
|---|---|---|
| v8 (oficial) | 87,73 | **77,96** |
| adapter-app-v1 | **87,92** | 74,84 |

Mesmo conjunto, mesma ordem, agregado pareado (R2 `avaliacao/mcnemar_resumo.json`;
também no placar vivo, https://pocket.omnigen.ia.br/placar/). Vantagem líquida do
candidato: **1 item de intenção** — dentro do ruído. F1 de campos do v8 3,1 pp acima.

**v8 permanece o oficial.** O candidato só justificaria a troca fechando o F1 de campos
e a LAMBADA (item 3 da seção anterior), sem perder a coerência de conversa que é o seu
diferencial.

Nota de método: como o v8 é LoRA e o candidato é SFT completo, a comparação exigiu
fundir o adapter na base (chaves gguf → state_dict) com **portão de sanidade** — ≥60 de
intenção em 50 itens antes de valer medição. Detalhe em `docs/PLACAR.md`, seção do
McNemar pareado.
