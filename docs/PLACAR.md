# PLACAR — toda versão × toda métrica

**Regra:** quando um teste novo é criado, ele é rodado em **todas as versões acompanhadas**,
não só na mais nova. Sem isso não existe parâmetro de comparação, e cada medição vira uma
ilha. Célula vazia é lacuna conhecida, não "não importa".

**Segunda regra:** número só entra aqui com o **n** e a **régua**. Foi por não registrar
isso que o CALAME da base (n=2.076) e o do v6 (n=400) viraram incomparáveis.

Atualizado em 07/08/2026.

## Versões acompanhadas

| id | o que é | onde |
|---|---|---|
| `base` | 269M treinado do zero, corpus-limpo-v1 | R2 `runs/corpus-limpo-v1` · HF |
| `oficial` | 1º adapter de ações (só MASSIVE) | R2 `runs/lora-acoes-oficial` |
| `v6` | adapter de ações + 9 funções nativas | R2 `runs/lora-acoes-nativas-v6` · HF |
| `assistente` | base + SFT de ações fundido (0,85/0,15) | HF |
| `qat2`, `qat4` | braços de QA-LoRA — **reprovados**, não são produto | local |
| `v7` | v6 + dado corrigido (calc literal, unidades, consistência, incompletude) | local — **reprovado** (call −5,9 real) |
| `v8` | v7 + tabela completa, verbos de call protegidos, +300 negativos compostos | local — **CANDIDATO a congelar** |

## Língua

| métrica | n | régua | `base` | `oficial` | `v6` | `assistente` |
|---|---|---|---|---|---|---|
| LAMBADA-PT | 5.153 | geração | 36,85 pub · **36,89** | **41,99** | **37,28** | — |
| CALAME-PT | 2.076 | geração | 53,13 pub · **53,32** | **49,90** | **48,22** | 50,9 ⚠️ |
| IFEval-pt | 308 | regras (estrita) | 39,61 pub · **41,56** | — | **42,86** ¹ | — |
| MultiBLiMP | 3.048 | par mínimo (log-verossimilhança) | 96,3 pub · **97,31** | — | **94,62** | — |

¹ IFEval do v8: empate com a base no total, mas com redistribuição — max_frases 0/28→13/28
(o treino de ações ensinou concisão) e ênclise 13→5. Desloca superfície, não apaga capacidade.

⚠️ o 50,9 do `assistente` veio de outra rodada, sem n registrado — tratar como indicativo.

**Medição pareada de 07/08** (mesmos itens, mesma régua, mesmo run — `lingua_pareado.json`):

| | base | base+v6 | Δ | McNemar |
|---|---|---|---|---|
| LAMBADA-PT | 36,89 | **37,28** | +0,39 | base+433 / v6+453 · **p=0,52 → ruído** |
| CALAME-PT | 53,32 | **48,22** | **−5,10** | base+208 / v6+102 · **p=0,0 → real** |

**Conclusão:** o adapter de ações NÃO custa LAMBADA — o número que sustenta "1º entre
modelos <1,2B em português" sobrevive com o motor ligado. Mas custa **5,1 pontos de
CALAME**, e a queda é estatisticamente inequívoca. Eu havia estimado ~2 pontos pelo card
do `assistente`; era menos da metade do real.

A esteira reproduz o publicado (36,85→36,89, 53,13→53,32, 96,3→97,31), então a diferença é
do adapter e não da medição.

### POR QUE o adapter machuca CALAME e não LAMBADA — respondido (07/08)

**Camada 1 — aritmética.** O adapter perturba os dois quase igual; o líquido depende de
onde a base começou:

| | certos | errados | perdeu | ganhou | taxa perda | taxa ganho | líquido |
|---|---|---|---|---|---|---|---|
| CALAME | 1.107 | 969 | 208 | 102 | 18,8% | 10,5% | **−106** |
| LAMBADA | 1.901 | 3.252 | 433 | 453 | 22,8% | 13,9% | **+20** |

Aplicando as MESMAS taxas médias aos dois, o previsto é −112 e +3 — contra −106 e +20
reais. O CALAME parte de 53% e tem pouco a recuperar; o LAMBADA parte de 37% e recupera na
mesma proporção em que perde. **Não há nada de especial no CALAME.**

**Camada 2 — a RÉGUA, e essa é a que importa mais:**

| benchmark | régua | itens que mudaram | taxa de perda |
|---|---|---|---|
| **MultiBLiMP** | verossimilhança da frase | **4,0%** | **3,4%** |
| CALAME | geração gulosa | 14,9% | 18,8% |
| LAMBADA | geração gulosa | 17,2% | 22,8% |

Geração gulosa é **5 a 7× mais sensível** ao adapter que verossimilhança de frase. `argmax`
de um token se decide por diferença mínima entre candidatos; somar log-probabilidade sobre
a frase inteira dilui o deslocamento.

**Conclusão:** o adapter NÃO apagou conhecimento de língua — a gramática segue em 94,6%
(acaso 50%). Ele deslocou qual token fica em primeiro lugar. É mudança de superfície, não
perda de capacidade. Toda comparação de versão precisa reportar a régua junto, senão
compara sensibilidade de medição e não capacidade de modelo.

### ACHADO GRANDE (07/08): o adapter `oficial` MELHORA o LAMBADA em 5,1 pontos

| | LAMBADA-PT | CALAME-PT |
|---|---|---|
| base | 36,89 | 53,32 |
| **base + `oficial`** | **41,99** | 49,90 |
| base + `v6` | 37,28 | 48,22 |

McNemar do `oficial` em LAMBADA: base+295 / oficial+558, **p=0,0** — ganho real, não ruído.

**Duas consequências:**

1. **O 36,85 publicado pode estar subestimando o modelo.** Com 20 MB de adapter sobre os
   mesmos pesos, é 41,99. Usar esse número exige declarar que é com o adapter — mas 5
   pontos é margem grande demais pra ignorar.
2. **As 9 funções nativas custaram 4,7 pontos de LAMBADA** (`v6` 37,28 vs `oficial` 41,99),
   além dos 8 de hard-OOS. Na tarefa original não custaram nada.

**Preço medido das 9 funções nativas** (v6 contra oficial, mesma esteira, mesmo dia):

| | Δ |
|---|---|
| hard-OOS | **−8,0** |
| LAMBADA | **−4,7** |
| intenção MASSIVE | −0,7 (ruído) |
| acerto conjunto MASSIVE | −0,3 (ruído) |

Vale a troca? Hoje sim — as funções são a razão do produto existir. Mas agora é uma troca
com preço medido, não um custo invisível.

## Ação — intenção e campos

| métrica | n | `base` | `oficial` | `v6` |
|---|---|---|---|---|
| intenção MASSIVE | 60 int. | n/a¹ | 88,6 | **90,0** |
| intenção MASSIVE (harness de previsões) | 300 | n/a¹ | **88,7** | **88,0** |
| **acerto conjunto MASSIVE** | 300 | n/a¹ | **65,0** | **64,7** |
| hard-OOS (harness de previsões) | 150 | n/a¹ | **62,7** | **54,7** |
| F1 de campos | — | n/a¹ | 78,2 | **79,9** |
| recusa fora de escopo | 150 | n/a¹ | 97,7 | **99,0** / 98,0 |
| falso-desconhecido | 300 | n/a¹ | 1,4% | **1,0%** |
| hard-OOS | 150 | n/a¹ | **61,0** | 54,5 / 54,7 |
| 9 funções nativas | 1.051 | não existem | não existem | **87,4** |

¹ a base pura não produz `intencao;campo=valor` — sem adapter ela completa texto.

## Features — acerto conjunto, 3 réguas (n≈120 cada, total 1.051)

| feature | `oficial` | `v6` estrito | `v6` canônico | `v6` teto |
|---|---|---|---|---|
| call | — | 70,6 | **95,0** | 95,0 |
| device_toggle | — | 91,3 | **91,3** | 91,3 |
| app_open | — | 88,0 | **88,0** | 88,0 |
| flashlight | — | 87,6 | **87,6** | 87,6 |
| timer_set | — | 79,2 | **79,2** | 79,2 |
| unit_convert | — | 45,0 | **60,8** | 72,5 |
| message_send | — | 38,3 | **54,2** | 54,2 |
| reminder_set | — | 47,5 | **50,8** | 50,8 |
| calc | — | 3,3 | **3,3** | 54,2 |
| **TOTAL** | — | 60,5 | **67,3** | 69,3 |
| campo ausente | — | — | **0,6%** | — |
| campo inventado | — | — | **10,6%** | — |

Lacuna: o `oficial` nunca foi medido sob acerto conjunto. Como ele não tem as 9 funções,
só as intenções do MASSIVE seriam comparáveis — vale medir para saber se o v6 melhorou ou
piorou a extração de campo.

## Iteração v7/v8 (07-08/08, noite) — acerto conjunto canônico

| feature | v6 | v7 | v8 | McNemar decisivo |
|---|---|---|---|---|
| call | 95,0 | 89,1 | **94,1** | v6×v7: 7:0 p=0,016 (real); recuperado no v8 |
| device_toggle | 91,3 | 91,3 | **92,3** | |
| timer_set | 79,2 | 78,3 | **87,5** | v6×v8: 12:2 p=0,013 (ganho REAL) |
| flashlight | 87,6 | 86,7 | 86,7 | |
| app_open | 88,0 | 86,3 | 86,3 | |
| calc | 3,3 | 71,2 | **70,2** | convenção consertada |
| reminder_set | 50,8 | **58,5** | 51,7 | |
| unit_convert | 60,8 | 55,0 | 49,2 (teto 65,8) | régua meio-canonizada nas 2 versões |
| message_send | 54,2 | **57,5** | 44,2 | v6×v8: 21:9 p=0,043 (queda REAL) |
| **TOTAL** | 67,3 | **74,6** | 73,2 | |
| MASSIVE conjunto | 64,7 | 66,3 | **68,3** | |
| hard-OOS | 54,7 | 54,7 | **57,3** | portão movido pro SISTEMA (modelo+regra ≈74) |

**Causa da queda da mensagem no v8 (verificada item a item):** os 300 negativos compostos
("entra no zap e manda um áudio" → desconhecido) envenenaram mensagens genuínas que citam
o app como CANAL ("Envia um WhatsApp pro meu irmão" → desconhecido). Terceira encarnação
do mesmo padrão do projeto (marca→desconhecido; luz_casa). A fronteira "app como canal"
(in-scope) vs "abrir app E agir dentro" (OOS) é sutil demais para 269M com 300 exemplos.

**Língua do v8 (medição pareada, conjuntos completos):**

| | base | v8 | veredito |
|---|---|---|---|
| LAMBADA-PT (5.153) | 36,89 | **38,39** | McNemar 507:430 p=0,013 — v8 MELHOR que a base |
| CALAME-PT (2.076) | 53,32 | 48,36 | mesmo custo do v6 (48,22) |
| MultiBLiMP (3.048) | 97,31 | **95,34** | melhor que o v6 (94,62) |

**CONGELADO (08/08, 23h30): v8 é o novo oficial.** Todos os portões passaram; LAMBADA
acima da própria base. Artefatos em R2 `runs/lora-acoes-v8/` (adapter + dado de treino +
teste). Produto continua com as MESMAS 4 features ligadas + regra de confiança; timer
(87,5) fica desligado até 95 por executar sem confirmação.

**Decisão (análise 3, agy): CONGELAR o v8.** As 4 features do produto empatam com o v6
(89,9 vs 90,5, ruído); v8 ganha MASSIVE, hard-OOS, timer e calc; mensagem/lembrete caem
mas estão DESLIGADOS. Timer 87,5 ainda não habilita (executa sem confirmação; régua 95).

## Quantização — 13 níveis

Medido no **`v6` fundido** (13 níveis) e na **`base` pura** (5 níveis, CALAME completo
n=2076; timeout no Q4 — `matriz_quant_base.json`): F16 53,85 · Q8 53,56 · Q6 54,09 ·
Q5 54,14 · Q4 53,08 — **nenhuma perda mensurável na base até Q4_K_M**, consistente com o
achado do v6. `oficial` não foi medido (valor marginal).
Tabela completa em [`quantizacao-269m.md`](quantizacao-269m.md).

| nível | MB | intenção | F1 | nativas | recusa | CALAME(400) |
|---|---|---|---|---|---|---|
| F16 | 514,6 | 87,67 | 75,38 | 88,33 | 98,0 | 49,25 |
| Q4_K_M | 168,8 | 87,00 | 75,16 | 87,67 | 98,0 | 49,75 |
| Q3_K_S | 118,8 | 88,00 | 76,38 | 88,33 | 97,33 | 48,75 |
| IQ2_M+imat | 100,1 | 86,67 | 73,63 | 88,67 | 96,0 | 48,50 |
| IQ1_S+imat | 70,8 | 78,67 | 60,36 | 64,33 | 85,33 | 36,50 |

## Quantização do v8 (formatos de entrega, com imatrix, teste v8)

| nível | MB | intenção | conjunto | nativas | recusa | CALAME(400) |
|---|---|---|---|---|---|---|
| F16 | 514,6 | 89,33 | 64,33 | 86,33 | 98,0 | 50,00 |
| **Q4_K_M** (entrega) | 168,8 | 89,00 | 63,67 | 86,67 | 98,0 | 50,50 |
| Q3_K_S | 118,8 | 88,00 | 62,33 | 85,67 | 98,0 | 49,75 |
| IQ2_M | 100,1 | 85,67 | 59,67 | 85,67 | 97,33 | 47,00 |

Mesma escada do v6: nada mensurável até Q3_K_S; IQ2_M cobra 3-4pp. A recusa do v8 aguenta
melhor a compressão que a do v6 (97,33 vs 96,0 no IQ2_M). **Arquivo de entrega validado.**

Sistema (modelo+regra) no v8: hard-OOS 55,0 cru → **74,5 com a regra** (medido,
completo_v6_vs_v8.json). Robustez: typos 61→**68** vs v6; caixa baixa 84 (ruído).

## Aparelho real

| medida | aparelho | `v6` Q4_K_M |
|---|---|---|
| velocidade | Galaxy A14 (Dimensity 700) | **45–63 tok/s** |
| RAM livre | Galaxy A14 | **1,46 GB** |
| troca de modo | Galaxy A14 | **~50 ms** |
| troca de modo | bancada x86 | 1 ms |

## Lacunas — EM ORDEM DE VELOCIDADE (mais rápida primeiro)

Tempos estimados dos ritmos medidos em 07/08: CALAME 2.076 itens ≈ 60 s/modelo,
LAMBADA 5.153 ≈ 180 s/modelo, previsões (1.651 itens) ≈ 100 s, quantização ≈ 10 min/nível.
Some ~3 min de subida de container em cada job.

| # | lacuna | tempo | observação |
|---|---|---|---|
| 1 | `oficial` sob acerto conjunto | **~5 min** | pipeline de previsões, pontuação local é instantânea |
| 2 | robustez na `base` (caixa baixa, sem acento, typos) | **~5 min** | mesma esteira |
| 3 | `v6` em IFEval-pt (n=308) | **~6 min** | script existe; geração mais longa que última-palavra |
| 4 | `oficial` em LAMBADA + CALAME | **~7 min** | mesmo job pareado, outro adapter |
| 5 | `base` no MASSIVE com few-shot | **~10 min** | prompt longo; é a LINHA DE BASE que falta |
| 6 | `assistente` em tudo | **~15 min** | está publicado no HF e quase não foi medido |
| 7 | quantização da `base` pura (13 níveis) | **~2 h** | só o `v6` fundido foi varrido |
| 8 | MultiBLiMP em qualquer versão | **bloqueado** | ⚠️ não há script no repo |

**De 1 a 5 cabem num job só** — mesma estrutura, parâmetros diferentes. Umas 25 min no
total, contra ~33 min se rodados separados, por causa da subida de container repetida.

⚠️ **MultiBLiMP:** o `96,3%` que está no card do Hugging Face e neste placar **não é
reproduzível aqui** — não existe script. Não sei de onde saiu nem com que régua. Ou alguém
recupera o script, ou o número deveria sair dos materiais públicos até ser refeito. Número
que não se reproduz é passivo, não ativo.
