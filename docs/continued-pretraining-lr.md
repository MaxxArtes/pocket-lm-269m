# Continuar treino de checkpoint decaído — o que a literatura manda (30/07)

Pesquisa motivada pela falha do ext1 (US$2, re-warmup ao pico total + decay de 180s
→ val 2,55→2,81) e pra calibrar o ext2 (rodando: LR 1/3 do pico, 50% decay).

## Achados com fonte

1. **Re-warmup em dados IGUAIS é dano puro.** Ibrahim et al. (TMLR 2024,
   https://arxiv.org/abs/2403.08763, §7.1): re-aquecer ao pico causa spike de +0,10
   de val loss (+0,20 com 2× pico) e "fail to recover quickly". Re-warmup só se paga
   quando há MUDANÇA de distribuição (adaptação a dado novo). Nosso caso (mesmo
   ClassiCC) não tem esse ganho pra compensar o dano.
2. **OLMo 2 nem re-aquece** (https://arxiv.org/pdf/2501.00656): mid-training continua
   do LR onde parou e anela linear até zero ao longo de ~1% do orçamento total, com
   souping de 3 seeds.
3. **LR ótimo de continuação CAI com o compute já acumulado**
   (https://arxiv.org/pdf/2606.05610): pra um 8B denso, previsto 5e-5 — bem abaixo
   do pico de pré-treino.
4. **Infinite schedules** (Ibrahim; Singh et al. https://arxiv.org/abs/2503.02844):
   warmup→cooldown→constante (~55% do pico)→annealing final elimina o problema do
   re-warmup entre ciclos. O WSD/MiniCPM (https://arxiv.org/abs/2404.06395) resolve
   igual: retomar SEMPRE do checkpoint da fase estável (pré-decay) — nunca do decaído.
5. **Decaimento precisa de 10-20% do total** (Hägele et al.
   https://arxiv.org/abs/2405.18392; forma 1-sqrt > linear). Nossos 180s do ext1 eram
   ~3% do decay original — consistente com a falha.

## Implicações práticas

- **ext2 (rodando)**: LR 1/3 do pico ainda re-aquece demais pro nosso caso. Vale como
  ponto de dado (medir o spike e a recuperação), não como receita final.
- **ext3 (se ext2 não bater 2,55)**: estilo OLMo 2 — SEM re-warmup, partir de ≤1/10
  do pico (1,2e-4) e decair linear a zero pelo orçamento inteiro. Decay em tempo
  absoluto ≥ o original (~5.760s) se a meta é recuperar/passar 2,55.
- **Mudança estrutural pro próximo treino grande**: salvar SEMPRE um checkpoint da
  fase estável (pré-decay) separado do final — é o único jeito de estender do jeito
  canônico do WSD. Hoje o save sobrescreve um arquivo único e o pré-decay se perde.

Relacionado: [[pocket-lm-estado-29-07]], docs/receita-treino.md, docs/gap-lambada-calame.md.

## Val subindo no ext3 — diagnóstico com número (31/07)

**Sintoma:** val (medida no corpus ANTIGO) oscilando em 2,59-2,61 contra 2,5541 da
largada, enquanto train cai devagar (2,45). **Veredito: esperado, e mais brando que o
publicado.** Não abortar.

**Por quê.** Ibrahim et al. (arXiv:2403.08763, TMLR 2024) mediram em **405M** — escala
quase igual à nossa — que **re-aquecer o LR sozinho, sem trocar dado nenhum**, já sobe a
val do corpus antigo: +0,10 com ηmax=3e-4, +0,20 com 6e-4, treinando no MESMO Pile. Com
shift fraco de domínio, +0,35 a +0,45. Nossa subida é de **+0,04 a +0,06** — de 2× a 9×
menor que o análogo publicado. A explicação é a nossa mistura: mantivemos 62% do corpus
original, o que é "replay" de 62%, muito acima dos 1-10% que eles testaram (na Tabela 2
deles, 50% de replay empata ou supera o baseline treinado na união dos dois corpora).

**O que esperar do decaimento.** OLMo 2 (arXiv:2501.00656) alerta direto para o nosso
caso: "a shorter hyperparameter experiment might come to the wrong conclusion" — os 4 LRs
testados terminam na MESMA loss depois de decair. Nosso ext2 mediu -0,021 com 50% do
orçamento em decay; com 20% (nosso caso, dentro da faixa ótima de 10-20% de Blakeney
et al., arXiv:2406.03476), esperar entre -0,015 e -0,03. **É bem possível terminar em
2,56-2,58, ou seja, acima do ponto de partida na val antiga — e isso não é falha**: é o
modelo trocando ajuste ao mix velho por capacidade no mix novo, que é o objetivo.

### Critério objetivo de aborto (vale para as próximas horas)

Abortar se **qualquer um**:
- val antiga **> 2,70** (+0,15) **e ainda subindo** depois de 30% do orçamento (~passo 4.800);
- **train loss subindo** em janela de 500 passos, ou spike que não recupera (foi assim
  que o OLMo 2 abandonou o LR de 30e-4);
- val antiga plana **e** train plana por mais de 20% do orçamento (não aprende nada novo).

Não é alarme: oscilar 2,59-2,61 com train em 2,45 caindo. O gap aqui é de **distribuição**
(mixes diferentes), não de generalização.

### Dois erros de método que isto expõe

1. **Não temos val do mix NOVO** — só do antigo. A val antiga é monitor de esquecimento,
   não medidor de sucesso; quem decide "está funcionando" é a val no domínio-alvo. Na
   próxima preparação de corpus, **separar shards de validação ANTES de treinar** (custo
   zero se feito na hora, caro depois).
2. **Não comparar CALAME de checkpoint em LR constante contra baseline decaído** — é
   comparação injusta e vai parecer pior mesmo quando for melhor. Avaliação só depois do
   decaimento. O OLMo 2 usa micro-anneals justamente por isso.

### 31% de sintético não é agressivo

Nemotron-CC usa 30,2% de sintético no corpus (45,5% no subconjunto HQ do run de 8B/15T);
o mix de mid-training do OLMo 2 tem 20,8% de Dolmino Math (quase todo sintético) + 16,6%
FLAN. Model collapse (Shumailov, Nature 2024) exige recursão auto-alimentada
**substituindo** dado real; Gerstgrasser et al. (arXiv:2404.01413) mostram que **acumular**
real+sintético dá erro com teto finito. Sintético de outro modelo + 68% de dado real não
é regime de colapso.
