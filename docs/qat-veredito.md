# QAT — veredito final (31/07/2026): DESCARTADO pela régua atual

A régua antiga era "QAT entra se melhorar KLD ≥30% sobre imatrix". A pesquisa fechou:

1. Nosso KLD Q4_K_M+imatrix (0,0055) já é excepcional (típico: 0,02-0,05); mesmo
   um corte de 50% seria imperceptível com same-top-p em 95,3%.
2. O caminho maduro de QAT (torchao/Unsloth/Gemma-3-QAT) mira **Q4_0** — formato
   PIOR de partida que Q4_K_M+imatrix; exportar QAT pra Q4_K_M segue sem
   correspondência de esquema (gambiarra).
3. Nenhum número publicado de QAT vencendo imatrix Q4_K_M em <1B; requants QAT da
   comunidade ficaram na margem de erro (stduhpf: PPL 8,2335 vs 8,2323).
4. Ordem, se um dia voltar: QAT é SEMPRE a última etapa (SFT em precisão cheia
   desfaz a adaptação à grade).

**Único experimento que sobra (centavos, pós-SFT)**: Intel AutoRound exporta direto
pra GGUF:Q4_K_M em minutos (github.com/intel/auto-round) — rodar no checkpoint
final e comparar KLD com o imatrix. Reavaliar QAT só se formos a Q3/Q2.

Fontes: developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/ ·
pytorch.org/blog/quantization-aware-training/ · github.com/intel/auto-round

Relacionado: docs/runbook-pos-treino.md, docs/pesquisa-fase2.md.

## A quantização piora com mais treino? — veredito de 01/08/2026

**Risco levantado:** existe lei medindo que o dano da quantização cresce com a razão
tokens/parâmetro (D/N). A nossa acabou de subir de 17 para **50**, e iria a 150 numa
próxima extensão. Se valesse plenamente, o modelo ficaria melhor no servidor e **pior no
celular** — que destruiria a tese do produto.

**Veredito: a ameaça é real como mecanismo e mal dimensionada para o nosso caso.**

**1. O trabalho original mediu bem menos do que alega.** Kumar et al.
(arXiv:2411.04330) ajustaram em modelos de 30-220M (o nosso porte) com 1,5-26 bi tokens,
mas os experimentos chegam a **D/N ≈ 10³**. O "dado extra vira prejuízo" está em
**D/N ≈ 10⁵ e é extrapolação analítica** (Apêndice E), não medição. Estamos em **D/N=50**,
duas ordens de grandeza abaixo do que eles mediram. Os próprios autores escrevem que os
resultados são *"suggestive rather than prescriptive"*.

**2. A lei numérica utilizável** vem de Ouyang et al. (arXiv:2411.17691, 1500+ checkpoints):
ΔqLoss = 0,017·D^0,5251 / (N^0,2261·P^5,4967). Aplicada a nós:

| tokens | D/N | previsto pela lei | escalando nosso KLD medido |
|---|---|---|---|
| 4,6 bi | 17 | 0,0123 | — |
| **13,4 bi (hoje)** | **50** | 0,0215 | **0,0055 (medido)** |
| 26,9 bi | 100 | 0,0310 | ~0,0079 |
| 40,4 bi | 150 | 0,0384 | ~0,0098 |

Dois fatos: o **nosso KLD medido é 4× melhor que a lei prevê** (a lei foi ajustada em
GPTQ; Q4_K_M com imatrix é outra coisa), e mesmo aceitando o expoente pessimista chegamos
a ~0,010 a 150 tokens/param — ainda dentro do "excelente" (o típico é 0,02-0,05).

**3. E há refutação direta do mecanismo.** arXiv:2510.06213 replicou o Kumar e mostrou que
**o confundidor era o cronograma de LR, não o volume de dado**: o erro de quantização
**dispara no decaimento**, não com os tokens. Com WSD a degradação cresce mais devagar, e
no SmolLM3 o erro fica **plano por 11 trilhões de tokens** na fase estável. Nós usamos WSD.

**4. Nosso fator de risco é o TAMANHO, não o dado.** A lei tem N^−0,23: modelo maior
quantiza melhor. O Llama 3.1 em D/N≈1875 recupera 99,4-99,8% em 4 bits. O 269M é o
lado frágil da equação.

### Protocolo para medir agora (~1 dia de CPU, custo zero)

1. Gerar GGUF **F16** e **Q4_K_M com imatrix** dos **dois** checkpoints (4,6 bi e 13,4 bi).
2. Corpus de avaliação pt-BR **held-out**, disjunto do treino **e do corpus do imatrix**,
   ≥200k tokens (~400 blocos de 512).
3. `llama-perplexity ... --kl-divergence-base base.kld` para o F16 e depois
   `--kl-divergence` para o quantizado. **Cada checkpoint contra o próprio F16.**
4. Registrar média do KLD, **quantis** (o q99 pega dano concentrado em poucos tokens, que a
   média esconde), same-top-p, RMS Δp e razão de perplexidade.
5. Varredura de precisão no novo: Q4_K_M, Q5_K_M, Q6_K, Q8_0 — dá a inclinação empírica.
6. Fechar com CALAME rodado **no arquivo quantizado**, n≥1000.

**Critério de decisão:** se ΔKLD(13,4 − 4,6 bi) **< 0,005** e o same-top-p continuar
**>94%**, o efeito é irrelevante para nós e seguimos para 100-150 tokens/param.

### Se a degradação aparecer, por ganho ÷ custo

1. **Q5_K_M ou Q6_K** — a lei tem P^5,5, então 5 bits cortam o dano ~3×. Custa ir de
   186 MB para ~225 ou ~260 MB. É o maior ganho garantido por MB.
2. **WSD com pico de LR maior, weight decay maior e média de pesos (LAWA)** — medido,
   grátis, e acontece no treino.
3. `--output-tensor-type Q6_K --token-embd-type Q6_K` — poucos MB, barato.
4. **QAT** só se 1-3 falharem, e fundido ao cooldown (arXiv:2509.22935), nunca do zero.

### CORREÇÃO de uma crença nossa: imatrix "em português" é meio folclore

Nós tratávamos "imatrix calibrado em pt-BR" como vantagem. **Imatrix contra nenhum imatrix
é ganho real** — isso a nossa própria medição confirmou (KLD 0,0055 contra 0,0080). Mas
**casar o idioma da calibração não é**: em Llama 3.3 70B o efeito da língua do corpus de
calibração foi **não-significativo** (arXiv:2503.03592). Manter porque já está feito e não
custa nada, mas **não investir mais nisso** nem usar como argumento de qualidade.
