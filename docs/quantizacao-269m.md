# Quantização do Pocket-LM 269M — matriz completa medida

Documento de referência: **quanto se perde em cada nível de quantização**, medido neste
modelo em vez de herdado das tabelas de 7B da comunidade. Modelo: 269M denso, GPT, MHA,
20 camadas, d_model 1024, FFN 4096, vocab 16384, embeddings amarradas. O adapter LoRA v6
(9 funções nativas de celular) está **fundido na base** — é assim que o produto seria
distribuído.

Mesmo conjunto de teste em todos os níveis: 300 MASSIVE pt-BR, 300 nativas, 150 fora de
escopo, 150 fora-de-escopo difícil, 400 CALAME-PT. Sementes fixas.

## Como ler estes números

Antes da tabela, os intervalos de confiança de 95%. **Diferença menor que o IC não é
resultado, é ruído** — e essa régua invalidou três leituras que eu tinha feito.

| métrica | n | IC 95% |
|---|---|---|
| intenção MASSIVE | 300 | ±3,7pp |
| F1 de campos | 300 | ~±4,0pp |
| funções nativas | 300 | ±3,7pp |
| recusa fora de escopo | 150 | ±2,7pp |
| hard-OOS | 150 | **±8,0pp** |
| CALAME-PT | 400 | ±4,9pp |

## Sem matriz de importância

| nível | MB | intenção | F1 campos | nativas | recusa | hard-OOS | CALAME | tok/s |
|---|---|---|---|---|---|---|---|---|
| F16 | 514,6 | 87,67 | 75,38 | 88,33 | 98,00 | 46,00 | 49,25 | 14,9 |
| Q8_0 | 274,6 | 87,33 | 75,71 | 88,33 | 98,00 | 46,00 | 49,00 | 38,7 |
| Q6_K | 212,6 | 87,67 | 75,99 | 88,67 | 98,00 | 47,33 | 49,00 | 25,3 |
| Q5_K_M | 190,0 | 87,33 | 75,72 | 88,67 | 97,33 | 47,33 | 49,25 | 25,4 |
| **Q4_K_M** | 168,8 | 87,00 | 75,16 | 87,67 | 98,00 | 51,33 | 49,75 | 25,3 |
| Q3_K_M | 141,3 | 89,67 | 75,99 | 87,33 | 97,33 | 49,33 | 50,50 | 24,6 |
| Q3_K_S | 118,8 | 88,00 | 76,38 | 88,33 | 97,33 | 51,33 | 48,75 | 26,1 |
| Q2_K | 110,7 | 86,00 | 71,97 | 87,33 | 96,67 | 61,33 | 45,75 | 26,0 |

## Com matriz de importância

Calibrada com 4.000 comandos do próprio produto, 200 chunks, contexto 512.

| nível | MB | intenção | F1 campos | nativas | recusa | hard-OOS | CALAME | tok/s |
|---|---|---|---|---|---|---|---|---|
| Q2_K | 110,7 | 87,00 | 74,46 | 86,67 | 97,33 | 44,00 | 47,00 | 31,1 |
| **IQ2_M** | 100,1 | 86,67 | 73,63 | 88,67 | 96,00 | 38,00 | 48,50 | 28,3 |
| Q2_K_S | 96,3 | 87,33 | 73,08 | 84,00 | 96,67 | 44,67 | 45,50 | 34,0 |
| IQ2_XXS | 80,3 | 84,33 | 70,07 | 81,33 | 86,00 | 28,67 | 43,75 | 30,8 |

`Q2_K_S` e toda a família `IQ` **recusam quantizar sem imatrix** (retornam erro em vez de
gerar peso ruim silenciosamente). `Q2_K` e os `K_M`/`K_S` aceitam.

## Conclusões

**1. De F16 até Q3_K_S não há perda mensurável.** Toda a variação cabe nos IC acima. Isso
diverge do que se reporta para modelos de 7B, onde Q3 já dói. Hipótese: a tarefa aqui é
classificar e extrair campos, não gerar texto livre — bem mais tolerante a peso grosseiro.

**2. "Q2_K" neste modelo quase não é 2 bits.** As regras de mistura do llama.cpp, aplicadas
à nossa arquitetura:

| tensor | sob Q2_K | sob Q3_K_S |
|---|---|---|
| `token_embd` (amarrado → regra de output) | Q6_K | Q6_K |
| `attn_qkv` (categoria "sensível"; MHA, n_gqa=1) | Q3_K | Q3_K |
| `attn_output` | Q3_K | Q3_K |
| `ffn_down` | Q3_K | Q3_K |
| `ffn_up` | **Q2_K** | Q3_K |

O único tensor realmente em 2 bits sob Q2_K é o `ffn_up`. Daí os 8% de diferença de
tamanho para o Q3_K_S, em vez dos 25% que a razão de bits por peso sugeriria. O 2-bit de
verdade é o `Q2_K_S`.

**3. A matriz de importância recupera a maior parte do dano em 2 bits.** Q2_K sem
calibração: F1 71,97 e CALAME 45,75. Com: 74,46 e 47,00 — cerca de 73% da perda de F1
recuperada. Em 2 bits a calibração deixa de ser opcional.

**4. Métrica de recusa nunca se lê sozinha.** O hard-OOS do Q2_K sem imatrix "sobe" para
61,33, o melhor da matriz. Não é ganho: é o modelo degradado caindo mais no "desconhecido".
A prova é que ao melhorar com imatrix, o hard-OOS **desaba** para 44,0. Qualquer métrica de
abstenção melhora quando o modelo piora.

**5. A fronteira do utilizável é IQ2_M, e ela é definida pela recusa.** O IQ2_XXS (80 MB)
ainda parece aceitável em intenção (84,33), mas a recusa cai de 98,0 para **86,0** — 12
pontos contra um IC de ±2,7pp, das poucas diferenças inequívocas de toda a matriz. Na
prática: 14% dos pedidos fora de escopo ele tenta executar em vez de dizer "não sei". Um
assistente que inventa ação é pior que um que recusa. **O modelo perde a noção do próprio
limite antes de perder a capacidade** — e é por isso que a recusa, não a acurácia, é o
critério de corte.

**6. O acerto conjunto é o número honesto.** Intenção certa E todos os campos certos:
**60,67%** no F16, contra 87,67 de intenção e 75,38 de F1 no mesmo conjunto. O F1 dá
crédito parcial por acertar dois campos de três; o usuário não — o alarme toca na hora
certa ou não toca. O gargalo do produto é extração de campos, não classificação.

## Escolha para distribuição

**Q4_K_M (168,8 MB) segue correto** — é o que está publicado. Não há razão medida para
trocar por Q5 ou acima.

Se o tamanho virar restrição, a escada é: Q3_K_S (118,8 MB, −30%, sem custo medido) →
IQ2_M com imatrix (100,1 MB, −41%, −2pp de recusa). Abaixo disso, não.

## Armadilhas de infraestrutura

- **`-DGGML_NATIVE=ON` ao compilar llama.cpp em container é bug latente.** O binário sai
  otimizado para o CPU do host que construiu a imagem; rodando em outro host morre com
  SIGILL — `rc=-4` e stdout/stderr vazios. Passa por sorte de alocação até não passar.
  Usar `NATIVE=OFF`.
- A versão que compilamos não escreve imatrix em `.gguf`; usar `.dat`. O `--in-file`
  combina `.dat` normalmente.

## Reprodução

| job | o que faz | saída |
|---|---|---|
| `modal_matriz_quant.py` | F16 + 6 níveis sem imatrix | `avaliacao/matriz_quantizacao.json` |
| `modal_quant_extremo.py` | níveis extremos sem imatrix | `avaliacao/quant_extremo.json` |
| `modal_quant_imatrix.py` | Q2/Q1 com imatrix | `avaliacao/quant_imatrix.json` |
| `modal_quant_imatrix_q3.py` | Q3/Q4 com imatrix + acerto conjunto | `avaliacao/quant_imatrix_q3.json` |
| `modal_imatrix_variantes.py` | 4 formas de calibrar a imatrix | `avaliacao/imatrix_variantes.json` |
