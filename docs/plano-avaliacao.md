# Plano de avaliação objetiva — Pocket-LM

> **STATUS: COMPLETO** (27/07/2026). Documento de especificação; nada foi executado
> nem instalado ao escrevê-lo. Todos os números de dataset foram verificados por API
> (`datasets-server.huggingface.co`) e todos os YAMLs de tarefa foram lidos do repositório.

Implementa a **DECISÃO Nº 1** de `docs/varredura-institucional.md` — *adotar uma métrica
objetiva antes de qualquer mudança de treino* — e a transforma em plano executável.

**Modelo alvo** (lido de `train_gpu.py`, `bpe_prepare.py`, `model.py`):
GPT decoder-only, `n_layer=8`, `n_head=8`, `n_embd=512`, `block_size=512`,
`vocab_size=16384` (BPE HF em `data_big/tokenizer.json`), weight tying
(`tok.weight = head.weight`) → **≈33,8M params**, dos quais 8,39M (24,7%) na embedding.

> **Correção (27/07/2026):** este documento continua sendo a especificação válida da
> avaliação — nada aqui foi refutado. Dois ajustes de número: **`bytes/token` é 3,948, não
> 19,82** (§2.1, o `all.bin` estava truncado) e o corpus passou de 27,3 M para **202,6 M
> tokens únicos**. E uma armadilha a acrescentar em §4.3: `val_loss` também não é comparável
> entre **`block_size` diferentes** — foi assim que os runs A (block 512) e B (block 1024)
> foram comparados errado. Ver `PLANO.md` §5, armadilhas 2 e 3.

**Bancada**: x86, **2 vCPU, 3,8 GB de RAM + 2 GB de swap, 6,5 GB de disco livre (82% usado)**,
`torch 2.13.0+cpu`, Python **3.14.4**, venv em `/opt/pocket-lm/venv` (já tem `datasets 5.0.0`,
`tokenizers 0.23.1`, `numpy 2.5.1`).

---

## Resumo em uma frase

**Nesta escala, toda métrica de acurácia em múltipla escolha está no chão e vai continuar
no chão** — o menor LM pt-BR publicado, o **Tucano-160m (162M params, 200B tokens)**, tira
**33,51 de F1-macro em ASSIN2 RTE**, que é *exatamente* o valor de uma predição constante.
O sinal utilizável em 33,8M está em **bits-por-byte**, em **CALAME-PT** (previsão da última
palavra, onde há 0→52 pontos de faixa dinâmica) e em **juiz-LLM com rubrica** à la TinyStories.

---

## 1. Quais tarefas rodar — e quais descartar

### 1.1 O que existe

**Grupo `portuguese_bench` do EleutherAI** (origem: IberoBench, COLING 2025, Baucells et al.)
- README: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/portuguese_bench/README.md
- YAMLs: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/portuguese_bench
- Conteúdo do `portuguese_bench.yaml` (verificado): `belebele_por_Latn`, `flores_pt`,
  `assin_paraphrase`, `assin_entailment`, `assin2_rte`, `assin2_sts`.

**Fork do Open PT LLM Leaderboard** (Eduardo Garcia)
- https://github.com/eduagarcia/lm-evaluation-harness-pt
- Diretório `lm_eval/tasks/portuguese/` (verificado): `assin2_rte`, `assin2_sts`, `bluex`,
  `enem`, `faquad_nli`, `hatebr_offensive`, `portuguese_hate_speech`, `tweetsentbr`,
  4 tarefas `sparrow_*`; e `oab_exams` em diretório próprio.
- Protocolo do leaderboard: few-shot 3 (ENEM/BlueX/OAB), 15 (ASSIN2 RTE, FaQuAD-NLI),
  10 (ASSIN2 STS), 25 (HateBR, hate speech, TweetSentBR).

**Não está em harness nenhum** (verificado: 0 ocorrências de "calame" no repo do EleutherAI;
ausente do diretório `portuguese/` do fork): **CALAME-PT** e **LAMBADA-PT**. O Tucano os
avaliou com scripts próprios (`lm-calame-pt-eval.py`, `lm-lambada-pt-eval.py`,
https://github.com/Nkluge-correa/Tucano/blob/main/evaluations/README.md).
**Isso é trabalho a fazer, e é o trabalho de maior retorno deste documento.**

### 1.2 Tabela de decisão

Tamanhos e distribuições verificados por `datasets-server.huggingface.co` (`/size`, `/statistics`).

| Task | Dataset HF | Itens | Tipo | Piso de acaso | Piso REAL (baseline trivial) | Teto medido em 160M–1,3B | Veredito |
|---|---|---|---|---|---|---|---|
| **CALAME-PT** | `NOVA-vision-language/calame-pt` (`all`, split `train`) | **2.076** | geração 0-shot, exact-match da última palavra | **~0%** (vocabulário aberto) | ~1–3% (chutar palavra frequente) | **52,3** (Tucano-160m) · 52,8 (GlórIA-1B3) | **PRINCIPAL** |
| **CALAME-PT-NLL** (variante nova, §2.3) | idem | 2.076 | `loglikelihood` da palavra ouro | — | **sem piso** (métrica contínua) | — | **PRINCIPAL** |
| **BPB / bits-per-byte** | texto congelado próprio | — | `loglikelihood_rolling` | — | **sem piso** | — | **PRINCIPAL** |
| **LAMBADA-PT** | `TucanoBR/lambada-pt` (5.153) | 5.153 | igual CALAME | ~0% | ~1% | **28,2** (Tucano-160m) | Secundária (§1.5) |
| `assin2_rte` | `nilc-nlp/assin2` (test) | **2.448** | MC 2 opções, F1-macro | 0,50 (aleatório) | **0,333** (constante) | **33,51** (Tucano-160m, 15-shot) = *chão* | **Só como null check** |
| `belebele_por_Latn` | `facebook/belebele` `por_Latn` (test) | **900** | MC 4 letras A/B/C/D, acc | 0,250 | **0,279** (chutar sempre "B") | — | **Só como null check** |
| `assin_entailment` | `nilc-nlp/assin` `full` (test) | 4.000 | MC 2 opções, acc crua | 0,50 | **0,735** (classe majoritária) | — | **DESCARTAR** (§1.4) |
| `assin_paraphrase` | idem | 4.000 | idem, **mesmos rótulos** | 0,50 | **0,735** | — | **DESCARTAR** (§1.4) |
| `assin2_sts` | `nilc-nlp/assin2` | 2.448 | `generate_until` + Pearson | — | indefinido | 11,02 (Tucano-160m) | **DESCARTAR** |
| `flores_pt` | `facebook/flores` | 1.012×N | tradução, BLEU/chrF | 0 | 0 | — | **DESCARTAR** |
| `enem`, `bluex`, `oab_exams` | fork pt | 1k–2k | MC 4–5 opções, 3-shot | 0,20–0,25 | ~0,25 | — | **DESCARTAR** (§1.6) |
| `faquad_nli`, `hatebr`, `hate_speech`, `tweetsentbr` | fork pt | 0,5k–2k | MC, few-shot 5–25 | 0,33–0,50 | classe majoritária | 43,97 (FaQuAD, Tucano-160m) | **DESCARTAR agora** (§1.7) |
| PoETa v2 (44 tarefas) | https://github.com/PoETaV2/PoETaV2 | — | vários | — | — | Falcon-3-**1B** = NPM 18,6 | **DESCARTAR** (já decidido, varredura §7.1) |

### 1.3 CALAME-PT é a tarefa principal — e por quê

- **Fonte:** dataset https://huggingface.co/datasets/NOVA-vision-language/calame-pt ·
  paper *GlórIA — A Generative and Open Large Language Model for Portuguese*,
  PROPOR 2024, https://arxiv.org/abs/2402.12969 (a seção que introduz o CALAME-PT).
- **O que é:** primeiro benchmark **zero-shot de modelagem de língua** em português.
  Dado um contexto, prever **a última palavra**. Inspirado no LAMBADA.
  **2.076 exemplos**: 406 escritos por falantes nativos + 1.670 gerados por GPT-3.5 com
  revisão humana. Campos: `id`, `sentence` (26–1.130 caracteres), `last_word` (2–17).
- **Protocolo exato** usado pelo Tucano (lido de `lm-calame-pt-eval.py`):
  geração **gulosa** (`do_sample=False`) com `max_new_tokens=10`; extrai a **primeira
  palavra** após o prompt (split por espaço); limpa pontuação com `re.sub(r'[.,?!]', '', w).strip()`;
  **exact match** contra `last_word`; acurácia = acertos/total.

**As quatro propriedades que faltam em todas as tarefas de múltipla escolha:**

1. **Piso ~zero.** Vocabulário aberto: não há alternativa a chutar, não há classe majoritária
   a explorar. Sair de 3% para 12% é progresso real e inequívoco.
2. **Faixa dinâmica enorme e comprovada.** Tucano-160m 52,31 · TeenyTinyLlama-160m 46,72 ·
   Xglm-564m 50,58 · mGPT-1b3 47,14 · GlórIA-1B3 52,79 (tabela do model card
   https://huggingface.co/TucanoBR/Tucano-160m). O benchmark **separa modelos do mesmo
   tamanho** (5,6 pontos entre Tucano-160m e TeenyTinyLlama-160m) — é sensível a receita,
   não só a escala. É exatamente o que o Pocket-LM precisa medir.
3. **É modelagem de língua pura**, o objetivo de treino do modelo. Não exige seguir instrução,
   nem associar letras a alternativas, nem few-shot. É a única família de tarefa em que 33,8M
   pode legitimamente competir.
4. **Cabe na janela.** `sentence` até 1.130 caracteres ≈ 285 tokens BPE, bem dentro de 512.

**Caveat de dialeto, com a evidência.** CALAME-PT vem da equipe do GlórIA (NOVA Lisboa) e é
**português europeu**. Isso preocuparia num modelo pt-BR, exceto que os números desmentem a
preocupação: **Tucano (pt-BR) tira 52,31 e GlórIA (pt-PT) tira 52,79** — 0,5 ponto de
diferença. O viés de dialeto no CALAME-PT é pequeno o bastante para não invalidar a métrica.
Ainda assim: registrar o caveat no README e, se um dia houver folga, construir um CALAME-BR
de 300–500 itens a partir do corpus próprio (§6, marco 4).

**Expectativa honesta para 33,8M:** primeira medição provavelmente **2–8%**, com o modelo
atual (`night_report.json`: amostra ainda ilegível) possivelmente **<1%**. Um modelo de 33,8M
bem treinado em ~2B tokens pt deveria alcançar **10–25%**. Chegar aos 52% do Tucano-160m
**não é o alvo** — o Tucano tem 4,8× os parâmetros e ~100× os tokens.

### 1.4 `assin_entailment` e `assin_paraphrase` são armadilhas — correção da recomendação original

`varredura-institucional.md` §7.3 recomendava rodar as duas. **Ambas devem sair.** Motivos
verificados, não teóricos:

Distribuição de `entailment_judgment` no split `test` de `nilc-nlp/assin` config `full`:
**`NONE` 2.939 · `ENTAILMENT` 822 · `PARAPHRASE` 239** (de 4.000).

Os dois YAMLs usam **o mesmo alvo binário**: `{{0 if entailment_judgment == 0 else 1}}`.
Só muda a palavra de contraste no `doc_to_choice`:
- `assin_entailment`: `premise + ", certo? Também, " + hypothesis` vs `... "Sim, " ...`
- `assin_paraphrase`: `premise + ", certo? Não, " + hypothesis` vs `... "Sim, " ...`

Portanto:
- **Não são duas medições independentes.** Rótulos idênticos. Reportar as duas dá falsa
  corroboração.
- A classe 0 vale **73,5%** e a métrica é **`acc` crua** (sem correção de desbalanceamento).
  Preferir sempre `"Não,"`/`"Também,"` marca **0,735** sem saber nada; preferir sempre
  `"Sim,"` marca **0,265**.
- O que essas tarefas medem num modelo de 33,8M é o **viés do LM entre três palavras
  funcionais** (`Sim`/`Não`/`Também`) depois de `", certo? "`. Isso oscila entre checkpoints
  por razões que não têm nada a ver com entailment. **É ruído com aparência de sinal — pior
  que não medir.**

Se forem rodadas assim mesmo (custam pouco), reportar **sempre ao lado de `piso_majoritario:
0.735`** e tratar qualquer valor em [0,70; 0,75] como indistinguível de constante.

### 1.5 LAMBADA-PT — secundária, e por quê

- `TucanoBR/lambada-pt`: **5.153 exemplos**, tradução automática do split de teste do LAMBADA
  pré-processado pela OpenAI. Mesmo formato (`sentence`, `last_word`).
- **Vale:** mesmo regime do CALAME (piso ~zero, LM puro), e discrimina
  (Tucano-160m 28,16 vs TeenyTinyLlama-160m 20,98).
- **Não é principal por três razões:** (a) é **tradução automática**, carrega *translationese*
  e o domínio é ficção anglófona, distante da mistura pt-BR do projeto; (b) contextos mais
  longos (178–1.010 caracteres) e **2,5× mais itens** → 2,5× o custo de CPU; (c) o CALAME-PT
  já dá o mesmo tipo de sinal, mais barato e em português nativo.
- **Uso:** rodar por marco, não por checkpoint, e de preferência num **subconjunto fixo de
  2.000 itens** (semente registrada) para cortar o custo.

### 1.6 ENEM / BlueX / OAB estão fora

Exames humanos de nível médio e superior, 4–5 alternativas, few-shot 3–5, enunciados longos
(muitos passam de 512 tokens e **truncariam contra o `block_size`**, tornando o número
incomparável com qualquer coisa). O PoETa v2 mede **Falcon-3-1B com NPM 18,6**
(https://arxiv.org/html/2511.17808) — um modelo **30× maior** já perto do chão. Rodar isso em
33,8M produz ruído garantido, ao maior custo de CPU da lista. Descartar sem hesitação.

### 1.7 `faquad_nli`, `hatebr`, `hate_speech`, `tweetsentbr` — descartar agora

Tecnicamente são do tipo certo (rótulo curto, F1-macro no fork). Dois bloqueios hoje:
- **Few-shot 5–25** no protocolo do leaderboard. Com `block_size=512`, 25 exemplos de HateBR
  **não cabem**; o harness trunca e o número deixa de ser comparável com o leaderboard —
  que era a única razão de adotar aquele protocolo.
- São **classificação de domínio** (ódio, sentimento, FAQ), não modelagem de língua. Sem SFT,
  um LM base de 33,8M não tem por que se destacar. Tucano-160m tira 43,97 em FaQuAD-NLI, que
  também está perto do piso de classe majoritária.

Reabrir **só** numa eventual fase de SFT em domínio estreito (Trilha B do `estado-da-arte.md`
§4.3), rodando **0-shot**, com F1-macro, comparado contra classe majoritária — não contra o
leaderboard.

### 1.8 Múltipla escolha: manter dois *null checks*, e só isso

Manter `assin2_rte` e `belebele_por_Latn` — não como métrica de progresso, mas como
**verificação de que nada emergiu**. Custam pouco, e a informação "continua no chão" tem
valor, inclusive de honestidade metodológica.

**`assin2_rte`** — o melhor desenho de MC disponível em português, e mesmo assim insuficiente:
- Rótulo **perfeitamente balanceado** no test: `{"NONE": 1224, "ENTAILMENT": 1224}` (verificado).
- Métrica é **macro-F1** (`aggregation: !function assin2_utils.macro_f1`), que pune predição
  constante: constante → **0,333**; aleatório balanceado → ~0,50.
- É **par mínimo**, não símbolo abstrato: compara a verossimilhança de
  `premise + ", certo? Não, " + hypothesis` contra `premise + ", certo? Sim, " + hypothesis`.
- Cabe folgado: `premise` média 55 caracteres, `hypothesis` 52 (máx. 159) → prompt ≈ 40 tokens.
- **E ainda assim: Tucano-160m tira 33,51 com 15-shot**, ou seja, exatamente o valor da
  predição constante. Se um modelo 4,8× maior treinado em 200B tokens colapsa nesta tarefa,
  o Pocket-LM vai colapsar. **Esperar 0,333 e não se assustar.**
- Rodar **0-shot** (15-shot não cabe em 512 tokens de contexto). Isso torna o número
  não-comparável ao leaderboard, o que é irrelevante — o leaderboard está fora de alcance.

**`belebele_por_Latn`** — manter com ressalva ainda maior:
- 900 itens, 4 alternativas, e o template do harness responde com **uma letra**:
  `doc_to_choice: ["A", "B", "C", "D"]`
  (https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/belebele/_default_template_yaml).
  É tarefa de **símbolo**, não de língua — capacidade que emerge com instruction tuning ou
  escala, não em 33,8M base.
- Como as 4 opções têm **1 byte cada**, **`acc_norm` ≡ `acc`**; reportar as duas é redundante.
- Comprimento: `flores_passage` média **521 caracteres** (máx. 1.480), `question` 78,
  4 alternativas ~110 → prompt médio ≈ 730 caracteres ≈ **190–220 tokens**, cauda até
  **~500 tokens** → **trunca na cauda** contra `block_size=512`.
- Distribuição da resposta certa: `{1: 206, 2: 251, 3: 246, 4: 197}` → chutar sempre "B" dá
  **27,9%**, não 25%.
- **Erro-padrão com n=900 em p=0,25: 1,44 pp → IC 95% de ±2,8 pp.** Qualquer diferença entre
  checkpoints menor que ~3 pp **não é observável**. Isso sozinho o desqualifica como métrica
  de rodada a rodada.
- **Uso:** 0-shot, uma vez por marco. Se cair em [0,22; 0,31], a conclusão é "sem sinal".

---

## 2. Métricas que funcionam abaixo do regime de múltipla escolha

Esta seção é onde está o sinal real. §1 responde "aprendeu alguma coisa?"; §2 responde
"melhorou quanto?".

### 2.1 BPB / bits-per-byte — a métrica invariante a tokenizer (prioridade máxima)

**Definição operacional**, a mesma do harness:

```
BPB = NLL_total(D) / (ln 2 × bytes_utf8(D))
```

`NLL_total` = soma de `-log p(token_i | contexto)` sobre **todos** os tokens do texto de
validação; `bytes_utf8(D)` = tamanho **do texto cru em bytes UTF-8, antes de tokenizar**.
É exatamente o que `lm-evaluation-harness` reporta em tarefas `loglikelihood_rolling`
(o `wikitext.yaml` lista `word_perplexity`, `byte_perplexity`, `bits_per_byte`):
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/wikitext.yaml

**Por que resolve o problema de `varredura-institucional.md` §10.2.**
`val_loss` está em **nats por token**. Se o vocabulário cair de 16k para 8k (Decisão 3 da
varredura), cada frase passa a ter **mais tokens**, cada um individualmente mais fácil de
prever → `val_loss` **cai** sem que o modelo tenha melhorado em nada, e **toda a série
histórica do projeto quebra silenciosamente**. BPB divide pelo **byte do texto original**,
que não muda quando o tokenizer muda. É a única forma de responder "8k é melhor que 16k?"
com um número.

**Conversão a partir do que o projeto já calcula:**

```
BPB = val_loss_nats_por_token × (tokens / bytes) / ln 2
```

O fator `tokens/bytes` é a **taxa de compressão do tokenizer** e tem de ser medido **no mesmo
texto**. Com o corpus atual: `data_big/corpus.txt` = 800.053.953 bytes →
`data_big/all.bin` = 80.749.200 bytes uint16 = **40.374.600 tokens** →
**19,82 bytes/token**. **Gravar esse número em todo resultado**: sem ele o BPB não é
reconstruível nem auditável.

> **Correção (27/07/2026): 19,82 bytes/token era um artefato de arquivo truncado.** O
> `all.bin` tinha sido interrompido por OOM na tokenização (22,3% do corpus). Foi refeito:
> `data_big/train.bin` (192.499.342 tokens) + `data_big/val.bin` (10.131.545) =
> **202.630.887 tokens** para os mesmos 800.053.953 bytes = **3,948 bytes/token**.
> Substituir 19,82 por 3,948 aqui e no exemplo de JSON de §5.4 (`bytes_por_token_val`).
> **Ressalva que continua valendo, e agora mais:** `analise-runs.md` §0.2a mediu 4,285
> bytes/token numa amostra de 3 MB — 8,5% de diferença, e o BPB é linear nisso. O número que
> vale é o medido **no `val_frozen.txt`**, nunca reaproveitado entre arquivos.
> Ver `PLANO.md` §1.4 e §7, contradição 3.

> **BPC vs BPB.** Bits por *caractere* e bits por *byte* coincidem em ASCII e divergem em
> pt-BR (acentos ocupam 2 bytes em UTF-8; ~3–4% dos caracteres). **Padronizar em BPB**, que
> é o que o harness calcula; mencionar "BPC" só em prosa.

**Faixa esperada.** Um LM pt-BR de 33,8M razoavelmente treinado deve ficar em
**~1,3–1,9 BPB**. Abaixo de 1,0 nesta escala é improvável e sugere contaminação
(texto de validação visto no treino). Acima de 2,5 é modelo ainda cru.

### 2.2 Perplexidade em conjunto fixo — três correções no que já existe

`eval.py` já implementa perplexidade em `val.bin` com `estimate_loss()`, re-semeando o RNG do
numpy para que todo checkpoint veja os mesmos batches. **A decisão de re-semear está certa
e deve ser preservada.** Três correções:

1. **Congelar o conjunto como TEXTO, não como `.bin`.** Hoje `data/val.bin` é uma sequência
   de ids do tokenizer atual; se o tokenizer mudar, o arquivo perde o sentido e a série
   histórica morre no mesmo instante em que o BPB deveria salvá-la. Criar
   **`eval/val_frozen.txt`** — 2 MB de texto pt-BR retido do corpus, nunca visto no treino,
   com **SHA-256 registrado** — e re-tokenizar sob demanda com o tokenizer vigente.
2. **Trocar amostragem aleatória por varredura determinística completa.** Batches aleatórios
   introduzem variância que compete com o efeito que se quer medir. Varrer o `val_frozen.txt`
   inteiro em janelas dá o mesmo número em toda execução, com desvio zero.
3. **Fixar o `stride`.** Janelas não sobrepostas penalizam os primeiros tokens de cada janela
   (que não têm contexto) — material com `block_size=512`. **Padronizar `stride=256`** (meia
   janela) e **registrar o stride no JSON**: mudar o stride muda o número, e é uma forma
   silenciosa de "melhorar" sem melhorar.

### 2.3 CALAME-PT-NLL — a variante sem piso (contribuição própria, e importante)

A acurácia do CALAME-PT vai começar perto de zero, e **acurácia perto de zero é uma métrica
de baixa resolução**: entre 1,2% e 1,8% não há como distinguir sinal de ruído com n=2.076
(erro-padrão em p=0,015 é 0,27 pp).

**Solução:** rodar o *mesmo* dataset como `loglikelihood`, medindo o **NLL médio da palavra
ouro** dado o contexto:

```
calame_pt_lastword_nll = mean_i [ -log p(last_word_i | sentence_i) ]   # nats, somado sobre os tokens da palavra
calame_pt_lastword_bpb = mean_i [ NLL_i / (ln2 × bytes_utf8(last_word_i)) ]
```

Propriedades:
- **Sem piso e contínua.** Move-se a cada rodada de treino, mesmo quando a acurácia é 0.
- **Mede exatamente a mesma competência** que a acurácia mede, só que sem discretizar.
- **Muito mais barata**: 1 forward por item, sem geração autoregressiva (§3).
- **Comparável entre tokenizers** na variante BPB (divide por bytes da palavra).
- Correlaciona monotonicamente com a acurácia quando a acurácia sai do chão, então a série
  histórica não quebra na transição.

**Reportar as duas juntas.** A acurácia é o número comunicável (comparável ao Tucano);
o NLL é o número que guia o treino.

Aplicar a mesma variante ao LAMBADA-PT (`lambada_pt_lastword_nll`).

### 2.4 A linha TinyStories: juiz-LLM com rubrica

**Fonte primária:** *TinyStories: How Small Can Language Models Be and Still Speak Coherent
English?*, Eldan & Li (Microsoft Research) — https://arxiv.org/abs/2305.07759
Versão legível: https://ar5iv.labs.arxiv.org/html/2305.07759

**Por que é a referência certa:** é o único trabalho de peso que avalia modelos de **1M a 80M
parâmetros** — a faixa exata do Pocket-LM — e que constata que benchmarks padrão são inúteis
nessa escala, propondo em troca avaliação por juiz com rubrica.

**Protocolo original (verificado no paper):**
- ~**50 prompts** de início de história, escritos à mão.
- Para cada prompt, o modelo gera **10 completions a `temperature=1.0`**.
- O juiz (GPT-4) recebe início + completion e é instruído a avaliar **como se fosse redação de
  aluno**: primeiro uma **avaliação verbal**, depois **notas numéricas de 0 a 10** por
  categoria.
- Categorias: **Grammar**, **Creativity**, **Consistency with the beginning of the story**
  (a variante TinyStories-Instruct acrescenta **Plot**).
- Além das notas, o juiz dá um **palpite da idade do aluno** numa escala fechada:
  A (≤3), B (4–5), C (6–7), D (8–9), E (10–12), F (13–16).
- Notas médias sobre as 10 amostras × 50 prompts.
- **Achado que orienta a expectativa aqui:** gramática é dominada por modelos pequenos e
  **satura cedo**; **consistência e criatividade só emergem com mais tamanho**; a dimensão de
  embedding importa mais para conhecimento factual e o **número de camadas** importa mais
  para rastrear contexto.

#### 2.4.1 Rubrica concreta para o Pocket-LM (pt-BR)

Adaptações necessárias: o corpus do Pocket-LM não é infantil, então "criatividade" vira
**fluência/naturalidade**; e a escala de idade vira uma **âncora de nível** de baixa resolução,
que é justamente o que se quer nesta escala.

**Prompts fixos:** 40 prompts pt-BR congelados em `eval/prompts_juiz.json`, em 4 famílias de
10, para separar capacidades como o TinyStories separa:

| Família | Exemplo de prompt | O que isola |
|---|---|---|
| Narrativa | `"Era uma vez uma menina que morava perto do rio. Um dia ela"` | coerência local |
| Enciclopédica | `"A cidade de Salvador foi fundada em"` | conhecimento factual (n_embd) |
| Factual curta | `"A capital do estado de Minas Gerais é"` | recuperação factual pontual |
| Consistência | prompt que introduz nome próprio + atributo na 1ª frase e interrompe a 2ª onde o atributo deve ser retomado | rastreio de contexto (n_layer) |

**Geração:** 5 completions por prompt, `temperature=1.0`, **`top_k` desligado** (top-k mascara
justamente os erros que se quer medir), `max_new_tokens=120`, seed fixa.
→ **200 amostras por checkpoint**.

**Rubrica — 0 a 10 por eixo, inteiros:**

| Eixo | 0–2 | 3–5 | 6–8 | 9–10 |
|---|---|---|---|---|
| **Gramática** | não forma palavras de português; sequências de letras sem sentido | palavras reais, mas concordância e flexão quebradas; frases não fecham | frases bem formadas, erros ocasionais de concordância/regência | pt-BR correto, pontuação e acentuação coerentes |
| **Coerência local** | frases consecutivas sem relação | relação vaga; sujeito troca no meio | encadeamento claro por 2–3 frases | encadeamento claro no trecho todo |
| **Consistência com o prompt** | ignora o prompt / muda de assunto na 1ª frase | mantém o tema mas contradiz um fato dado | mantém tema e fatos; retoma nome/atributo | continua como se fosse o mesmo autor |
| **Fluência/naturalidade** | repete o mesmo token ou frase em loop | repetição perceptível; texto "genérico" | varia estrutura, sem loop | passaria por trecho real de pt-BR |
| **Factualidade** (só família enciclopédica) | inventa entidade inexistente | entidade real, fato errado | fato aproximadamente correto | fato correto |

**Âncora de nível** (substitui a escala de idade), pedida junto das notas:

`N0` ruído · `N1` palavras soltas · `N2` frases isoladas corretas · `N3` parágrafo com tema
único · `N4` texto com progressão.
**Meta realista para 33,8M: sair de N0/N1 e chegar a N2→N3.**

**Regras que transformam o juiz em medição, e não em impressão:**
1. **Juiz fixo e versionado.** Gravar o model id exato no JSON. Trocar o juiz invalida a
   comparação com resultados anteriores tanto quanto trocar o tokenizer.
2. **Cego e embaralhado.** O juiz recebe amostras de **dois checkpoints misturadas e
   anônimas**, em ordem aleatória. Nunca "avalie o checkpoint novo".
3. **Avaliação verbal antes da nota** (como no paper) — reduz nota reflexa.
4. **Âncora calibrada por lote:** incluir 3 amostras de referência fixas com nota previamente
   acordada (uma claramente N0, uma N2, e um trecho de texto humano real). **Se o juiz errar
   a âncora por mais de 2 pontos, o lote inteiro é descartado e refeito.**
5. **Reportar média e desvio por eixo**, nunca uma nota única agregada. Gramática satura
   antes; agregar esconde exatamente o que interessa.
6. **Custo:** ~200 amostras × ~150 tokens + rubrica ≈ **60–80k tokens de entrada por
   checkpoint**. É a métrica mais cara em dinheiro. Usar **por marco**, nunca por checkpoint.

### 2.5 Sanity checks quase gratuitos — rodar sempre, antes de tudo

Detectam colapso e regressão a custo próximo de zero. Teriam diagnosticado o estado atual
(`night_report.json`: `val_loss` caindo 3,7495 → 3,2455 com amostra ainda ilegível — prova
direta de que `val_loss` não distingue "melhorou" de "mudou"):

| Métrica | Definição | Alerta |
|---|---|---|
| `rep_token_rate` | fração de tokens iguais ao anterior numa geração de 200 tokens | > 0,15 = loop |
| `distinct_2` | bigramas únicos ÷ bigramas totais na geração | queda brusca = colapso |
| `palavras_no_lexico` | fração de palavras geradas presentes numa whitelist de ~200k formas extraída de `data_big/corpus.txt` | **é o proxy barato de "gramática" do §2.4** |
| `entropia_media_bits` | entropia média da distribuição de saída nos prompts fixos | cai bruscamente no colapso |

`palavras_no_lexico` merece destaque: em 33,8M, **sair de 40% para 70% de palavras reais é o
primeiro progresso visível do projeto**, é medível sem juiz nenhum, e custa segundos.

---

## 3. Custo real na bancada x86

### 3.1 Modelo de custo

FLOPs por posição de token no forward ≈ `2N + 2·n_layer·T·n_embd`
= `2×33,8M + 2×8×512×512` ≈ **72 MFLOP/posição** (T=512).

2 vCPU com `torch 2.13.0+cpu` (oneDNN, fp32): throughput sustentado realista de GEMM
**15–30 GFLOP/s**. Adotando **20 GFLOP/s** como número de planejamento →
**≈ 275 posições de token por segundo**. Faixa plausível: 150–500 pos/s.

> **Este número é uma estimativa e precisa ser substituído pela medição da primeira execução.**
> Gravar `custo.segundos_cpu` e `custo.pico_rss_mb` no JSON desde o primeiro run (§5.4) serve
> exatamente para isso.

### 3.2 Custo por tarefa (planejando 275 pos/s)

| Tarefa | Requests | Posições de token | Tempo estimado | Pico de RAM |
|---|---|---|---|---|
| **BPB em `val_frozen.txt` 2 MB** | 101k tokens, stride 256 | ~202k (2× por sobreposição) | **~12 min** | ~1,0 GB (batch 4) |
| **BPB em 500 KB (modo rápido)** | 25k tokens | ~50k | **~3 min** | ~1,0 GB |
| **`calame_pt_lastword_nll`** | 2.076 × 1 | ~266k (pad 128) | **~16 min** | ~0,9 GB (batch 16) |
| **`calame_pt_acc` COM KV cache** (`GPTExport`) | 2.076 × (prefill 63 + 4 decode) | ~139k | **~8 min** | ~0,9 GB |
| **`calame_pt_acc` SEM KV cache** (`model.generate` atual) | idem, prefixo refeito a cada passo | ~545k + overhead batch=1 | **1,5–3 h** | ~0,9 GB |
| **`assin2_rte` 0-shot** | 2.448 × 2 = 4.896 | ~313k (pad 64) | **~19 min** | ~0,8 GB (batch 32) |
| **`belebele_por_Latn` 0-shot** | 900 × 4 = 3.600 | ~922k (pad 256) | **~56 min** | ~1,3 GB (batch 4) |
| **`lambada_pt_nll`** (5.153 itens) | 5.153 | ~1,32M (pad 256) | **~80 min** | ~1,3 GB |
| `lambada_pt_nll` (subconjunto 2.000) | 2.000 | ~512k | ~31 min | ~1,3 GB |
| **Amostragem p/ juiz** (200×120 tok, KV cache) | 200 | ~27k | **~2 min** | ~0,6 GB |
| Sanity checks (§2.5) | — | ~5k | **<1 min** | ~0,6 GB |

**Achado operacional decisivo:** `model.py::generate` (linhas 86–98) **refaz o forward do
prefixo inteiro a cada token gerado** (`cond = idx[:, -block_size:]`; forward completo).
Para gerar 4 tokens sobre 63 de contexto isso é ~4× o trabalho necessário, e com `batch=1`
o overhead de despacho do PyTorch domina: na prática **10–20× mais lento**. O projeto **já
tem a solução pronta**: `model_export.py` define `GPTExport` com `prefill()` e
`decode_step()` e cache de KV, e `quantize.py::generate_export` já a usa. **Toda avaliação
que gera texto deve usar `GPTExport`, não `GPT.generate`.** Custo de adotar: zero — o código
existe e já foi validado por `_run_equivalence()`.

### 3.3 Orçamento de RAM — o gargalo é o tensor de logits

Com 3,8 GB totais e **~2,2 GB disponíveis**:

| Componente | RAM |
|---|---|
| import de `torch` (RSS) | 350–500 MB |
| import de `lm_eval` + `transformers` + `datasets` | +400–600 MB |
| pesos do modelo fp32 (33,8M × 4 B) | 135 MB |
| datasets em memória (todos juntos) | < 10 MB |
| **tensor de logits `B × T × 16384 × 4 B`** | **B=4,T=512 → 134 MB · B=8 → 268 MB · B=16 → 537 MB** |
| cópia do `log_softmax` | **mesmo tamanho de novo** |

**O logits é o gargalo, não os pesos.** `vocab_size=16384` × `block_size=512` × fp32 = 33,5 MB
**por sequência**, e o `log_softmax` duplica. Regras:

- **Nunca usar `--batch_size auto`.** O auto-probe do harness escala até estourar, e aqui
  estourar significa OOM global (foi assim que a sessão anterior morreu).
- **Batch fixo por tarefa:** `4` para tarefas com T≈512 (BPB, Belebele), `16` para T≤128
  (ASSIN2, CALAME).
- Na implementação de `_loglikelihood_tokens`, **fazer o `log_softmax` e o `gather` por
  amostra, em laço sobre o batch**, e descartar os logits imediatamente — nunca materializar
  `B×T×V` duas vezes.
- **Rodar sempre sob `heavy.sh`** (`MEMCAP=2200M`, `SWAPCAP=1G`). Isso dá as duas garantias
  que o próprio script documenta: o `flock` impede sobreposição com o treino do cron, e o
  `MemoryMax` do cgroup faz um estouro matar só a avaliação, não a sessão inteira.
  Comando: `MEMCAP=2200M WAIT=1800 ./heavy.sh venv/bin/python -m evalx.run_eval ...`

**Total estimado por execução completa: ~1,5–1,6 GB de pico.** Cabe nos 2,2 GB — **mas só se
o treino não estiver rodando**, e é o `flock` do `heavy.sh` que garante isso.

### 3.4 Disco

- Datasets (parquet): ASSIN 2,7 MB · ASSIN2 0,57 MB · Belebele `por_Latn` 0,32 MB ·
  CALAME-PT ~1 MB · LAMBADA-PT 1,25 MB. **Total < 10 MB**; com o cache Arrow do
  `datasets`, ~50 MB.
- `val_frozen.txt`: 2 MB. Whitelist de léxico: ~3 MB.
- Instalação do harness (wheels verificados no PyPI em 27/07/2026, com suporte a **Python
  3.14** — ponto de risco checado): `lm_eval 0.4.12` (pure-python, 8,9 MB) ·
  `transformers 5.14.1` (pure-python, 11,6 MB) · `scipy 1.18.0` (**tem wheels cp314**,
  37,5 MB) · `scikit-learn 1.9.0` (**tem wheels cp314**, 9,3 MB) · `sacrebleu 2.6.0` ·
  `evaluate 0.4.6` · `pytablewriter 1.2.1` · `sqlitedict 2.1.0` (**sem wheel — compila do
  sdist**, único risco real em 3.14) · `jsonlines`.
  **Instalado em disco: ~400–600 MB.**
- Livres hoje: **6,5 GB (82% usado)**. Cabe, mas o `data_big/corpus.txt` sozinho ocupa 800 MB
  e `data/corpus.txt` outros 47 MB — não deixar a margem cair abaixo de ~3 GB.

### 3.5 O que NÃO cabe aqui e teria que ir para GPU alugada

| Item | Por quê | Onde vai |
|---|---|---|
| Baselines ≥ 600M params em fp32 | Tucano-630m = 2,5 GB só de pesos; Tucano-1b1 = 4,4 GB. Não cabe em 2,2 GB | GPU alugada, ou GGUF/`llama.cpp` quantizado |
| Baseline **Tucano-160m** (650 MB fp32) | cabe, **tight** — ~5× mais lento que o Pocket-LM → BPB em 2 MB ≈ **1–1,5 h** | **cabe no x86, à noite, sob `heavy.sh`** |
| Baseline **Gemma-3-270M** (1,08 GB fp32) | cabe sob o teto de 2,2 GB, mas sem folga para logits de vocab 256k | preferir GPU; se no x86, batch 1–2 |
| PoETa v2 completo (44 tarefas) | horas a dias de CPU, e o resultado é ruído nesta escala | não rodar em lugar nenhum |
| Juiz-LLM local | um juiz confiável tem ≥7B params | **API** (é o caminho certo, e é barato: ~80k tokens/checkpoint) |
| Protocolos few-shot 15/25 | bloqueados por `block_size=512`, não por RAM | só depois que o contexto crescer |
| `assin2_sts`, `flores_pt` | exigem geração livre parseável / tradução | não rodar |

---

## 4. Comparação honesta

### 4.1 Baselines — números publicados, todos verificados no model card do Tucano-160m

Fonte: https://huggingface.co/TucanoBR/Tucano-160m ·
paper *Tucano: Advancing neural text generation for Portuguese*,
https://www.sciencedirect.com/science/article/pii/S2666389925001734 ·
avaliações: https://github.com/Nkluge-correa/Tucano/blob/main/evaluations/README.md

| Modelo | Params | Tokens de treino | CALAME-PT | LAMBADA-PT | HellaSwag-PT | ARC-PT | ASSIN2 RTE (F1) |
|---|---|---|---|---|---|---|---|
| **Pocket-LM (alvo)** | **33,8M** | **~1–2B** | **? (esperado 10–25)** | ? | — | — | **? (esperado ~33,3 = chão)** |
| TeenyTinyLlama-160m | 162M | ~6B | 46,72 | 20,98 | 29,29 | 26,15 | — |
| **Tucano-160m** | 162M | **200B** | **52,31** | 28,16 | 33,07 | 27,01 | **33,51** (chão) |
| Xglm-564m | 564M | multilíngue | 50,58 | 27,42 | 34,64 | 25,56 | — |
| mGPT-1b3 | 1,3B | multilíngue | 47,14 | 29,92 | 26,37 | 23,81 | — |
| GlórIA-1B3 (pt-PT) | 1,3B | ~35B | 52,79 | 27,71 | 37,04 | 26,67 | — |
| Tucano-630m | 630M | 200B | — | — | — | — | 57,79 |
| Tucano-1b1 | 1,1B | 200B | — | — | — | — | 60,82 |

**Duas leituras que orientam todo o resto do plano:**

1. **CALAME-PT satura perto de 52 mesmo em 1,3B** e ainda assim **separa dois modelos de
   162M por 5,6 pontos** (Tucano vs TeenyTinyLlama). É sensível à **receita**, não só à
   escala. É o benchmark certo.
2. **ASSIN2 RTE colapsa em 162M (33,51 = predição constante) e só descola em 630M (57,79).**
   Existe um limiar de emergência entre 162M e 630M para múltipla escolha em português.
   **O Pocket-LM está 5× abaixo do lado errado desse limiar.** Isso não é pessimismo — é o
   dado que justifica não gastar tempo com MC.

### 4.2 Baselines escolhidos para o Pocket-LM

- **Referência de teto (não competidor):** **Tucano-160m** — Apache 2.0, pt-BR nativo,
  a menor referência publicada, e a única que roda na bancada. Rodar **o mesmo código de
  avaliação** nele para obter BPB no `val_frozen.txt`, CALAME-PT acc e NLL.
- **Referência de piso:** o **próprio checkpoint char-level de 3,06M** (`night_report.json`,
  `data/vocab.json`) e um **checkpoint aleatório não treinado** do mesmo tamanho.
  O modelo aleatório é o baseline mais importante e o mais esquecido: dá o valor exato de
  "zero conhecimento" em BPB e em CALAME-NLL, para o tokenizer vigente.
- **Baseline não-neural:** **n-grama de ordem 5 com backoff Kneser-Ney** treinado no mesmo
  corpus. É barato (KenLM ou implementação própria), roda em minutos, e responde a pergunta
  que ninguém quer fazer: *o transformer de 33,8M está batendo um n-grama?* Se não estiver, o
  problema não é a receita de treino.
- **Fora do escopo por enquanto:** Tucano-630m/1b1, GlórIA, Sabiá, Gemma-3-270M — GPU ou
  quantizados, e nenhum deles muda uma decisão hoje.

### 4.3 Como evitar comparação injusta — as sete regras

1. **Nunca comparar `val_loss` entre vocabulários diferentes.** Nats/token não é uma
   grandeza física; é uma grandeza por unidade de tokenização. Mudar 16k→8k faz `val_loss`
   cair sozinho. **Qualquer tabela que compare rodadas com tokenizers diferentes só pode ter
   BPB.** (Este é o item 2 de `varredura-institucional.md` §10, e é a razão de existir deste
   documento.)
2. **BPB é a única métrica cross-tokenizer, e só se o denominador for idêntico.** O texto de
   validação tem de ser **o mesmo arquivo, byte a byte**, verificado por SHA-256 gravado no
   JSON. Comparar BPB de textos diferentes é tão inválido quanto comparar `val_loss` de
   vocabulários diferentes — e é um erro muito mais fácil de cometer sem perceber.
3. **Reportar params E tokens de treino, sempre juntos.** Pocket-LM 33,8M / ~1–2B tokens vs
   Tucano-160m 162M / 200B tokens = **4,8× params e ~100–200× tokens**. Uma tabela que mostre
   só os params sugere uma comparação que não existe.
4. **Mesmo código, mesmo prompt, mesmo stride, mesma normalização, para todos.** Nunca citar
   número de leaderboard ao lado de número medido em casa. O Tucano mediu ASSIN2 RTE com
   **15-shot**; a bancada só consegue **0-shot** (block 512). São grandezas diferentes com o
   mesmo nome.
5. **Registrar o `stride` e a normalização de palavra do CALAME.** `re.sub(r'[.,?!]', '', w).strip()`
   e nada mais — casefold, remoção de acento ou stemming inflacionam a acurácia e quebram a
   comparação com o Tucano.
6. **Contaminação.** `data_big/corpus.txt` provavelmente compartilha fontes com o GigaVerbo
   (que treinou o Tucano). O `val_frozen.txt` tem de vir de material com proveniência e data
   conhecidas, e **o mesmo texto pode ter sido visto pelo baseline**. Consequência prática:
   um BPB do Tucano melhor que o do Pocket-LM pode ser memorização. Declarar isso na tabela
   em vez de silenciar. O CALAME-PT é mais seguro nesse aspecto (curado e parcialmente
   gerado por GPT-3.5, não raspado da web).
7. **Comparar o quantizado sempre contra o próprio fp32 no mesmo texto.** Para a Decisão 2 da
   varredura (QAT no int4), a métrica é **Δ BPB entre fp32 e int4 no `val_frozen.txt`** — que
   é exatamente o formato do resultado do Google no QAT do Gemma (queda de perplexidade
   reduzida em 54%, medida com `llama.cpp perplexity`). Isso **plugga direto** em
   `quantize.py`, que já constrói as variantes int8/int4 e já tem `estimate_loss_export`.

---

## 5. Especificação de implementação

**Nada disto foi escrito.** Esta seção especifica o que escrever.

### 5.1 Estrutura de arquivos

```
/opt/pocket-lm/
  evalx/                      # pacote novo — NÃO mexer no eval.py atual, que continua útil
    __init__.py
    lm_wrapper.py             # PocketLM(TemplateLM): a interface do harness
    bpb.py                    # BPB / byte-ppl / token-ppl no texto congelado
    lastword.py               # CALAME-PT e LAMBADA-PT: acurácia (gen) + NLL (loglik)
    sanity.py                 # rep_token_rate, distinct_2, palavras_no_lexico, entropia
    judge.py                  # gera as 200 amostras e monta o lote cego pro juiz
    baselines.py              # n-grama KN-5 + modelo aleatório + carregamento do Tucano-160m
    run_eval.py               # CLI orquestrador; única coisa que escreve o JSON
  eval/
    val_frozen.txt            # 2 MB de texto pt-BR retido; SHA-256 no JSON
    lexico.txt                # ~200k formas extraídas de data_big/corpus.txt
    prompts_juiz.json         # 40 prompts em 4 famílias
    rubrica_juiz.md           # a rubrica de §2.4.1, versionada e com hash
    ancoras_juiz.json         # 3 amostras de calibração com nota acordada
  results/
    eval-<step>-<git_sha>.json
```

### 5.2 `evalx/lm_wrapper.py` — a única peça que fala com o harness

Referência da interface:
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md

```python
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model

@register_model("pocketlm")
class PocketLM(TemplateLM):
    def __init__(self, ckpt: str, tokenizer: str, batch_size: int = 8,
                 max_length: int | None = None, device: str = "cpu",
                 quant: str | None = None): ...

    # --- propriedades exigidas ---
    @property
    def eot_token_id(self) -> int: ...      # ver nota crítica abaixo
    @property
    def max_length(self) -> int: ...        # cfg.block_size = 512
    @property
    def max_gen_toks(self) -> int: ...      # 16
    @property
    def batch_size(self) -> int: ...
    @property
    def device(self) -> str: ...

    # --- exigidos por TemplateLM ---
    def tok_encode(self, string: str, **kw) -> list[int]: ...
    def tok_decode(self, tokens: list[int]) -> str: ...
    def _loglikelihood_tokens(self, requests, disable_tqdm=False) -> list[tuple[float, bool]]: ...

    # --- exigidos pela LM base ---
    def loglikelihood_rolling(self, requests) -> list[float]: ...
    def generate_until(self, requests) -> list[str]: ...
```

Contratos (do `model_guide.md`):
- `loglikelihood(requests) -> list[tuple[float, bool]]`, `Instance.args = (context, continuation)`.
  O `bool` é `is_greedy`: `True` se a continuação é a sequência de maior probabilidade.
- `loglikelihood_rolling(requests) -> list[float]`, `Instance.args = (string,)`.
  Log-verossimilhança do texto inteiro, condicionada só no token EOT.
- `generate_until(requests) -> list[str]`, `Instance.args = (context, {"until": [...],
  "max_gen_toks": N})`.

**Seis armadilhas de implementação, todas concretas:**

1. **`eot_token_id` não existe hoje.** `bpe_prepare.py` treina o BPE sem EOS/BOS. Decidir e
   **documentar**: reservar o id 0 como EOT (`tok_decode([0])` deve ser `""`), ou re-treinar
   o tokenizer com `<|endoftext|>`. `loglikelihood_rolling` **prefixa EOT** e `loglikelihood`
   com contexto vazio depende dele — sem uma decisão explícita, os números vêm errados em
   silêncio.

   > **Correção (27/07/2026): a premissa é falsa — o token JÁ EXISTE.**
   > `bpe_prepare.py::treina_tokenizer()` chama
   > `tok.train(..., special_tokens=["<|endoftext|>"])`, e `data_big/tokenizer.json` confirma:
   > `added_tokens = [{"id": 0, "content": "<|endoftext|>", "special": true}]`. Ou seja
   > **`eot_token_id = 0` está correto** (o exemplo de JSON de §5.4 acertou), não há decisão a
   > tomar e não há re-treino de tokenizer a fazer. O que **fica** da armadilha: o token existe
   > no vocabulário mas **nunca é emitido** — o `tokeniza()` atual separa documentos só por
   > `\n\n` no texto. Emitir o EOT entre documentos é uma mudança do `bpe_prepare.py` v2
   > (`montagem-corpus.md` §5.3 item 3), e ela muda a distribuição de tokens: anotar em
   > `meta.json` como `eot_entre_docs`, senão a comparação com o run A deixa de ser auditável.
2. **Truncar pela ESQUERDA.** Ao exceder `max_length=512`, cortar o **contexto**, nunca a
   continuação. Belebele tem prompts que chegam perto de 500 tokens.
3. **`is_greedy` é obrigatório** e é o que faz `acc` funcionar. Calcular comparando o argmax
   por posição com o token-alvo.
4. **Último token do alvo não entra no forward.** O `model_guide.md` é explícito:
   *"the final token of the target is not passed into the LM"*. Errar isso desloca todos os
   índices em 1 e produz números plausíveis mas errados.
5. **RAM: `log_softmax` por amostra, em laço.** Nunca `F.log_softmax` sobre `B×T×16384`
   inteiro (§3.3). Gather das posições de continuação e `del logits` imediato.
6. **Geração usa `GPTExport`, não `GPT.generate`.** `generate_until` deve instanciar
   `GPTExport.from_training_model(gpt)` e usar `prefill`/`decode_step` (§3.2). Diferença de
   10–20× no tempo de parede.

**Como invocar.** Duas rotas; a segunda é a preferida:
```bash
# rota A: CLI (exige o arquivo dentro de lm_eval/models/ e import no __init__)
venv/bin/python -m lm_eval --model pocketlm \
  --model_args ckpt=gpu_out/t4-2/ckpt.pt,tokenizer=data_big/tokenizer.json,batch_size=16 \
  --tasks assin2_rte --num_fewshot 0 --batch_size 16 --output_path results/

# rota B: como biblioteca — não precisa tocar no pacote instalado
#   lm_eval.simple_evaluate(model=PocketLM(...), tasks=["assin2_rte","belebele_por_Latn"],
#                           num_fewshot=0, batch_size=16)
```

### 5.3 `evalx/bpb.py` e `evalx/lastword.py` — não dependem do harness

Ambos usam só `torch`, `numpy`, `tokenizers` e `datasets` — **tudo já instalado**.
Esta é a razão de existirem separados: dão o primeiro número **sem instalar nada** (§7).

```python
# bpb.py
def bpb(model, tok, texto: str, block: int = 512, stride: int = 256,
        batch: int = 4) -> dict:
    """-> {bits_per_byte, byte_perplexity, token_nll_nats, token_perplexity,
           tokens, bytes, bytes_por_token, stride}"""

# lastword.py
def lastword_nll(model, tok, itens: list[dict], batch: int = 16) -> dict:
    """itens = [{'sentence':..., 'last_word':...}]
    -> {nll_medio_nats, bpb_palavra, n}"""

def lastword_acc(export_model, tok, itens: list[dict],
                 max_new_tokens: int = 4) -> dict:
    """Greedy via GPTExport (prefill + decode_step).
    Normalização IDÊNTICA à do Tucano: re.sub(r'[.,?!]', '', w).strip(), exact match.
    -> {acc, n, erro_padrao}"""
```

Carregamento dos datasets (sem download manual, < 10 MB no total):
```python
load_dataset("NOVA-vision-language/calame-pt", "all", split="train")   # 2076
load_dataset("TucanoBR/lambada-pt", split="train")                     # 5153
load_dataset("nilc-nlp/assin2", split="test")                          # 2448
load_dataset("facebook/belebele", "por_Latn", split="test")            # 900
```

### 5.4 Formato do JSON de resultado

Um arquivo por avaliação, em `results/eval-<step>-<git_sha>.json`. Campos obrigatórios —
a regra é: **tudo que, se mudar, muda o número, tem de estar aqui.**

```json
{
  "schema": "pocket-lm/eval/1",
  "gerado_em": "2026-07-28T03:14:00Z",
  "git_sha": "182c7bf",
  "host": "x86",
  "modelo": {
    "ckpt": "gpu_out/t4-2/ckpt.pt",
    "sha256_ckpt": "…",
    "step": 12500,
    "params": 33847296,
    "params_embedding": 8388608,
    "n_layer": 8, "n_head": 8, "n_embd": 512,
    "block_size": 512, "vocab_size": 16384,
    "quant": "fp32"
  },
  "tokenizer": {
    "path": "data_big/tokenizer.json",
    "sha256": "…",
    "vocab_size": 16384,
    "eot_token_id": 0,
    "bytes_por_token_val": 19.82
  },
  "dados": {
    "val_frozen": "eval/val_frozen.txt",
    "sha256": "…",
    "bytes": 2097152,
    "tokens": 105800
  },
  "metricas": {
    "bits_per_byte":          {"valor": 1.842, "stride": 256},
    "byte_perplexity":        {"valor": 3.585},
    "token_nll_nats":         {"valor": 3.245, "aviso": "NAO comparavel entre vocabularios"},
    "token_perplexity":       {"valor": 25.66, "aviso": "idem"},

    "calame_pt_acc":          {"valor": 0.081, "n": 2076, "erro_padrao": 0.006,
                               "piso": 0.0, "ref_tucano160m": 0.5231},
    "calame_pt_lastword_nll": {"valor": 7.91,  "n": 2076},
    "calame_pt_lastword_bpb": {"valor": 2.13,  "n": 2076},

    "lambada_pt_acc":         {"valor": 0.021, "n": 2000, "subset_seed": 1337,
                               "ref_tucano160m": 0.2816},

    "assin2_rte_f1_macro":    {"valor": 0.341, "n": 2448, "shots": 0,
                               "piso_constante": 0.333, "piso_aleatorio": 0.500,
                               "ref_tucano160m_15shot": 0.3351},
    "belebele_por_acc":       {"valor": 0.252, "n": 900, "erro_padrao": 0.0144, "shots": 0,
                               "piso": 0.25, "piso_majoritario": 0.279},

    "rep_token_rate":         {"valor": 0.11},
    "distinct_2":             {"valor": 0.62},
    "palavras_no_lexico":     {"valor": 0.58},
    "entropia_media_bits":    {"valor": 5.2}
  },
  "juiz": {
    "modelo": "<model id exato>",
    "rubrica_sha256": "…",
    "n_amostras": 200, "temperatura": 1.0, "top_k": null,
    "cego": true, "ancora_ok": true,
    "gramatica":    {"media": 4.2, "desvio": 1.8},
    "coerencia":    {"media": 2.1, "desvio": 1.4},
    "consistencia": {"media": 1.9, "desvio": 1.5},
    "fluencia":     {"media": 3.0, "desvio": 1.6},
    "factualidade": {"media": 1.1, "desvio": 0.9},
    "nivel_moda": "N2"
  },
  "baselines": {
    "aleatorio_33m":   {"bits_per_byte": 14.0, "calame_pt_acc": 0.0},
    "ngram_kn5":       {"bits_per_byte": 2.31, "calame_pt_acc": 0.041},
    "tucano_160m":     {"bits_per_byte": 1.12, "calame_pt_acc": 0.5231,
                        "aviso": "162M params / 200B tokens; possivel contaminacao do val_frozen"}
  },
  "custo": {"segundos_cpu": 3480, "pico_rss_mb": 1490, "tokens_por_s": 292},
  "harness": {"lm_eval_version": "0.4.12", "tasks": ["assin2_rte", "belebele_por_Latn"]}
}
```

Regras do formato:
- Todo valor de acurácia carrega **`piso`** ou **`piso_constante`/`piso_majoritario`**.
  Um número de acurácia sem o piso ao lado é inutilizável e não deve ser emitido.
- Toda métrica dependente de tokenizer carrega **`aviso`**.
- `custo` é preenchido desde o primeiro run — é o que substitui as estimativas do §3.
- **`results/` versionado no git** (são KB), para que a série histórica exista de fato.

### 5.5 Dependências e tamanhos

**Já instalado no venv** (nada a fazer): `torch 2.13.0+cpu`, `numpy 2.5.1`,
`tokenizers 0.23.1`, `datasets 5.0.0`.

**Nível 1 — BPB + CALAME + sanity (§7): ZERO dependências novas.**

**Nível 2 — harness (só quando quiser `assin2_rte`/`belebele`):**

| Pacote | Versão (27/07/2026) | Wheel cp314? | Disco instalado (aprox.) |
|---|---|---|---|
| `lm_eval` | 0.4.12 | pure-python | ~30 MB |
| `transformers` | 5.14.1 | pure-python | ~60 MB |
| `scipy` | 1.18.0 | **sim** (20 wheels) | ~120 MB |
| `scikit-learn` | 1.9.0 | **sim** (12 wheels) | ~45 MB |
| `sacrebleu` | 2.6.0 | pure-python | ~15 MB |
| `evaluate` | 0.4.6 | pure-python | ~2 MB |
| `pytablewriter` | 1.2.1 | pure-python | ~5 MB |
| `sqlitedict` | 2.1.0 | **não — compila do sdist** | < 1 MB |
| `jsonlines`, `numexpr`, misc | — | — | ~20 MB |
| **Total** | | | **~400–600 MB** |

Risco único: **`sqlitedict` não tem wheel** e precisa compilar em Python 3.14. Se falhar,
o plano B é um venv paralelo com Python 3.12 só para o harness — os módulos de Nível 1
continuam funcionando no venv principal, o que é justamente por que eles não dependem do
harness.

**Nível 3 — baselines:** `huggingface_hub` (já vem com `datasets`) para baixar o
Tucano-160m (**~650 MB de pesos em disco**) e `transformers` para carregá-lo.
Só depois que o Nível 1 der número.

---

## 6. Ordem de execução

Três níveis de granularidade, por custo:

### Nível A — "por checkpoint", ~10 min de CPU, zero dependência nova
Roda dentro do próprio ciclo de treino, ou logo depois de cada `fetch_ckpt.py`.
1. `sanity.py` (< 1 min) — pega colapso antes de gastar qualquer coisa.
2. `bpb.py` em 500 KB do `val_frozen.txt` (~3 min).
3. `lastword.py::lastword_nll` num subconjunto fixo de 500 itens do CALAME-PT (~4 min).

**Isto é o que dá sinal já na próxima rodada de treino.** Não exige harness, não exige
instalar nada, não exige GPU.

### Nível B — "por rodada de GPU", ~1 h de CPU, sob `heavy.sh`
Depois de cada job de GPU concluído.
4. `bpb.py` completo em 2 MB (~12 min).
5. `lastword_nll` + `lastword_acc` completos no CALAME-PT (~24 min).
6. `assin2_rte` 0-shot (~19 min) — **null check**, esperar 0,333.

### Nível C — "por marco", noite inteira + custo de API
Quando algo estrutural muda (vocab 8k, destilação MiniPLM, QAT int4, mistura nova).
7. `belebele_por_Latn` 0-shot (~56 min) — null check.
8. `lambada_pt` (subconjunto 2.000, ~31 min).
9. **Juiz-LLM** com a rubrica de §2.4.1 (200 amostras, cego, com âncora).
10. Baselines: n-grama KN-5, modelo aleatório e (uma vez só) **Tucano-160m** no
    `val_frozen.txt` (~1–1,5 h).

### Ordem de construção (o que escrever primeiro)

| # | Entrega | Depende de | Esforço |
|---|---|---|---|
| 1 | `eval/val_frozen.txt` + SHA-256 + `evalx/bpb.py` | nada | ~2 h |
| 2 | `evalx/sanity.py` + `eval/lexico.txt` | nada | ~1,5 h |
| 3 | `evalx/lastword.py` (NLL primeiro, acc depois) | `datasets` (já tem) + `GPTExport` (já tem) | ~3 h |
| 4 | `evalx/run_eval.py` + esquema JSON de §5.4 | 1–3 | ~2 h |
| 5 | `evalx/baselines.py` (aleatório + n-grama) | 4 | ~3 h |
| 6 | `evalx/lm_wrapper.py` + instalar Nível 2 | 4 | ~4 h |
| 7 | `evalx/judge.py` + prompts + rubrica + âncoras | 4 | ~4 h |
| 8 | Tucano-160m como baseline | 6 | ~2 h |

**Justificativa da ordem:** os itens 1–4 dão **um número objetivo com zero risco de
infraestrutura** (sem instalação, sem download, sem RAM extra) e cobrem as três métricas que
efetivamente vão se mover em 33,8M. O harness (item 6) entra depois porque as tarefas que ele
traz são **null checks**, não métricas de progresso — inverter essa ordem é gastar o esforço
mais arriscado (instalação em Python 3.14, +600 MB de disco, +600 MB de RAM) para obter os
números que menos informam.

**Antes de qualquer mudança de vocabulário** (Decisão 3 da varredura), os itens 1 e 4 têm de
estar prontos e ter produzido **pelo menos um BPB registrado** no checkpoint atual. Caso
contrário a comparação 16k vs 8k não será possível — que é literalmente o alerta de
`varredura-institucional.md` §10.2.

---

## 7. PRÓXIMO PASSO CONCRETO

**Objetivo:** ter o primeiro número objetivo e comparável do projeto — **BPB** e
**CALAME-PT-NLL** no checkpoint atual — **sem instalar nada, sem baixar modelo, sem GPU.**

**Arquivos a criar (3):**
1. `eval/val_frozen.txt` — 2 MB retidos de `data_big/corpus.txt`, de um trecho **não usado
   no treino**, com o SHA-256 anotado.
2. `evalx/bpb.py` — a função `bpb()` de §5.3. ~80 linhas. Reaproveita `load_model()` e
   `HFTok` do `eval.py` atual, que já fazem exatamente o carregamento necessário.
3. `evalx/lastword.py` — só a função `lastword_nll()` de §5.3 (a `lastword_acc` fica para
   depois; a NLL é mais barata e mais sensível nesta escala). ~70 linhas.

**Comandos:**

```bash
cd /opt/pocket-lm

# 1. BPB do checkpoint atual — não instala nada, usa o venv como está
MEMCAP=2200M WAIT=1800 ./heavy.sh venv/bin/python -m evalx.bpb \
  --ckpt ckpt.pt --tok data_big/tokenizer.json \
  --texto eval/val_frozen.txt --stride 256 --batch 4 \
  --json results/bpb-baseline.json

# 2. CALAME-PT NLL (baixa 1 MB de parquet; datasets 5.0.0 ja esta instalado)
MEMCAP=2200M WAIT=1800 ./heavy.sh venv/bin/python -m evalx.lastword \
  --ckpt ckpt.pt --tok data_big/tokenizer.json \
  --dataset NOVA-vision-language/calame-pt --config all --split train \
  --batch 16 --json results/calame-nll-baseline.json

# 3. Baseline de "zero conhecimento" — o mesmo BPB num modelo nao treinado.
#    Sem isto o numero do passo 1 nao tem escala.
MEMCAP=2200M WAIT=1800 ./heavy.sh venv/bin/python -m evalx.bpb \
  --random-init --vocab 16384 --tok data_big/tokenizer.json \
  --texto eval/val_frozen.txt --stride 256 --batch 4 \
  --json results/bpb-random.json
```

**Tempo:**

| Etapa | Tempo |
|---|---|
| Escrever `val_frozen.txt` + os dois módulos | **~4–5 h de engenharia** |
| Comando 1 (BPB, 2 MB) | ~12 min de CPU |
| Comando 2 (CALAME-PT NLL, 2.076 itens) | ~16 min de CPU |
| Comando 3 (BPB do modelo aleatório) | ~12 min de CPU |
| **Total de máquina** | **~40 min**, todos sob `heavy.sh` (teto de 2,2 GB, serializado contra o cron de treino) |

**O que se sabe ao fim:** dois números com escala definida — *quanto* o modelo aprendeu acima
de zero (comando 3) e *o quanto falta* para o Tucano-160m (52,31 em CALAME-PT). E, mais
importante, **a partir daí toda mudança de treino da varredura (vocab 8k, MiniPLM, QAT int4)
passa a ser verificável**, que era exatamente a condição que a DECISÃO Nº 1 impunha.

---

## Fontes

**Harness e tarefas**
- https://github.com/EleutherAI/lm-evaluation-harness
- https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/portuguese_bench/README.md
- https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/portuguese_bench
- https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/belebele/_default_template_yaml
- https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/wikitext.yaml
- https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md
- https://github.com/eduagarcia/lm-evaluation-harness-pt

**Datasets**
- https://huggingface.co/datasets/NOVA-vision-language/calame-pt
- https://huggingface.co/datasets/TucanoBR/lambada-pt
- https://huggingface.co/datasets/nilc-nlp/assin2
- https://huggingface.co/datasets/nilc-nlp/assin
- https://huggingface.co/datasets/facebook/belebele
- https://huggingface.co/datasets/ruanchaves/napolab · https://github.com/ruanchaves/napolab

**Modelos e resultados de referência**
- https://huggingface.co/TucanoBR/Tucano-160m
- https://github.com/Nkluge-correa/Tucano/blob/main/evaluations/README.md
- https://www.sciencedirect.com/science/article/pii/S2666389925001734 (Tucano, Patterns)
- https://nkluge-correa.github.io/Tucano/

**Papers**
- https://arxiv.org/abs/2305.07759 · https://ar5iv.labs.arxiv.org/html/2305.07759 (TinyStories)
- https://arxiv.org/abs/2402.12969 (GlórIA / CALAME-PT, PROPOR 2024)
- https://arxiv.org/html/2511.17808 · https://github.com/PoETaV2/PoETaV2 (PoETa v2)
- https://arxiv.org/pdf/2605.12928 (fórmula de bits-per-byte)
- https://huggingface.co/papers (substituto do Papers with Code, encerrado em 24/07/2025)

**Documentos internos**
- `docs/varredura-institucional.md` §7 (bloco de avaliação), §10.2 (armadilha do `val_loss`)
- `docs/estado-da-arte.md` §1.1 (nota do Tucano a corrigir), §4.3 (Trilha B)
- `docs/receita-treino.md` (mudanças que esta avaliação precisa medir)
- `eval.py` (perplexidade + amostragem já existentes), `model_export.py` (`GPTExport`,
  KV cache), `quantize.py` (`estimate_loss_export`, variantes int8/int4), `heavy.sh`
