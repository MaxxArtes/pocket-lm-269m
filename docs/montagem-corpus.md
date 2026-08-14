# Montagem do corpus — de Wikipedia-pt puro para ≥400 M tokens únicos

**STATUS: COMPLETO** (27/07/2026). Especificação executável, não levantamento.

> **Nota de consolidação (27/07/2026):** este documento foi escrito **depois** da correção do
> corpus e já usa os números certos (202,6 M tokens únicos, 3,948 bytes/token, `all.bin`
> truncado como modo de falha) — não precisa de correção, e é a especificação vigente do item
> 3 do `PLANO.md`. **Duas ressalvas herdadas de documentos que foram corrigidos:**
> 1. **A escolha de máquina do passo 7 ("Run T4, ~1,0 unidade") vem de `analise-runs.md` §6.1,
>    cuja premissa caiu.** A H200 compra **1,47× mais FLOPs por unidade** que a T4 (2,15× a
>    MFU igual). A T4 continua certa **para um orçamento de ~1 unidade** (8 min de H200 não
>    amortizam o spin-up), mas se o orçamento subir para ≥3 unidades a resposta inverte.
>    Ver `PLANO.md` §7, contradição 4.
> 2. **§5.3 item 3 está certo e corrige `plano-avaliacao.md`:** `<|endoftext|>` **existe**, é o
>    id 0, e é criado por `bpe_prepare.py::treina_tokenizer()` via
>    `special_tokens=["<|endoftext|>"]` — verificado em `data_big/tokenizer.json`. O que falta
>    é **emiti-lo**, não criá-lo.
O levantamento de fontes e licenças está em [`corpus-ptbr.md`](corpus-ptbr.md); aqui a
**Mistura A** vira números, ordem de execução, orçamento de disco e contrato de código.

**Premissa herdada** (`analise-runs.md` §6, `prune-and-distill.md` §5.5): o gargalo é **D**,
não **N**. Manter `N = 33,87 M` e gastar tudo em dados. Alvo mínimo **≥98 M tokens únicos**;
alvo do experimento de destilação **≥400 M**. Ganho estimado só de dados: **0,77 nat**,
contra **0,1175 nat** que 6,5× mais parâmetros compraram por 3,14 unidades de H200.

---

## 0. Estado medido na bancada, hoje

Números apurados nesta sessão, não estimados:

| Fato | Valor | Fonte |
|---|---|---|
| `data_big/corpus.txt` | 800.053.953 bytes, só Wikipedia-pt | `ls` |
| `data_big/meta.json` (retokenização recém-concluída) | **202.630.887 tokens** | arquivo |
| bytes/token medido, BPE 16k sobre Wikipedia-pt | **3,948** | 800.053.953 / 202.630.887 |
| bytes/token medido em amostra de 3 MB | 4,285 | `analise-runs.md` §0.2(a) |
| bytes/token medido em amostra de 4 MB | 4,140 | `vocab-e-destilacao.md` §A.1 |
| bytes/token do `all.bin` truncado pelo OOM | **19,82** | `plano-avaliacao.md` §2.1 |
| Disco livre em `/` | 6,3 GB (planejar com **5 GB**) | `df` |
| RAM / swap | 3.814 MB / 2.047 MB | `free` |
| vCPU | **2** (AMD EPYC-Rome) | `nproc` |

### 0.1 A constante de planejamento: 4,42 bytes/token

O documento usa **4,42 bytes/token** como constante de orçamento, conforme instruído. Ela
**não** é a compressão medida em Wikipedia (3,948) — é a compressão esperada do **mesmo BPE
16k, treinado em Wikipedia, aplicado à mistura nova**. A razão é mecânica: o BPE atual
aprendeu merges de prosa enciclopédica limpa; texto de web, código, matemática e OCR de
1870 caem fora desses merges e fragmentam mais. `3,948 × 1,12 = 4,42` — uma penalidade de
compressão de **12%**, que é a ordem de grandeza esperada para troca de domínio sem
retreinar merges.

Consequência prática, e é a direção certa do erro:

- Para **orçar disco e download** (tokens → bytes), 4,42 é **conservador**: pede mais bytes
  por token do que o real, então o plano de disco tem folga embutida.
- Para **contar tokens colhidos** (bytes → tokens), 4,42 é **pessimista**: se a compressão
  real vier em 4,1, colhemos ~7% mais tokens que o alvo. Aceitável.
- Se vier **pior** que 4,42 (ex.: 4,8, plausível se o OCR do domínio público pesar), o alvo
  de 420 M cai para ~387 M. Por isso o alvo interno deste plano é **420 M**, não 400 M: os
  5% de folga cobrem exatamente essa incerteza.

**Toda etapa mede o valor real e o grava em `meta.json`.** A constante é para planejar, não
para reportar. Ver §5.4 — é a mesma medição que teria pego o truncamento de hoje.

### 0.2 Por que não retreinar o BPE nesta rodada

`corpus-ptbr.md` §5.5 recomenda retreinar o BPE sobre a mistura nova. **Recomendo adiar**,
por três razões que só ficaram visíveis depois de `analise-runs.md`:

1. **Confunde o experimento.** A rodada recomendada existe para **medir β** com `ΔN = 0`
   exatamente e tudo o mais constante (§6.4 de `analise-runs.md`). Trocar os merges muda a
   distribuição de tokens, a entropia unigrama (7,5181 nats hoje) e a escala da loss. Fica
   impossível dizer se a melhora veio de mais dados ou de melhor tokenização.
2. **Quebra a comparabilidade com o professor de 219 M**, que é 16k treinado no BPE atual —
   pré-condição 4 e 5 de `prune-and-distill.md` §5.5.
3. **Custa a etapa mais cara e mais arriscada da bancada.** Treinar `ByteLevelBPETokenizer`
   em 200 MB com **2 vCPU** é 15–30 min e é o único passo que se aproxima do teto de RAM.
   O ganho é ~12% de compressão — que se compra de graça colhendo 12% mais bytes.

**Decisão:** reusar `data_big/tokenizer.json` sem alteração. Retreinar o BPE vira um
experimento próprio, depois de β medido, junto com a troca 16k→4k de `vocab-e-destilacao.md`.

---

## 1. A mistura concreta — 420 M tokens únicos

Proporções da **Mistura A** de `corpus-ptbr.md` §4, agora com orçamento por fonte. Alvo
interno **420 M** (5% de folga sobre os 400 M pedidos, ver §0.1).

### 1.1 Tabela mestra

| % | Dataset (nome exato no HF) | Config / subset | Split | Filtro | Campo | Tokens | Texto (×4,42) | Licença |
|---:|---|---|---|---|---|---:|---:|---|
| **40%** | `Polygl0t/gigaverbo-v2` | `default` | `train` | `edu_int_score >= 3 and toxic_int_score == 1` | `text` | **168,0 M** | 742,6 MB | `other` = ODC-By 1.0 + CC ToU |
| **20%** | `Polygl0t/gigaverbo-v2-synth` | `default` | `train` | nenhum | `text` | **84,0 M** | 371,3 MB | **Apache-2.0** |
| **15%** | `Boakpe/historinhas` | (único) | `train` | `len(texto) >= 200` | `texto` | **63,0 M** | 278,5 MB | **Apache-2.0** |
| **12%** | *local* `data_big/corpus.txt` (= `wikimedia/wikipedia` `20231101.pt`) | — | — | faixa de bytes declarada, disjunta do `val_frozen` | — | **50,4 M** | 222,8 MB | CC-BY-SA-3.0 + GFDL |
| **8%** | `Polygl0t/gigaverbo-v2` | `default` | `train` | `edu_int_score <= 2 and toxic_int_score == 1` | `text` | **33,6 M** | 148,5 MB | idem 40% |
| **5%** | `PleIAs/Portuguese-PD` | (único) | `train` | anti-ortografia-arcaica + `doc_cap = 100 kB` | `text` | **21,0 M** | 92,8 MB | **domínio público** |
| | | | | | **TOTAL** | **420,0 M** | **1.856,4 MB** | |

**420,0 M ≥ 400 M.** Fecha com 20 M de margem mesmo antes da folga de compressão.

### 1.2 Quanto de cada fonte isso consome (a checagem que importa)

Nenhuma fonte é raspada perto do fim — não há risco de "acabou o dataset no meio do build":

| Fonte | Disponível | Usado | Fração |
|---|---:|---:|---:|
| `gigaverbo-v2` `edu_int_score>=3` | 118,73 B tok | 168,0 M | **0,142%** |
| `gigaverbo-v2` `edu_int_score<=2` | ~199,0 B tok | 33,6 M | **0,017%** |
| `gigaverbo-v2-synth` | 9,322 B tok | 84,0 M | **0,90%** |
| `Boakpe/historinhas` | ~0,36–0,45 B tok | 63,0 M | **~15%** |
| Wikipedia-pt (já em disco) | 202,6 M tok (medido) | 50,4 M | **24,9%** |
| `PleIAs/Portuguese-PD` | ~1,00 B tok | 21,0 M | **2,1%** |

A única fonte com aperto é `historinhas` a ~15%. É aceitável: são 1,26 M histórias e
tomamos ~174 mil delas, escolhidas com passo determinístico (§4.2), não as 174 mil
primeiras — a localidade do parquet agrupa tema e personagem.

### 1.3 Documentos por fonte (dimensiona o dedup e o buffer de shuffle)

| Fonte | tok/doc médio | Docs colhidos |
|---|---:|---:|
| gv2 edu>=3 | ~854 | ~197 k |
| gv2 edu<=2 | ~854 | ~39 k |
| synth | ~829 | ~101 k |
| historinhas | ~362 | ~174 k |
| wikipedia | ~500 | ~100 k |
| Portuguese-PD (com cap de 100 kB) | ~22,6 k | **~930** |
| **TOTAL** | | **~612 k documentos** |

Os 612 k documentos são o número que governa a §3: cabe folgadamente em RAM como conjunto
de hashes, e não cabe como MinHash em tempo de CPU aceitável.

### 1.4 Três correções ao levantamento

Verificadas nos cards hoje, valem para `corpus-ptbr.md`:

1. **`Polygl0t/gigaverbo-v2-synth` é Apache-2.0.** O levantamento diz "o card não declara
   licença explícita". O card declara. É a segunda melhor licença da mistura, o que torna os
   20% de sintético confortáveis mesmo no cenário "isto vira produto".
2. **Colunas exatas do `gigaverbo-v2`:** `text`, `source`, `subset`, `id` (md5), `token_count`,
   `edu_score`, `edu_int_score`, `toxic_score`, `toxic_int_score`. O `id` é md5 e serve de
   chave de dedup entre execuções **de graça** (§3.2). O `token_count` permite orçar sem
   tokenizar (§5.2).
3. **Colunas do `Portuguese-PD`:** `identifier`, `creator`, `title`, `publication_date`,
   `word_count`, `text` — e `publication_date` habilita um filtro melhor que o de dígrafos:
   pegar preferencialmente os títulos mais recentes (perto de 1884), que já estão mais
   próximos da ortografia moderna.

### 1.5 Mistura mínima (Etapa 1) — 112 M tokens únicos

Mesmas proporções, mesmo código, só `CORPUS_TARGET_TOKENS` menor. 112 M dá **3,49 épocas**
nos 391 M processados da config recomendada — dentro do regime "até 4 épocas ≈ dado novo"
(§3.4).

| % | Fonte | Tokens | Texto |
|---:|---|---:|---:|
| 40% | gv2 `edu_int_score>=3` | 44,8 M | 198,0 MB |
| 20% | gv2-synth | 22,4 M | 99,0 MB |
| 15% | historinhas | 16,8 M | 74,3 MB |
| 12% | wikipedia (local) | 13,4 M | 59,4 MB |
| 8% | gv2 `edu_int_score<=2` | 9,0 M | 39,6 MB |
| 5% | Portuguese-PD | 5,6 M | 24,8 MB |
| | **TOTAL** | **112,0 M** | **495,1 MB** |

**Não use a Mistura C** (`70% gv2-edu + 30% historinhas`) para a Etapa 1. Ela existia como
"mínimo esforço", mas com o desenho de §5 as seis fontes custam os mesmos minutos que duas —
e duas fontes reintroduzem exatamente o problema de monotonia de registro que motivou o
documento inteiro. Mistura C só sobrevive como plano de contingência se `Portuguese-PD` ou
`historinhas` estiverem fora do ar na hora do build.

---

## 2. Plano de download e disco

### 2.1 O resultado que muda a intuição

> **400 M tokens são 1,77 GB de texto e 800 MB de `.bin`. Cabem na bancada com folga.**

A intuição de que "400 M tokens exige data center" está errada por uma ordem de grandeza. O
que **não** cabe é o *dataset de origem* (834 GB do GigaVerbo-v2) — e ele nunca é
materializado, porque `streaming=True` lê row-group por row-group e descarta.

**A montagem roda na bancada x86.** Ver §2.5 para a única parte que não roda.

### 2.2 Orçamento de disco, Etapa 2 (420 M)

| Artefato | Tamanho | Vive até |
|---|---:|---|
| `parts/*.txt.gz` (6 arquivos, gzip nível 1, razão ~3,2×) | **~580 MB** | fim do build |
| `train.bin.part` + `val.bin.part` (uint16) | **840 MB** | renomeados no fim |
| `eval/val_frozen.txt` | 2 MB | permanente |
| `data_big/tokenizer.json` | 1,2 MB | permanente |
| **Pico** | **~1,42 GB** | |
| Residente após o build (bins + tokenizer + meta + val_frozen) | ~843 MB | |

Com 5 GB livres isso deixa **3,6 GB de margem**. Não há aperto.

Três decisões que produzem esse número, e cada uma resolve um risco real:

1. **`parts/` é gzip.** `gzip.open(..., 'wt', compresslevel=1)` comprime pt-BR em ~3,2× a
   ~50 MB/s por core — irrelevante ao lado da rede. Economiza 1,28 GB de pico.
2. **Não existe `corpus.txt` intermediário.** A Etapa B (§2.4) tokeniza direto dos `parts/`
   para os `.bin`. Isso mata 1,86 GB de pico. Só é possível porque **não retreinamos o BPE**
   (§0.2) — se retreinássemos, precisaríamos de um arquivo de texto contíguo para o trainer.
   As duas decisões se sustentam.
3. **Não existe `all.bin`.** O código atual escreve `all.bin` inteiro, faz memmap e copia
   para `train.bin` + `val.bin`, e só então apaga — pico de **2×** o tamanho final. A v2
   escreve os dois `.bin` direto (§5.3), o que também elimina o artefato que hoje está
   truncado.

Ganho de espaço adicional, depois do primeiro build bem-sucedido: apagar
`data_big/corpus.txt` (**800 MB**), já que a fatia de Wikipedia vira `parts/wikipedia.txt.gz`
(~70 MB) e o `val_frozen.txt` está extraído. Não apagar antes.

### 2.3 Orçamento de download (rede, não disco)

Amplificação de leitura do filtro: `edu_int_score >= 3` retém 118,73 B de 317,7 B tokens =
**37,37%**, logo o stream lê **2,676 tokens por token aproveitado**. Densidade do parquet do
GigaVerbo-v2: `834 GB / 317,7 B tok = 2,625 bytes/token`. Custo efetivo:
**7,02 bytes baixados por token colhido de edu**.

| Fonte | Baixado (Etapa 2) | Baixado (Etapa 1) | Observação |
|---|---:|---:|---|
| gv2 `edu>=3` **+** `edu<=2` | **1,18 GB** | 0,31 GB | **uma única passada serve as duas fatias** |
| gv2-synth | 0,24 GB | 0,07 GB | 2,854 B/tok de parquet, 0,9% do dataset |
| historinhas | ~0,28 GB | ~0,08 GB | ~15% do dataset |
| Portuguese-PD | ~0,40 GB | ~0,11 GB | o `doc_cap` corta tokens, não download: o parquet entrega o livro inteiro |
| wikipedia | **0** | **0** | já está em `data_big/corpus.txt` |
| **TOTAL** | **~2,10 GB** | **~0,57 GB** | tudo descartado em streaming |

**A otimização que vale o dobro:** os 8% de web não-educacional saem dos documentos
**rejeitados** pela mesma passada que colhe os 40% de edu. A passada examina
`168 M / 0,3737 = 449,6 M` tokens, dos quais `449,6 × 0,6263 = 281,6 M` são `edu<=2` — **8,4×
mais** que os 33,6 M necessários. Fazer duas passadas separadas dobraria o download e o
tempo de parede da etapa mais cara, para colher algo que já passou pelo cabo.

Nada disso é *gated*: **nenhum `HF_TOKEN` é necessário** em nenhuma das seis fontes.

### 2.4 Ordem de download — por que esta e não outra

O desenho é **duas fases**, e a razão é RAM, não elegância.

`interleave_datasets` sobre 6 streams (o exemplo de `corpus-ptbr.md` §5.3) mantém **6
conexões HTTP e 6 decodificadores parquet simultâneos**. Um row-group do GigaVerbo-v2 pode
ter centenas de MB descomprimidos; seis buffers desses numa máquina de 3,8 GB é a receita
exata do OOM de hoje. Além disso, qualquer falha derruba o build inteiro sem deixar nada
reaproveitável.

**Fase A — colheita, uma fonte por vez, sequencial.** Um stream aberto por vez. Cada fonte
escreve seu próprio `parts/<nome>.txt.gz` com um orçamento de bytes fixo. Se uma fonte
falhar, as outras estão intactas em disco e o retry é só dela.

Ordem, do mais barato/certo para o mais caro/incerto — assim uma interrupção sempre deixa a
maior parte pronta:

| # | Fonte | Por que nesta posição | Tempo (Etapa 2) |
|---|---|---|---|
| 1 | **wikipedia (local)** | zero rede, zero risco, e produz o `val_frozen` de que todo o resto depende | ~1 min |
| 2 | **historinhas** | dataset pequeno, sem filtro, Apache-2.0; valida a esteira ponta a ponta em minutos | ~4–8 min |
| 3 | **gigaverbo-v2-synth** | 0,9% do dataset, sem filtro | ~4–8 min |
| 4 | **Portuguese-PD** | 6 arquivos grandes, docs enormes; falha cedo se o `doc_cap` estiver errado | ~4–8 min |
| 5 | **gigaverbo-v2** (edu + web, passada única) | o item caro; entra por último com todo o resto garantido | ~15–30 min |

**Fase B — mistura e tokenização.** Abre os 6 `parts/` (só 6 descritores de gzip, ~6 MB de
buffer), sorteia a fonte de cada documento com as probabilidades da §1.1, passa por um
buffer de shuffle de 20 mil documentos (~70 MB) e escreve `val.bin.part` e depois
`train.bin.part`. Memória total da Fase B: **< 400 MB**, dominada pelo `encode_batch`.

### 2.5 O que precisa acontecer fora da bancada

Só uma coisa: **o transporte dos `.bin` até a GPU**.

`submit_train.py` embarca o código como **base64 numa variável de ambiente** (`CODE_B64`).
Isso serve para ~50 KB de `.py`; **não serve para 840 MB de `.bin`**. Hoje o contorno é que
`run_gpu.sh` chama `bpe_prepare.py` *dentro do job*, ou seja, o corpus é montado com GPU
alugada ociosa ao lado — desperdício direto, e é o que `corpus-ptbr.md` §5.1 já condena.

O canal correto já existe no repo: `Teamspace.upload_folder` / `download_folder`
(`push_out.py`, `fetch_ckpt.py`) e o `path_mappings` documentado em `gpu-cron.md` §"path_mappings".

**Rota recomendada:**

1. Criar uma data-connection `pocket-lm-data` (mesmo mecanismo do `pocket-lm-state`).
2. Da bancada, subir `train.bin.gz`, `val.bin.gz`, `meta.json`, `tokenizer.json` — os `.bin`
   uint16 comprimem para ~55–65%, então **~500 MB** em vez de 840 MB.
3. Em `submit_train.py`, uma linha: `path_mappings={"/work/out": "pocket-lm-state", "/work/data": "pocket-lm-data"}`.
4. Em `run_gpu.sh`, antes de `bpe_prepare.py`:
   ```bash
   if [ -f /work/data/train.bin.gz ]; then
     mkdir -p data && gunzip -c /work/data/train.bin.gz > data/train.bin
     gunzip -c /work/data/val.bin.gz > data/val.bin
     cp /work/data/tokenizer.json /work/data/meta.json data/
   else
     python bpe_prepare.py
   fi
   ```
   **Descomprimir para disco local é obrigatório.** `np.memmap` sobre pasta montada em rede
   destrói o throughput de treino — que é exatamente a variável (MFU) que a §6.5 de
   `analise-runs.md` diz valer 0,471 nat.

Custo evitado: **8–15 min de download+tokenização por job pago**, em cada rodada, para
sempre. Numa T4 de 57 min isso é ~0,25 unidade por run.

**Veredito da pergunta:** *400 M tokens cabem na bancada — a montagem inteira roda no x86,
com pico de 1,42 GB de disco e < 2,2 GB de RAM. O que precisa sair da bancada é apenas o
artefato pronto, e o canal para isso é o Drive do Teamspace, não o bundle base64.*

**Fallback**, se a data-connection não estiver pronta na hora de disparar: rodar
`bpe_prepare.py` dentro do job com `CORPUS_MIX=A` e `CORPUS_TARGET_TOKENS=112000000`. Custa
~8–15 min de GPU e funciona. Aceitável **uma vez**; não como regime.

---

## 3. Deduplicação e limpeza

### 3.1 Por que dedup importa nesta escala — a evidência

Não é higiene, é o mecanismo de dano mais direto ao modelo pequeno:

- **Hernandez et al., 2022, "Scaling Laws and Interpretability of Learning from Repeated
  Data"** ([arXiv:2205.10487](https://arxiv.org/abs/2205.10487)). Repetir **0,1% dos dados
  100 vezes** degrada um modelo de **800 M ao nível de um de 400 M** — metade do modelo
  perdido — com 90% dos tokens ainda únicos. O mecanismo medido é que a repetição danifica
  desproporcionalmente as *induction heads* e as estruturas de cópia associadas à
  generalização, e há **double descent**: a loss de teste *sobe* no meio do treino. Para nós
  isto é literal: com 33,87 M de params, capacidade gasta memorizando um boilerplate
  repetido é capacidade que não existe para modelar a língua.
- **Lee et al., 2021, "Deduplicating Training Data Makes Language Models Better"**
  ([arXiv:2107.06499](https://arxiv.org/abs/2107.06499)). O C4 continha **uma frase de 61
  palavras repetida mais de 60.000 vezes**; deduplicar encolheu datasets em até **19%**,
  reduziu emissão de texto memorizado em **10×** e alcançou a mesma acurácia com **menos
  passos de treino**, sem piora de perplexidade. E o achado que a §4.3 usa: a sobreposição
  treino-teste afetava **mais de 4% do conjunto de validação**.
- **Muennighoff et al., 2023, "Scaling Data-Constrained Language Models"**
  ([arXiv:2305.16264](https://arxiv.org/abs/2305.16264)). Até **4 épocas** de dado repetido
  produzem mudança desprezível na loss frente a dado único; acima disso o valor do compute
  adicional decai a zero. É o que fixa 98–112 M como o mínimo defensável (§3.4).

Os três combinados dão a regra operacional: **duplicata concentrada é veneno (Hernandez),
repetição uniforme até 4× é de graça (Muennighoff), e o custo de evitar a primeira é baixo
(Lee)**.

### 3.2 O que cabe em 3,8 GB de RAM e 2 vCPU

**Duplicata exata de documento — obrigatória, custo desprezível.**
xxhash64 do texto normalizado, guardado num `set` de inteiros. 612 k documentos ≈ **43 MB**
de RAM. Para o GigaVerbo-v2 nem é preciso hashear: a coluna **`id` já é um md5** do
documento — dedup de graça, e o mesmo conjunto persiste entre execuções, o que faz a Etapa 2
pular tudo o que a Etapa 1 já colheu. Custo: **< 30 s** no build inteiro.

**Duplicata exata de parágrafo — obrigatória, via Bloom filter.**
~612 k docs × ~12 parágrafos ≈ **7,3 M parágrafos**. Um `set` de inteiros do CPython custaria
~440 MB (≈60 bytes por entrada). Um **Bloom filter em `bytearray`** com `m/n = 16` bits e
`k = 3` custa **20 MB** com taxa de falso-positivo de **0,17%** — perder 0,17% dos parágrafos
únicos é irrelevante, e não há falso-negativo (nenhuma duplicata escapa). Normalização antes
do hash: `casefold` + colapso de espaços + remoção de pontuação, e só parágrafos com
**≥200 caracteres** entram (parágrafos curtos legitimamente se repetem: "Referências",
"Ver também", "Fonte:"). Custo: **~2–4 min** de CPU nos 1,86 GB.

**MinHash / LSH — NÃO fazer. Recomendação explícita.**
Cabe em RAM (612 k × 128 hashes × 4 B = 313 MB de assinaturas) mas **não cabe em tempo**: em
Python puro sobre 2 vCPU, shingling de 5-gramas + 128 permutações roda a ~50–200 doc/s, ou
**1 a 3,4 horas** — mais que o build inteiro. E o retorno é próximo de zero por três motivos
concretos:

1. **As fontes já são MinHash-dedupadas internamente.** GigaVerbo-v2 aplicou o pipeline
   FineWeb-2 com `datatrove`: MinHash de 14 buckets × 8 hashes, 5-gramas, xxHash. FineWeb-2
   e CulturaX idem. O que sobra para nós é dedup **entre fontes**, não dentro delas.
2. **A sobreposição entre fontes é majoritariamente paráfrase, não cópia.** As sementes do
   `gigaverbo-v2-synth` incluem Wikipedia (996 M tok) e Historinhas (179 M tok), mas o
   sintético é **reescrita**, não cópia — MinHash de 5-gramas não pega paráfrase de qualquer
   jeito. Gastar 3 h para não achar nada.
3. **A sobreposição que é cópia literal é exatamente a que o hash de parágrafo pega**:
   Wikipedia espelhada dentro do crawl do GigaVerbo-v2.

**Veredito:** hash exato de documento + Bloom de parágrafo. **~4 min de CPU no total**, 63 MB
de RAM, e cobre a duplicação literal, que é a forma de dano documentada por Hernandez.

### 3.3 Duplicata esperada nessas fontes

| Par | Sobreposição esperada | Detectada por |
|---|---|---|
| gv2 (crawl) ↔ Wikipedia local | **1–4%** dos docs de gv2 — CommonCrawl varre espelhos da Wikipedia | hash de parágrafo |
| gv2 edu>=3 ↔ gv2 edu<=2 | ~0 — partição disjunta pelo score, mesma passada | trivial |
| synth ↔ Wikipedia e ↔ historinhas | **~0 em cópia literal**, alta em *conteúdo* | nada pega; é aceitável (é reescrita) |
| interna a gv2 / synth / historinhas | **< 1%** — já dedupadas na origem | hash de doc |
| Portuguese-PD ↔ qualquer | ~0 — livros anteriores a 1884 | — |
| boilerplate interno de historinhas | **médio** — campos controlados (`moral`, `tonalidade`) geram aberturas e fechos repetidos | hash de parágrafo, parcialmente |

Remoção total esperada: **2–5% dos documentos**. Orçar 5% a mais no alvo (já embutido nos
420 M vs 400 M) e **reportar a taxa real em `meta.json`** — uma taxa muito acima de 8% indica
filtro errado; muito abaixo de 1% indica que o dedup não está rodando.

### 3.4 Limpeza — o mínimo que muda o resultado

Aplicada por documento, na Fase A, na ordem abaixo (barata antes de cara):

1. **Normalização** — a de hoje (`\n{3,}→\n\n`, `[ \t]+→ `, `strip`) mais NFC Unicode e
   remoção de caracteres de controle (o OCR do PD injeta bastantes).
2. **Comprimento mínimo 200 bytes** — igual ao de hoje. O `default` do GigaVerbo-v2 já
   removeu docs com <50 tokens.
3. **Toxicidade**: `toxic_int_score == 1` em ambas as fatias do gv2. O `default` já removeu
   >3; exigir `== 1` é barato e a coluna já veio no registro.
4. **`Portuguese-PD`, dois filtros específicos:**
   - **`doc_cap = 100 kB` por livro.** Sem cap, os 21 M tokens vêm de ~166 títulos — sete
     autores dominariam 5% do corpus. Com cap, ~930 títulos. É diversidade de graça.
   - **Anti-ortografia arcaica.** Contar ocorrências de `ph`, `th`, `elle`, `ella`, `sciencia`,
     `-mente` com trema, `y` intervocálico por 1.000 palavras; descartar o documento acima de
     um limiar. Preferir `publication_date` alto (mais perto de 1884). Sem isso, 5% de
     "pharmacia" contamina a ortografia de um modelo que só tem 33,87 M de params para gastar.
5. **`historinhas`:** usar **só** o campo `texto`. Concatenar `titulo`/`tema`/`moral` criaria
   um cabeçalho de formato fixo em 15% do corpus — precisamente o molde repetitivo que
   estamos fugindo.
6. **Nada de classificador próprio.** `edu_int_score` e `toxic_int_score` vêm pré-computados
   no registro. Rodar BERTimbau na bancada seria horas de CPU para reproduzir uma coluna que
   já veio no parquet.

---

## 4. Mistura, ordem e blindagem do `val_frozen`

### 4.1 Embaralhar tudo — sem currículo

**Recomendação: embaralhar. Não fazer currículo.**

A evidência para modelo minúsculo é consistentemente negativa. O BabyLM Challenge explorou
currículo em escala pequena de forma exaustiva e o resultado agregado é que **currículos de
propósito geral — ordenação por dificuldade/relevância, mascaramento progressivo de
vocabulário, objetivos auxiliares — dão ganhos marginais ou nulos sobre ordem aleatória**
([Martinez et al., "CLIMB", arXiv:2311.08886](https://arxiv.org/pdf/2311.08886);
[Salhan et al., arXiv:2410.22886](https://arxiv.org/abs/2410.22886), que registra que
currículo "não levou a melhorias consideráveis sobre modelos sem currículo"). O que
ocasionalmente funciona são currículos **específicos de língua e teoricamente motivados por
aquisição** — pesquisa própria, não uma alavanca de engenharia.

Contra isso pesa um custo garantido: `corpus-ptbr.md` §5.3 já registra que **concatenar
blocos causa deriva de distribuição** — o modelo termina o treino tendo visto só o último
bloco recentemente, e as amostras finais refletem isso. Num run com decay WSD de 20% no fim,
onde o LR baixo grava com força o que passa por último, um currículo mal-ordenado escreve o
viés direto no checkpoint entregue.

**A única "ordem" que fica:** o *cooldown* de 20% do schedule WSD é, de fato, uma fase — e a
literatura de mid-training sugere enriquecê-la com dado de alta qualidade. **Não fazer nisto
nesta rodada:** mudaria simultaneamente dados e schedule e destruiria a medição de β, que é o
propósito do run (`analise-runs.md` §6.4/§7). Anotar como experimento seguinte.

**Como embaralhar, concretamente.** Duas camadas, porque uma só não basta:

1. **Mistura por documento na Fase B.** Para cada documento de saída, sortear a fonte com as
   probabilidades da §1.1 (`random.choices`, `seed=1337`). Isso garante que as proporções
   valem **localmente em qualquer janela do arquivo**, não só no total. Importa muito aqui:
   é o que permite treinar o BPE numa amostra do começo (se um dia retreinarmos) e é o que
   garante que uma rodada interrompida em 60% viu a mistura, não 60% de GigaVerbo.
2. **Buffer de shuffle de 20 mil documentos** por cima. Cada `part` tem localidade forte
   (mesmo domínio, mesmo crawl, mesmo gerador, mesmo autor). ~70 MB de RAM.

Sortear a fonte por documento e não por token é intencional: por token exigiria conhecer o
tamanho antes de ler. O viés resultante é de segunda ordem e é **medido e corrigido**: a
Fase B contabiliza tokens reais por fonte e ajusta as probabilidades a cada 10 M tokens para
convergir aos alvos da §1.1. As proporções finais reais vão em `meta.json`.

### 4.2 Amostragem determinística dentro de cada fonte

Nunca "os primeiros N documentos". O parquet tem localidade: os primeiros 174 k registros de
`historinhas` provavelmente compartilham tema e personagem; os primeiros arquivos de
`Portuguese-PD` são 2.000 livros de um sorteio, mas lidos em ordem de digitalização.

Regra: aceitar o documento se `xxhash64(id_ou_texto) % 10_000 < taxa * 10_000`, com `taxa`
igual à fração da fonte que precisamos. É determinístico (o mesmo build dá o mesmo corpus),
não exige estado, distribui uniformemente por todo o dataset e **é estável quando o alvo
cresce**: subir a taxa de 3,7% (Etapa 1) para 15% (Etapa 2) mantém os documentos da Etapa 1
dentro do conjunto. Para as fatias em que a taxa é minúscula (0,142% do gv2), o custo é que
ainda assim varremos o stream — mas isso já estava orçado em §2.3.

### 4.3 `val_frozen` sem vazamento — a checagem concreta

`plano-avaliacao.md` §2.2 exige `eval/val_frozen.txt` (2 MB, SHA-256 registrado) como texto,
não como `.bin`, e §4 item 6 já registra o risco: *"`data_big/corpus.txt` provavelmente
compartilha fontes com o GigaVerbo"*. Com a mistura nova o risco deixa de ser provável e
passa a ser **certo**: o crawl do GigaVerbo-v2 contém espelhos de Wikipedia, e o `val_frozen`
sai de Wikipedia.

Sem tratamento, o BPB reportado é otimista por memorização e a série histórica inteira do
projeto fica sem valor. Lee et al. mediram **>4% de sobreposição treino-teste** em corpora
padrão — não é um risco hipotético.

**Procedimento, em quatro passos:**

**Passo 1 — extrair com proveniência declarada.** `val_frozen.txt` = bytes
`[760.000.000, 762.000.000)` de `data_big/corpus.txt`, alinhados ao próximo `\n\n`. A fatia
de Wikipedia do corpus é limitada a `[0, 700.000.000)`. As faixas são **disjuntas por
construção, gravadas em `meta.json` e verificáveis por qualquer um**. Isso resolve o
autovazamento (fonte 4 contra ela mesma) sem custo nenhum.

**Passo 2 — índice do `val_frozen`.** Segmentar em parágrafos, normalizar (`casefold`,
colapso de espaço, remoção de pontuação), descartar os com <200 caracteres, hashear com
xxhash64. Resultado: **~700–1.100 hashes**, um `set` de **~70 KB**. Gravar o SHA-256 do
arquivo e a contagem de parágrafos em `meta.json`.

**Passo 3 — quarentena durante a Fase A, em todas as seis fontes.** Todo parágrafo colhido
já é normalizado e hasheado para o Bloom de dedup (§3.2). **Reusar exatamente esse hash**
para uma consulta extra no set do `val_frozen`. Se um documento contiver **≥1** parágrafo do
`val_frozen`, o documento inteiro é descartado e `docs_vazamento[fonte]` incrementa.
**Custo marginal: um lookup em `set` por parágrafo — ~0 s.**

Descartar o documento inteiro, não o parágrafo, é deliberado: um documento que reproduz um
parágrafo da validação quase certamente reproduz os vizinhos com pequenas variações que o
hash exato não pega.

**Passo 4 — auditoria depois do build, e é ela que vale como prova.** Segunda passada
independente sobre os `parts/` finais, contando quantos parágrafos do `val_frozen` aparecem.
Critério de aceitação: **zero**. Não zero ⇒ o build falha e não gera `.bin`. Custo: ~2 min
sobre 580 MB comprimidos. Isso é uma verificação *do artefato*, não *do processo* — pega
inclusive um bug no passo 3.

Registrar em `meta.json`: `val_frozen_sha256`, `val_frozen_paragrafos`,
`docs_descartados_vazamento` **por fonte** e `auditoria_vazamento_hits: 0`. A contagem por
fonte é, ela mesma, a medição da contaminação: se `gigaverbo-v2` descartar 3% dos documentos
por conter Wikipedia, esse é o número a citar quando alguém comparar o Pocket-LM com o
Tucano — que treinou no GigaVerbo.

**Opcional, mais forte, só se sobrar tempo:** contenção por 13-gramas ao estilo GPT-3/Llama,
amostrando 1 a cada 16 n-gramas para caber em ~60–100 s de CPU. O parágrafo exato entrega
~95% do valor por ~5% do custo; fica como reforço, não como requisito.

---

## 5. Especificação do código — `bpe_prepare.py` v2

**Não escrever código a partir daqui: isto é o contrato.** Formato de saída, caminhos e
dtype **não mudam** — `train_gpu.py`, `train_night.py`, `eval.py`, `quantize.py` e
`model_export.py` continuam funcionando sem tocar em uma linha.

### 5.1 Variáveis de ambiente

| Var | Default | Papel |
|---|---|---|
| `CORPUS_TARGET_TOKENS` | `112000000` | **novo knob principal.** Substitui `CORPUS_META_MB` como fonte da verdade |
| `CORPUS_MIX` | `A` | `A` \| `C` \| `wiki` \| caminho para um JSON de mistura |
| `BYTES_PER_TOKEN` | `4.42` | constante de orçamento (§0.1). Só planeja; o real é medido |
| `TOKENIZER_JSON` | `data_big/tokenizer.json` | **se existir, o BPE NÃO é retreinado** (§0.2) |
| `VAL_TOKENS` | `4000000` | substitui o split 95/5 |
| `VAL_FROZEN` | `eval/val_frozen.txt` | vazio desliga a quarentena (só para debug) |
| `PARTS_DIR` | `$DATA_DIR/parts` | onde ficam os `.txt.gz` |
| `SEED` | `1337` | amostragem e shuffle |
| `PD_DOC_CAP_BYTES` | `100000` | cap por livro do `Portuguese-PD` |
| `HF_RETRIES` | `5` | retries com backoff por fonte |
| `DATA_DIR`, `VOCAB_SIZE`, `TOK_TRAIN_MB` | como hoje | inalterados |
| `CORPUS_META_MB` | — | **alias legado**: se setado e `CORPUS_TARGET_TOKENS` não, converte por `BYTES_PER_TOKEN` e avisa em `stderr` |

O alias legado não é cortesia: `submit_train.py` documenta `--env CORPUS_META_MB=80` no
próprio docstring. Quebrar isso quebra o histórico de comandos.

### 5.2 Estrutura

```
bpe_prepare.py
  FonteSpec  (dataclass)  nome, repo, config, split, campo, filtro, peso,
                          doc_cap_bytes, licenca, local_path
  mixes.json              as misturas A / C / wiki, versionadas junto do código
                          -> ACRESCENTAR a BUNDLE em submit_train.py (1 linha)

  carrega_mistura(nome)             -> list[FonteSpec]
  orcamento(specs, alvo_tokens)     -> dict[fonte, bytes_alvo]   # peso * alvo * BYTES_PER_TOKEN
  normaliza(txt)                    -> str                        # regex atual + NFC + ctrl
  Dedup                             # set[int] de docs + Bloom(bytearray) de parágrafos
    .doc_novo(h) -> bool
    .paragrafos_novos(txt) -> (bool_ok, n_dup, bool_vazou)
  indice_val_frozen(path)           -> set[int]                   # ~1k hashes, §4.3 passo 2
  colhe(spec, bytes_alvo, dedup, sink) -> EstatFonte              # Fase A, UMA fonte
  mistura(parts, alvo_tokens, seed) -> Iterator[str]              # Fase B, doc a doc
  tokeniza(iter_docs, tok, alvo)    -> EstatTok                   # escreve val/train .part
  audita_vazamento(parts, idx)      -> int                        # §4.3 passo 4
  verifica(estat)                   -> None | raise               # §5.4
  escreve_meta(estat)               -> None
  main()
```

`treina_tokenizer()` **permanece**, intocada, mas só é chamada se `TOKENIZER_JSON` não
existir. `baixa_corpus()` some, substituída por `colhe()`.

### 5.3 Formato de saída e as três mudanças

Iguais a hoje: `data/train.bin`, `data/val.bin`, `numpy.uint16` cru, sem cabeçalho, lido por
`np.memmap`. Três mudanças, todas necessárias:

1. **`all.bin` deixa de existir.** `val.bin.part` é escrito primeiro, dos primeiros
   `VAL_TOKENS` tokens do fluxo misturado; o resto vai para `train.bin.part`. Elimina o pico
   de disco de 2× **e** elimina o artefato que hoje está truncado.
2. **Corte por documento, nunca no meio de um.** O split de hoje corta o array de ids em
   `int(total*0.95)` — no meio de uma frase, e com o mesmo documento podendo cair dos dois
   lados. Vazamento treino/validação por construção. A v2 fecha o `val.bin` no fim do
   documento corrente. E `VAL_TOKENS=4 M` absoluto substitui os 5%, que em 420 M dariam 21 M
   de tokens de validação — absurdo para uma métrica de acompanhamento.
3. **`<|endoftext|>` (id 0) entre documentos.** O tokenizer já tem o token especial e o
   código de hoje nunca o emite: os documentos são separados só por `\n\n` no texto. Com uma
   fonte só isso passava; com seis registros diferentes colados sem fronteira, o modelo
   aprende transições espúrias entre uma sentença jurídica e uma historinha infantil. Custo:
   ~612 k tokens = **0,15%** do corpus. *Ressalva honesta:* é uma diferença a mais em relação
   ao run A. É pequena e ortogonal a `N`, mas anotar em `meta.json` (`eot_entre_docs: true`)
   para que a comparação seja auditável.

### 5.4 Progresso, interrupção e integridade

Esta subseção existe por causa do modo de falha de hoje: **um OOM no meio da tokenização
deixou `all.bin` truncado em 22,3%, e o `main()` na execução seguinte teria aceitado o
resultado como bom**, porque só testa `os.path.exists("train.bin")`. O sintoma ficou visível
apenas quando alguém dividiu 800 MB por 40,37 M tokens e achou **19,82 bytes/token** contra
os ~4,3 esperados — um outlier de **4,6×** que ficou dias em disco sem ser notado.

Cinco mecanismos. Os dois primeiros sozinhos já tornam a falha silenciosa impossível.

**(1) Escrever em `.part` e renomear atomicamente.**
Toda saída vai para `train.bin.part` / `val.bin.part`. Só depois de `verifica()` passar vem
`os.replace()` para o nome final, seguido de `fsync` do diretório. Um processo morto deixa
`.part` — que `main()` **recusa** e apaga. É a correção mais barata do documento: um arquivo
truncado nunca chega a se chamar `train.bin`.

**(2) `bytes/token` monitorado continuamente, não no fim.**
A cada 10 M tokens escritos, uma linha JSON em `stdout`:
```json
{"fase":"tokenizando","fonte":"gigaverbo-v2-edu","docs":142033,
 "bytes_lidos":640118272,"tokens":150000000,"bytes_por_token":4.267,
 "alvo_tokens":420000000,"pct":35.7,"eta_s":812}
```
`verifica()` exige `3.6 <= bytes_por_token <= 5.2`. **A falha de hoje daria 19,82 e seria
abortada no primeiro relatório de 10 M tokens**, ~30 s depois de começar, em vez de virar um
arquivo ruim consumido em silêncio. E o `alvo_tokens`/`pct` só existem porque a v2 sabe o
alvo antes de começar (§5.1) — o código de hoje não sabe, e por isso não podia detectar nada.

**(3) `progress.json` com fsync.**
Escrito a cada relatório: `{fase, fonte, docs, bytes_lidos, tokens, ts}`. Um kill deixa o
último ponto visível em disco — diagnóstico sem depender de o log ter sido capturado.

**(4) `verifica()` — as assertivas, todas obrigatórias:**

| # | Assertiva | O que pega |
|---|---|---|
| a | `3.6 <= bytes_lidos/tokens <= 5.2` | **o truncamento de hoje** (19,82) |
| b | `getsize(train.bin.part) + getsize(val.bin.part) == 2 * tokens_escritos` | escrita curta, flush parcial, disco cheio |
| c | `tokens_escritos >= 0.97 * CORPUS_TARGET_TOKENS` | fonte que secou, stream cortado |
| d | cada fonte entregou `>= 0.90 *` seu orçamento, **nomeando** a que faltou | filtro errado, dataset fora do ar |
| e | cada `part` foi lido até o EOF do gzip (`ISIZE` bate com os bytes lidos) | gzip truncado, leitura interrompida |
| f | `max(ids)` numa amostra de 1 M tokens de cada `.bin` `< vocab_size` | descasamento tokenizer/bin |
| g | `auditoria_vazamento_hits == 0` (§4.3 passo 4) | vazamento para o `val_frozen` |
| h | `2 <= taxa_dedup_% <= 8` — **aviso**, não erro | dedup desligado (0%) ou filtro errado (>8%) |

Falha em (a)–(g) ⇒ os `.part` **não** são promovidos, `exit != 0`, e nada de bom é
sobrescrito.

**(5) `main()` idempotente de verdade.**
Hoje o short-circuit é `exists(train.bin) and exists(TOKJSON)`. A v2 exige, além disso:
`meta.json` presente, `tokens` do `meta.json` batendo com `getsize(train.bin)/2 +
getsize(val.bin)/2`, e `meta["target_tokens"] >= CORPUS_TARGET_TOKENS`. Se qualquer uma
falhar, refaz. Se o alvo subiu (112 M → 420 M), as `parts/` são **reaproveitadas e
estendidas** — o `set` de ids já colhidos evita recolher e o download da Etapa 2 fica
reduzido ao delta.

`os._exit(rc)` no fim **fica** (o streaming do HF deixa threads penduradas), mas só depois do
`os.replace` e do `fsync`. Hoje ele roda depois de um `os.remove(all.bin)` sem fsync — em
teoria, mais uma janela de arquivo inconsistente.

### 5.5 `meta.json` v2

```json
{
  "versao": 2,
  "mistura": "A",
  "target_tokens": 420000000,
  "tokens": 421337012,
  "train_tokens": 417337012,
  "val_tokens": 4000000,
  "vocab_size": 16384,
  "tokenizer_sha256": "...",
  "tokenizer_retreinado": false,
  "eot_entre_docs": true,
  "bytes_texto": 1798412903,
  "bytes_por_token": 4.268,
  "bytes_por_token_planejado": 4.42,
  "docs": 611402,
  "seed": 1337,
  "fontes": [
    {"nome":"gigaverbo-v2-edu","repo":"Polygl0t/gigaverbo-v2","config":"default",
     "filtro":"edu_int_score>=3 and toxic_int_score==1","licenca":"ODC-By + CC ToU",
     "peso_alvo":0.40,"peso_real":0.398,"tokens":167712000,"docs":196880,
     "docs_dup":4102,"docs_vazamento":5913}
  ],
  "val_frozen":{"path":"eval/val_frozen.txt","sha256":"...","paragrafos":947,
                "faixa_bytes_origem":[760000000,762000000],
                "faixa_bytes_treino":[0,700000000],
                "auditoria_hits":0},
  "dedup":{"metodo":"xxhash64 doc + bloom paragrafo m/n=16 k=3",
           "docs_removidos":18744,"taxa_pct":2.98,"bloom_fp_rate":0.0017},
  "git_sha":"...","data":"2026-07-27T.."
}
```

`bytes_por_token`, `auditoria_hits` e `taxa_pct` são os três números que qualquer pessoa
pode conferir em 10 segundos para saber se o corpus presta.

---

## 6. Ordem de execução com custos

Tudo sob `heavy.sh` — `flock` contra o cron noturno e `MemoryMax=2200M` num scope próprio.
O OOM de hoje foi o treino do cron (1,9 GB) e a tokenização (1,3 GB) colidindo numa máquina
de 3,8 GB; `heavy.sh` existe exatamente para isso. **Nunca rodar `bpe_prepare.py` direto.**

### 6.1 Etapa 1 — mínimo viável, 112 M tokens únicos (bancada)

| # | Etapa | Onde | Tempo | Pico RAM | Pico disco |
|---|---|---|---|---|---|
| 1.0 | Implementar `bpe_prepare.py` v2 + `mixes.json` | bancada | **4–6 h de engenharia** | — | — |
| 1.1 | Extrair `eval/val_frozen.txt` + SHA-256 | bancada | ~1 min | 50 MB | +2 MB |
| 1.2 | Fase A: wikipedia local | bancada | ~1 min | 300 MB | +20 MB gz |
| 1.3 | Fase A: historinhas | bancada | 3–6 min | 800 MB | +23 MB gz |
| 1.4 | Fase A: gv2-synth | bancada | 3–6 min | 1,2 GB | +31 MB gz |
| 1.5 | Fase A: Portuguese-PD | bancada | 2–5 min | 1,2 GB | +8 MB gz |
| 1.6 | Fase A: gv2 edu + web (passada única) | bancada | **6–15 min** | 1,6 GB | +74 MB gz |
| 1.7 | Auditoria de vazamento | bancada | ~1 min | 200 MB | — |
| 1.8 | Fase B: mistura + tokenização (495 MB) | bancada | **2–4 min** | 400 MB | +224 MB |
| 1.9 | `verifica()` + `os.replace` + `meta.json` | bancada | ~10 s | 100 MB | — |
| | **Total de máquina** | | **~20–40 min** | **< 1,7 GB** | **~380 MB** |

### 6.2 Transporte e treino (Etapa 1)

| # | Etapa | Onde | Tempo | Custo |
|---|---|---|---|---|
| 2.0 | Criar data-connection `pocket-lm-data` (uma vez) | Lightning UI | ~5 min | 0 |
| 2.1 | `gzip` dos bins (224 MB → ~140 MB) + upload | bancada | 3–8 min | 0 |
| 2.2 | `path_mappings` + guard no `run_gpu.sh` (2 linhas) | bancada | ~15 min | 0 |
| 2.3 | **Correção de MFU (SDPA + QKV fundido)** | bancada | 1–2 h eng. | **0 — vale 0,471 nat** |
| 2.4 | Run T4, `N=33,87 M`, 391 M tokens, ~11.900 steps | GPU | ~57 min | **~1,0 unidade** |

Ordem obrigatória, de `analise-runs.md` §6.5: **MFU primeiro, corpus segundo, GPU por
último.** A MFU compra 0,471 nat de graça; sem ela 1 unidade de T4 rende 0,075 nat e o run
não vale ser disparado.

### 6.3 Etapa 2 — 420 M tokens únicos (bancada)

Mesmo código, `CORPUS_TARGET_TOKENS=420000000`. As `parts/` da Etapa 1 são estendidas.

| # | Etapa | Tempo | Pico |
|---|---|---|---|
| 3.1 | Fase A, delta das 5 fontes remotas (~1,5 GB de download novo) | **20–40 min** | 1,6 GB RAM |
| 3.2 | Auditoria de vazamento sobre 580 MB de gz | ~3 min | 200 MB |
| 3.3 | Fase B: mistura + tokenização de 1,86 GB @ ~3,3 MB/s | **8–12 min** | **1,42 GB disco** |
| 3.4 | `verifica()` + promoção | ~30 s | — |
| 3.5 | `gzip` + upload de 840 MB → ~500 MB | 10–30 min | — |
| | **Total** | **~45–90 min** | |

A taxa de 3,3 MB/s de tokenização não é chute: é o que a retokenização concluída hoje às
10:36 entregou (96 M tokens em ~2 min de relógio, 2 vCPU).

### 6.4 Caminho mais curto até ≥98 M tokens únicos

`prune-and-distill.md` §5.5 chama isto de *"o maior risco de desperdiçar a rodada"*, e
`analise-runs.md` §6.4 exige ≥98 M para as 4,0 épocas da config recomendada.

**O caminho mais curto tem duas linhas, e não requer o código v2:**

`data_big/` já tem **202,6 M tokens únicos** de Wikipedia-pt (retokenização concluída hoje).
Isso **já satisfaz o critério numérico** de ≥98 M — o `train.bin` de 385 MB está pronto agora.

**E é a coisa errada a fazer.** O critério de ≥98 M nunca foi sobre a contagem; era o proxy
de que o modelo veria dado suficiente para não memorizar. 202,6 M tokens de uma fonte só
mantêm intacto o problema que originou tudo — registro único, molde enciclopédico, zero
diálogo, zero código, zero narrativa fechada. Trocar 27 M por 202 M de Wikipedia compra
parte de −0,546 nat do eixo D e **nada** do eixo de diversidade.

**Portanto:** os 202,6 M servem de **rede de segurança** (se a Etapa 1 falhar às 23h, dispara-se
o run com eles em vez de perder a noite), e não de plano. O plano é a Etapa 1: 20–40 min de
bancada depois do código pronto.

---

## MENOR CAMINHO ATÉ TREINAR

Sequência exata, do estado de agora até um checkpoint melhor que o run A. Tempos de máquina;
o de engenharia está marcado.

```bash
cd /opt/pocket-lm
```

| # | Comando / etapa | Tempo | Bloqueia? |
|---|---|---|---|
| **0** | **Congelar a validação antes de qualquer coisa.** Extrair `eval/val_frozen.txt` dos bytes `[760 M, 762 M)` de `data_big/corpus.txt` (alinhado a `\n\n`), gravar SHA-256. Feito depois do corpus, não há como provar que não vazou. | **2 min** | sim |
| **1** | **Correção de MFU** — SDPA + QKV fundido em `model.py` (`receita-treino.md` §1/§2). Zero custo de GPU, **vale 0,471 nat**, mais que tudo que já se comprou com dinheiro. | **1–2 h eng.** | sim |
| **2** | Escrever `bpe_prepare.py` v2 + `mixes.json` conforme §5. Sem retreino de BPE, sem `all.bin`, `.part` + `verifica()`. | **4–6 h eng.** | sim |
| **3** | `CORPUS_MIX=A CORPUS_TARGET_TOKENS=112000000 DATA_DIR=data_mix MEMCAP=2200M WAIT=1800 ./heavy.sh venv/bin/python bpe_prepare.py` | **20–40 min** | sim |
| **4** | Conferir `data_mix/meta.json`: `bytes_por_token` em 4,0–4,6 · `auditoria_hits == 0` · `taxa_pct` em 2–8% · 6 fontes com `peso_real ≈ peso_alvo`. **Se `bytes_por_token` sair fora, parar.** | **1 min** | sim |
| **5** | Criar a data-connection `pocket-lm-data`; `gzip` dos bins (~140 MB) e upload via `Teamspace.upload_folder`. | **10–20 min** | não* |
| **6** | `path_mappings={"/work/data": "pocket-lm-data"}` em `submit_train.py` + guard no `run_gpu.sh` (§2.5). | **15 min eng.** | não* |
| **7** | `python submit_train.py --name d-112m-t4 --machine T4 --interruptible --env TRAIN_BUDGET_S=3300 --env MAX_STEPS=11900` — `N=33,87 M`, 391 M tokens processados, 3,49 épocas. | **~57 min · ~1,0 unidade** | — |
| **8** | `python fetch_ckpt.py --run d-112m-t4`, depois BPB e CALAME-NLL sobre o **mesmo** `val_frozen.txt`, `stride=256` (`plano-avaliacao.md` §7). | **~40 min CPU** | — |
| **9** | Só então: `CORPUS_TARGET_TOKENS=420000000` (§6.3) para habilitar o experimento de 4 braços de `prune-and-distill.md` §5. | **45–90 min** | — |

\* Os passos 5–6 são puláveis **uma vez**: sem eles, `run_gpu.sh` monta o corpus dentro do
job (~8–15 min de GPU paga, ~0,25 unidade). Vale pular se o objetivo for disparar hoje;
não vale como regime.

**Até o primeiro checkpoint novo:** ~5–8 h de engenharia + ~2 h de máquina + **1,0 unidade**.
Previsão de `analise-runs.md` §6.4: **val_loss ~3,08 antes do cooldown, 2,63–2,83 depois** —
contra 3,63 do run A e 3,51 do run B, este último por 3,14 unidades.

**Os dois erros que anulam tudo:**

1. **Disparar GPU antes do passo 1.** A MFU vale 4× o que o run B inteiro comprou. Sem ela a
   rodada rende 0,075 nat e queima uma unidade.
2. **Extrair o `val_frozen` depois de montar o corpus.** Feito nessa ordem, não há prova de
   ausência de vazamento, e toda a série histórica de BPB do projeto nasce sem valor.

---

## Fontes

**Datasets**
- GigaVerbo-v2: https://huggingface.co/datasets/Polygl0t/gigaverbo-v2
- GigaVerbo-v2 Synth: https://huggingface.co/datasets/Polygl0t/gigaverbo-v2-synth
- Historinhas: https://huggingface.co/datasets/Boakpe/historinhas
- Wikipedia (Wikimedia): https://huggingface.co/datasets/wikimedia/wikipedia
- Portuguese-PD: https://huggingface.co/datasets/PleIAs/Portuguese-PD
- FineWeb-2: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2

**Papers**
- Deduplicating Training Data Makes Language Models Better (Lee et al., 2021): https://arxiv.org/abs/2107.06499
- Scaling Laws and Interpretability of Learning from Repeated Data (Hernandez et al., 2022): https://arxiv.org/abs/2205.10487
- Scaling Data-Constrained Language Models (Muennighoff et al., 2023): https://arxiv.org/abs/2305.16264 · https://huggingface.co/papers/2305.16264
- CLIMB: Curriculum Learning for Infant-inspired Model Building: https://arxiv.org/pdf/2311.08886
- Less is More: Cognitively-Plausible Curriculum Learning for Small-Scale LMs: https://arxiv.org/abs/2410.22886
- TinyStories (Eldan & Li, 2023): https://arxiv.org/pdf/2305.07759

**Ferramentas / pipelines**
- datablations (código do Muennighoff et al.): https://github.com/huggingface/datablations
- FineWeb-2 pipeline: https://github.com/huggingface/fineweb-2
- datatrove (MinHash usado pelo GigaVerbo-v2): https://github.com/huggingface/datatrove
- lm-evaluation-harness (EleutherAI): https://github.com/EleutherAI/lm-evaluation-harness

**Documentos internos**
- `docs/corpus-ptbr.md` — levantamento de fontes e licenças
- `docs/analise-runs.md` — §6.4 config recomendada, §6.5 ordem de execução, §0.2 medições
- `docs/vocab-e-destilacao.md` — §A.1 `f(V)` medido
- `docs/plano-avaliacao.md` — §2.1 BPB, §2.2 `val_frozen`, §7 próximo passo
- `docs/prune-and-distill.md` — §5.5 pré-condições, §RECOMENDAÇÃO item 2
- `docs/gpu-cron.md` — `path_mappings`
