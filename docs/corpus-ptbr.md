# Corpora em português para treinar um LM pequeno do zero

Levantamento feito em 27/07/2026 para o Pocket-LM (GPT estilo nanoGPT, 33M–218M params,
BPE byte-level 16k, rodadas de GPU de 25–50 min). Hoje o corpus é só
`wikimedia/wikipedia` `20231101.pt`.

> **Correção (27/07/2026):** o diagnóstico deste documento continua exatamente certo — o
> problema é **distribuição, não tamanho** — e ficou mais nítido: o corpus de Wikipedia-pt foi
> re-tokenizado e hoje tem **202.630.887 tokens únicos** (800 MB, 3,948 bytes/token), contra
> os 27 M com que todos os outros documentos calculam. **Volume deixou de ser o gargalo;
> registro continua sendo.** O alvo prático da Mistura C passa a ser **≥400 M tokens únicos
> com diversidade de registro**, e a Wikipedia cai de 100% para ~12% da mistura como âncora
> factual. Restrição operacional nova: **6,1 GB de disco livre** — montar em streaming,
> escrever o `.bin` direto, nunca materializar o texto todo. Ver `PLANO.md` §1.4 e §3 item 3.

O problema com Wikipedia-pt sozinha não é tamanho, é **distribuição**: um único registro
(enciclopédico, terceira pessoa, definição-primeiro), frases longas e altíssima repetição
de padrão de abertura ("X é um/uma ... localizado em ..."). Um modelo de 33M params gasta
sua capacidade inteira aprendendo esse molde. Para modelos pequenos, o resultado é
dominado por **diversidade de registro + limpeza**, não por número de parâmetros — é a
lição do TinyStories, do FineWeb-Edu e das ablações do próprio GigaVerbo-v2.

---

## 1. Orçamento de tokens (o que cabe numa rodada)

| Modelo | Params | Chinchilla (~20 tok/param) | Alvo prático (over-training) |
|---|---:|---:|---:|
| pequeno | 33M | 0,66B | 0,8–1,5B tokens |
| médio | 110M | 2,2B | 2–3B tokens |
| grande | 218M | 4,4B | 3–6B tokens |

**Conclusão de engenharia:** o corpus alvo é de **1,5 a 3 bilhões de tokens**, ou seja
~6–12 GB de texto puro. Isso é 0,5% do que os grandes corpora pt oferecem — ou seja,
**dá para escolher a nata em vez de pegar tudo**. Não existe pressão por volume; existe
pressão por qualidade e por variedade de registro.

---

## 2. Fichas dos datasets

### 2.1 GigaVerbo-v2 — `Polygl0t/gigaverbo-v2` (RECOMENDADO como base)

- **Carregar:**
  ```python
  ds = load_dataset("Polygl0t/gigaverbo-v2", "default", split="train", streaming=True)
  ```
  Configs: `default` (limpo) e `excluded` (o que foi barrado por toxicidade — não usar).
- **Tamanho:** 372.108.576 documentos, **834 GB**, **317,7B tokens** no `default`.
  Dos quais **118,7B tokens têm `edu_int_score >= 3`** (o filtro "educacional" estilo
  FineWeb-Edu).
- **Origem/tipo:** crawl web. Fontes agregadas: mC4-pt, `thegoodfellas/mc4-pt-cleaned`,
  FineWeb-2 pt e snapshots recentes do CommonCrawl (até CC-MAIN-2025-30, corte dez/2025).
  É web, não é literatura — mas é web de registro variado (blog, notícia, tutorial, fórum).
- **Licença:** campo HF = `other`. Na prática **ODC-By v1.0 + CommonCrawl Terms of Use**,
  herdado das fontes. É o mesmo regime de risco de qualquer coisa derivada de CommonCrawl
  (FineWeb, C4, Dolma). Aceitável para treinar modelo próprio; **não** é domínio público e
  não deve ser redistribuído como se fosse.
- **Qualidade/limpeza:** o melhor do levantamento. Pipeline completo do FineWeb-2
  implementado com `datatrove`: extração Trafilatura (precisão sobre recall), URL blocklist,
  FastText FT176 + GlotLID para LID, filtros heurísticos FineWeb-2/MassiveText ajustados
  ao português, correção de encoding, remoção de PII, **dedup MinHash** (14 buckets, 8 hashes,
  5-gramas, xxHash). Por cima disso, dois classificadores treinados a partir de anotações
  do Qwen2.5-32B-Instruct:
  - `edu_score` / `edu_int_score` (1 = não-educacional, 5 = altamente educacional)
  - `toxic_score` / `toxic_int_score` (1 = não-tóxico, 5 = altamente tóxico)
  Documentos com <50 tokens e com toxicidade >3 já foram removidos do `default`.
- **Por que importa aqui:** as colunas `edu_int_score` e `token_count` vêm **no próprio
  registro**, então dá para filtrar durante o streaming sem rodar classificador nenhum.
  Isso resolve o problema clássico de "web crawl é lixo" sem custo de GPU.
- **Gated:** não. Sem token HF.
- URL: https://huggingface.co/datasets/Polygl0t/gigaverbo-v2

### 2.2 GigaVerbo-v2 Synth — `Polygl0t/gigaverbo-v2-synth` (RECOMENDADO)

- **Carregar:**
  ```python
  ds = load_dataset("Polygl0t/gigaverbo-v2-synth", "default", split="train", streaming=True)
  ```
- **Tamanho:** 11.237.546 documentos, **26,6 GB**, **9,32B tokens**.
- **Origem/tipo:** 100% **sintético**, gerado por Qwen2.5-32B/14B/7B-Instruct a partir de
  sementes reais (estilo Cosmopedia). Conteúdo: textos didáticos, tutoriais, artigos
  acadêmicos, posts de blog, resoluções matemáticas passo a passo, tutoriais de programação
  e conversas sintéticas. Distribuição das sementes:
  Cosmopedia-v2 (1,67B tok), FineMath (1,36B), Wikipedia (1,00B), LegalPT (0,99B),
  FineWeb-Edu (0,98B), CodeParrot (0,79B), BlogSet-BR (0,67B), StarCoder (0,56B),
  Teses CAPES (0,37B), BDTD (0,37B), SciELO (0,29B), **Historinhas (0,18B)**,
  FinePersonas (0,085B), Baixe Livros / domínio público (0,0097B), SEP (0,0024B).
- **Licença:** o card não declara licença explícita. Os geradores (família Qwen2.5-Instruct)
  são Apache-2.0, o que na prática não restringe o uso das saídas; as sementes são
  ODC-By/CC. Risco baixo, mas **não é "sem licença declarada = domínio público"** —
  registrar como pendência se algum dia o modelo for comercializado.
- **Qualidade:** altíssima densidade de texto bem-formado. É o equivalente pt do
  Cosmopedia. As ablações do próprio projeto mostram ganho grande em benchmark ao
  adicionar os 46B tokens educacionais + sintéticos sobre o GigaVerbo v1.
- **Gated:** não.
- URL: https://huggingface.co/datasets/Polygl0t/gigaverbo-v2-synth

### 2.3 Historinhas — `Boakpe/historinhas` (o "TinyStories em português")

- **Carregar:**
  ```python
  ds = load_dataset("Boakpe/historinhas", split="train", streaming=True)  # campo: "texto"
  ```
- **Tamanho:** 1.255.240 histórias, **0,79 GB** em parquet (~2 GB de texto,
  aproximadamente **0,4–0,5B tokens**).
- **Origem/tipo:** **sintético**, inspirado declaradamente no TinyStories. Histórias curtas,
  vocabulário simples, narrativa completa com início/meio/fim. Colunas: `titulo`, `tema`,
  `personagem`, `moral`, `cenario`, `tonalidade`, `texto`.
- **Licença:** **Apache-2.0**. É a licença mais limpa de todo o levantamento.
- **Qualidade:** limpo por construção (gerado, não crawleado). Cuidado: por ser gerado com
  campos controlados, tem estrutura repetitiva própria — em dose alta o modelo vira contador
  de historinha infantil. Serve como ~10–20% da mistura, não como base.
- **Por que importa:** é exatamente o mecanismo do paper TinyStories — ensina **coerência
  de discurso e resolução narrativa** com pouquíssimos parâmetros. É o antídoto direto para
  o "modelo que só sabe escrever verbete".
- URL: https://huggingface.co/datasets/Boakpe/historinhas

*(Não existe uma tradução de referência do `roneneldan/TinyStories` para pt no HF.
`Dxniz/TinyStories-Multilingual` tem 15.688 histórias em 28 idiomas — cerca de 560 em pt,
inútil por volume. `Historinhas` é a alternativa real.)*

### 2.4 Wikipedia-pt — `wikimedia/wikipedia`, config `20231101.pt` (manter, mas reduzir)

- **Carregar:**
  ```python
  ds = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)
  ```
- **Tamanho:** 1.112.246 artigos, **1,58 GB** em parquet, ~4–5 GB de texto,
  ordem de **0,7–1,0B tokens**.
- **Origem/tipo:** enciclopédico, dump oficial da Wikimedia já com wikitext resolvido.
- **Licença:** **CC-BY-SA-3.0 + GFDL**. Uso para treino é tranquilo; redistribuir o corpus
  exige atribuição e share-alike.
- **Qualidade:** limpíssimo, mas monótono — é a causa do problema atual. Continua valendo
  como **âncora factual** (nomes próprios, datas, entidades, geografia brasileira), coisa
  que web crawl e sintético não entregam com a mesma densidade.
- **Recomendação:** cair de 100% para ~10% da mistura.
- URL: https://huggingface.co/datasets/wikimedia/wikipedia

### 2.5 Portuguese Public Domain — `PleIAs/Portuguese-PD` (literatura, risco jurídico zero)

- **Carregar:**
  ```python
  ds = load_dataset("PleIAs/Portuguese-PD", split="train", streaming=True)
  ```
- **Tamanho:** 7.840 títulos (monografias e periódicos), **672.197.538 palavras**
  (~0,9–1,0B tokens), **3,3 GB** em parquet distribuídos em 6 arquivos de 2.000 livros cada.
- **Origem/tipo:** livros e periódicos digitalizados, recuperados do Internet Archive e de
  bibliotecas nacionais europeias. Prosa longa, registro literário e histórico.
- **Licença:** **domínio público integral**. Critério aplicado: autor morto há mais de 70
  anos, e por precaução só publicações **anteriores a 1884**. É a única fonte grande do
  levantamento com risco jurídico literalmente nulo.
- **Qualidade:** o ponto fraco é o **OCR** — todo o texto foi transcrito automaticamente,
  com erros de reconhecimento, cabeçalhos e numeração de página não removidos, tabelas e
  layout multi-coluna mal formatados. Os próprios curadores listam a correção de OCR como
  trabalho futuro.
- **Cuidado de língua:** ortografia **pré-1884**, portanto anterior aos acordos ortográficos
  de 1911, 1943, 1971 e 1990. Traz "pharmacia", "elle", "sciencia". Em dose alta contamina
  a ortografia do modelo. Usar no máximo ~5% e, idealmente, só depois de um filtro simples
  de dígrafos arcaicos (`ph`, `th`, `elle`, `-mente` com trema etc.).
- URL: https://huggingface.co/datasets/PleIAs/Portuguese-PD

### 2.6 LegalPT (dedup) — `eduagarcia/LegalPT_dedup` (registro formal, opcional)

- **Carregar:**
  ```python
  ds = load_dataset("eduagarcia/LegalPT_dedup", "parlamento-pt", split="train", streaming=True)
  ```
- **Tamanho total:** 11.946.015 docs no config `all`, **44,6 GB**. Por config:
  `mlp_pt_CJPG` 6,26M docs / 19,9 GB · `tesemo_v2` 1,74M / 12,9 GB ·
  `parlamento-pt` 2,11M / 1,2 GB · `acordaos_tcu` 462k / 1,6 GB ·
  `mlp_pt_eurlex-legislation` 95k / 1,05 GB · `mlp_pt_legal-mc4` 188k / 2,2 GB ·
  `datastf` 310k / 0,85 GB · `iudicium_textum` 153k / 0,37 GB ·
  `mlp_pt_BRCAD-5` 543k / 1,9 GB · `mlp_pt_eurlex-caselaw` 79k / 0,6 GB ·
  `mlp_pt_eurlex-contracts` 8,5k / 0,1 GB.
- **Origem/tipo:** jurídico e legislativo — acórdãos do TCU, jurisprudência do TJSP (CJPG),
  STF, EUR-Lex, transcrições do Parlamento português (`parlamento-pt`, que é **fala
  transcrita**, não peça processual) e teses (`tesemo_v2`).
- **Licença:** não declarada no card. Documentos públicos oficiais no Brasil não são
  protegidos por direito autoral (Lei 9.610/98, art. 8º, IV), então o risco é baixo, mas a
  ausência de declaração explícita é um incômodo. EUR-Lex tem regime próprio de reutilização.
- **Qualidade:** deduplicado (exact hash + MinHash, metadados de dedup preservados nas
  colunas). Muito limpo, mas o jurídico brasileiro é **extremamente repetitivo e formulaico**
  — em dose alta um modelo pequeno aprende a produzir ementa infinita. Se entrar, entrar com
  `parlamento-pt` (fala) e não com `mlp_pt_CJPG` (jurisprudência).
- URL: https://huggingface.co/datasets/eduagarcia/LegalPT_dedup

### 2.7 FineWeb-2, subset português — `HuggingFaceFW/fineweb-2`, config `por_Latn`

- **Carregar:**
  ```python
  ds = load_dataset("HuggingFaceFW/fineweb-2", "por_Latn", split="train", streaming=True)
  ```
- **Tamanho:** 199.771.825 documentos, **275,9 GB** em parquet (~712 GB descomprimido).
  Ordem de 80–100B tokens.
- **Origem/tipo:** crawl web multilíngue (1000+ idiomas), pipeline de referência do HF.
- **Licença:** **ODC-By** (mais explícita e mais confortável que o `other` do GigaVerbo-v2).
- **Qualidade:** LID com GlotLID, dedup **global por idioma** com metadados de tamanho de
  cluster preservados, filtros do FineWeb adaptados (removido `short_line_thr`,
  `char_dup_ratio` ajustado para 0.1), anonimização de PII (e-mail/IP) e correção FTFY.
  Faz **rehydration**: documentos são reamostrados conforme o tamanho do cluster de dedup
  (cluster de 2–3 repete N vezes, de 4 repete 3x, de 5–99 repete 5x, 1000+ volta a 1x).
- **Ressalva:** **não tem score de qualidade educacional**. O GigaVerbo-v2 já engloba o
  FineWeb-2 pt *e* adiciona os classificadores por cima — então usar FineWeb-2 direto é
  estritamente pior aqui, exceto se a licença ODC-By explícita for requisito.
- Variante: `epfml/FineWeb2-HQ` (subconjunto filtrado por qualidade, verificar cobertura pt).
- URL: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2

### 2.8 CulturaX — `uonlp/CulturaX`, config `pt` (NÃO recomendado: gated)

- **Carregar:** exige aceite de termos e `HF_TOKEN`:
  ```python
  ds = load_dataset("uonlp/CulturaX", "pt", split="train", streaming=True, token=True)
  ```
- **Tamanho (pt):** 190.289.658 documentos, **136,9B tokens**, ~68 GB. Total 6,3T tokens
  em 167 idiomas / 16 TB parquet.
- **Origem/tipo:** fusão de mC4 v3.1.0 + todos os OSCAR (20.19, 21.09, 22.01, 23.01),
  limpos e deduplicados. Para pt, mC4+OSCAR23 somam 456,6 GB brutos, reduzidos ~17%.
- **Licença:** "segue estritamente as de mC4 e OSCAR" — ou seja, licença **por referência**,
  sem texto próprio. É a situação jurídica menos clara do levantamento.
- **Qualidade:** boa — LID, filtro por URL, limpeza por métricas, refinamento de documento
  e dedup fuzzy MinHash em nível de documento.
- **Bloqueador prático:** `gated: auto`. Funciona, mas exige token HF dentro do job de GPU,
  o que é exatamente o tipo de dependência que quebra rodada automatizada de madrugada.
- URL: https://huggingface.co/datasets/uonlp/CulturaX

### 2.9 OSCAR-2301 — `oscar-corpus/OSCAR-2301` (INDISPONÍVEL)

- **Tamanho (pt):** 36.062.800 documentos, 15,17B palavras, **105 GB**.
- **Licença:** CC0-1.0 sobre o empacotamento, metadados e anotações. O texto em si vem do
  CommonCrawl e os autores do OSCAR declaram não deter copyright algum sobre ele.
- **Status: acesso suspenso.** O card diz textualmente que foram forçados a suspender o
  acesso temporariamente, implementado como gate com **aprovação manual**, e que **nenhum
  acesso será concedido** até a situação se esclarecer. Na prática, OSCAR está fora.
- Alternativa: o OSCAR-2301 pt já está incorporado, deduplicado, dentro de
  `eduagarcia/CrawlPT_dedup` (config `OSCAR-2301`, 10.888.966 docs, 42,6 GB) e do CulturaX.
- URL: https://huggingface.co/datasets/oscar-corpus/OSCAR-2301

### 2.10 mC4 / C4 em português — `allenai/c4`, config `pt`

- **Carregar:**
  ```python
  ds = load_dataset("allenai/c4", "pt", split="train", streaming=True)
  ```
- **Tamanho:** ~169,4 milhões de documentos estimados (o servidor de datasets reporta
  1.831.606 linhas já convertidas e estima 169.400.662 no total). Ordem de 60–80B tokens.
- **Origem/tipo:** CommonCrawl com limpeza C4 (badwords, linhas sem pontuação final,
  remoção de boilerplate/JavaScript, dedup por linha de 3 sentenças).
- **Licença:** **ODC-By**, declarada explicitamente. Boa.
- **Qualidade:** limpeza de 2020, sem dedup fuzzy em nível de documento e sem filtro
  aprendido. É o ancestral, não o estado da arte. Versão limpa comunitária:
  `thegoodfellas/mc4-pt-cleaned` (1,76M docs, 3,1 GB, Apache-2.0).
- **Veredito:** não usar direto — está inteiro dentro do GigaVerbo-v2, já filtrado.
- URL: https://huggingface.co/datasets/allenai/c4

### 2.11 Corpus Carolina (USP) — `carolina-c4ai/corpus-carolina`

- **Carregar:**
  ```python
  ds = load_dataset("carolina-c4ai/corpus-carolina", taxonomy="leg")  # sem streaming confiável
  ```
- **Tamanho:** ~2,1 milhões de documentos, **15 GB** no total. Por taxonomia:
  `wik` wikis 5,3 GB · `dat` datasets/corpora 4,3 GB · `leg` legislativo 4,2 GB ·
  `jud` judiciário 1,0 GB · `uni` domínios universitários 162 MB · `soc` redes sociais
  49 MB · `pub` obras de domínio público 4,5 MB. Versão atual v2.0.1 ("Bea").
- **Origem/tipo:** **português brasileiro contemporâneo (1970–)** com metadados de
  proveniência e tipologia, curado por linguistas do LaViHD/C4AI-USP. Formato TEI P5 XML.
- **Licença:** **CC-BY-4.0** para os cabeçalhos Carolina. Mas o card avisa: o corpus é
  composto de textos de repositórios digitais diversos, **cujas licenças são múltiplas e
  devem ser observadas** ao usar o corpus. Ou seja, CC-BY cobre a curadoria, não
  necessariamente o conteúdo.
- **Qualidade:** curadoria acadêmica, ótima proveniência, mas **sem dedup fuzzy**. `wik` é
  em boa parte Wikipedia (duplicaria o que já temos) e `pub` é minúsculo (4,5 MB). O
  diferencial real seria `uni` + `soc`, que juntos dão só ~210 MB.
- **Veredito:** valor científico alto, valor prático baixo para esta rodada. O GigaVerbo v1
  já o incorporava (2,1M linhas). Não entra na mistura.
- URL: https://huggingface.co/datasets/carolina-c4ai/corpus-carolina

### 2.12 brWaC — `UFRGS/brwac` / `dominguesm/brwac` (EVITAR por licença)

- **Tamanho:** 3,53 milhões de documentos, 2,68 bilhões de tokens, 5,79 milhões de types
  (versão de janeiro de 2017). Deduplicado em `eduagarcia/CrawlPT_dedup` config `brwac`:
  3.513.588 docs, 11,2 GB.
- **Origem/tipo:** Brazilian Portuguese Web as Corpus, construído no framework WaCky.
  É o corpus web clássico de pt-BR, base do BERTimbau e do PTT5.
- **Licença: PROBLEMA.** Disponível **exclusivamente para fins de pesquisa acadêmica**, com
  compromisso expresso de **não uso comercial**. A PORTULAN chegou a publicar uma variante
  "Albertina PT-BR No-brWaC" treinada sem brWaC justamente para poder oferecer licença mais
  permissiva.
- **Veredito:** dado o requisito de risco jurídico baixo, **não usar**. E como o brWaC está
  dentro do `CrawlPT_dedup` config `all`, evitar também o config `all` desse dataset.
- URL: https://huggingface.co/datasets/UFRGS/brwac

### 2.13 Outros avaliados (fichas curtas)

| Dataset | Tamanho | Tipo | Licença | Veredito |
|---|---|---|---|---|
| `TucanoBR/GigaVerbo` (v1) | 145,3M docs / 411 GB / >200B tok | agregado (mC4, HPLT, CrawlPT, Carolina, Wikipedia, LegalPT, instruct) | `other`, mistura CC0/CC-BY/Apache/**CC-BY-NC-SA** | Superado pela v2. Contém componente **NC** — evitar |
| `bastao/VeraCruz_PT-BR` | 190,3M docs / 373 GB · config `Brazil (BR)` 90,1M docs / 157 GB | CulturaX-pt separado por variante PT vs BR via URL + classificador | herda mC4 + OSCAR | Interessante para **forçar pt-BR puro**, mas herda a licença nebulosa do CulturaX |
| `eduagarcia/CrawlPT_dedup` | 104,9M docs / 170 GB (`all` 52,5M / 81 GB) | OSCAR-2301 + brWaC + CC100, deduplicados | não declarada; **`all` inclui brWaC (NC)** | Usar só config `OSCAR-2301` ou `cc100`, nunca `all` |
| `dominguesm/canarim` | 342,8M docs / ~1 TB | CommonCrawl → Markdown via Trafilatura | **CC-BY-4.0** | Só dedup por URL, sem dedup textual nem filtro de conteúdo nocivo. Bruto demais |
| `Madras1/corpus-ptbr-v2` | ~8,77M docs / ~7,06B tok | 60% real (C4-pt + FineWeb2-pt) + 40% sintético Mistral | **ODC-By 1.0** | Licença limpa, tamanho conveniente. Camada sintética gerada por `mistral-small/medium` — qualidade não auditada externamente |
| `nicholasKluge/Pt-Corpus` | 5,77M docs / 11,5 GB | corpus do TeenyTinyLlama | `other` | Bom tamanho de "corpus inteiro cabe no disco", mas superado pelo GigaVerbo-v2 |
| `Polygl0t/gigaverbo-v2-sft` | 4,09M exemplos / 4,7 GB / ~2,1B tok | instruções em 12 categorias (code, math, reasoning, retrieval...) | **Apache-2.0** | Não é pré-treino. Guardar para uma fase 2 de SFT |
| `thegoodfellas/blogset-br` | 2,04M docs / 3,1 GB | posts de blogs brasileiros | **Apache-2.0** | Registro informal pt-BR com licença limpa. Boa reserva se quiser mais coloquial |
| `eduagarcia/scielo_abstracts` | 1,21M docs / 1,7 GB | resumos científicos SciELO | não declarada | Nicho; já representado via sementes do synth |
| `eduagarcia/capes_teses_dissertacoes` | 1,58M docs / 3,2 GB | resumos de teses CAPES | não declarada | Idem |
| `iara-project/news-articles-ptbr-dataset` | 352k docs / 0,77 GB | notícias pt-BR | não declarada | Pequeno; notícia tem direito autoral vivo. Evitar |
| `PleIAs/common_corpus` | (repo HF expõe só amostra de 69,9k linhas / 0,4 GB) | multilíngue de licença aberta | aberta por construção | A parte pt é essencialmente o `Portuguese-PD` |
| `laion/Project-Gutenberg` | Gutenberg completo, 10 idiomas incl. pt | livros domínio público | domínio público | A fatia pt do Gutenberg é pequena (algumas centenas de títulos). `Portuguese-PD` cobre melhor |
| Portal Domínio Público (MEC) | ~250k obras | acervo brasileiro | misto | **Sem dataset HF pronto e sem API estável.** Exigiria scraping próprio; fora de escopo |

---

## 3. Armadilhas de licença e de acesso (resumo executivo)

**Bloqueadores duros:**
- `oscar-corpus/OSCAR-2301` — acesso suspenso, aprovação manual, nada sendo concedido.
- `UFRGS/brwac` — só pesquisa acadêmica, sem uso comercial. Contamina
  `eduagarcia/CrawlPT_dedup` config `all`.
- `TucanoBR/GigaVerbo` (v1) — contém componente CC-BY-**NC**-SA-4.0.
- `uonlp/CulturaX` — gated (auto), exige `HF_TOKEN` no job.

**Licença limpa de verdade (para o caso de o modelo virar produto):**
- `PleIAs/Portuguese-PD` — domínio público integral.
- `Boakpe/historinhas` — Apache-2.0.
- `thegoodfellas/blogset-br`, `thegoodfellas/mc4-pt-cleaned` — Apache-2.0.
- `Polygl0t/gigaverbo-v2-sft` — Apache-2.0.
- `wikimedia/wikipedia` — CC-BY-SA-3.0 + GFDL (exige atribuição e share-alike se
  redistribuir o corpus; treinar é tranquilo).
- `HuggingFaceFW/fineweb-2`, `allenai/c4`, `Madras1/corpus-ptbr-v2` — ODC-By.

**Zona cinzenta aceitável para modelo próprio:** GigaVerbo-v2 (`other` = ODC-By +
CommonCrawl ToU). É o mesmo regime de FineWeb, C4 e Dolma; se isso for inaceitável, a
mistura alternativa da seção 5 resolve.

---

## 4. RECOMENDAÇÃO: a mistura

### Mistura A — "Padrão" (recomendada), alvo 2,0B tokens

| % | Dataset | Filtro | Tokens |
|---:|---|---|---:|
| **40%** | `Polygl0t/gigaverbo-v2` (`default`) | `edu_int_score >= 3` e `toxic_int_score == 1` | 0,80B |
| **20%** | `Polygl0t/gigaverbo-v2-synth` | nenhum | 0,40B |
| **15%** | `Boakpe/historinhas` | campo `texto` | 0,30B |
| **12%** | `wikimedia/wikipedia` `20231101.pt` | nenhum | 0,24B |
| **8%** | `Polygl0t/gigaverbo-v2` (`default`) | `edu_int_score <= 2` (web "comum") | 0,16B |
| **5%** | `PleIAs/Portuguese-PD` | filtro anti-ortografia-arcaica | 0,10B |

**Por que essas proporções:**

- **40% web educacional** é a espinha dorsal: é onde está a variedade de assunto, de sintaxe
  e de vocabulário moderno. O `edu_int_score >= 3` é o mesmo mecanismo que fez o FineWeb-Edu
  render mais que o FineWeb bruto com menos tokens; aqui ele vem pré-computado, de graça.
- **20% sintético** é a aposta central do documento. Texto gerado por LLM tem densidade de
  informação por token muito maior que web: sem boilerplate, sem menu de navegação, sem
  "aceite nossos cookies", com estrutura argumentativa completa. Para um modelo de 33M–218M
  params, cada token desperdiçado é caro. Cobre também matemática, código e explicação
  didática, que a Wikipedia-pt não cobre.
- **15% Historinhas** é o mecanismo TinyStories: histórias curtas e completas ensinam
  **fechamento de discurso** — o modelo aprende que um texto termina, e como. É o que falta
  hoje. Não passar de 15% ou o modelo vira contador de historinha.
- **12% Wikipedia** cai de 100% para 12% mas continua presente: ela é a única fonte com
  densidade alta de entidades nomeadas, datas e geografia brasileira. Remover totalmente
  degrada a "fatualidade" aparente das amostras.
- **8% de web NÃO-educacional** é intencional e contra-intuitivo. Se todo o corpus for
  filtrado por qualidade educacional, o modelo nunca vê registro informal, frase curta,
  diálogo, gíria, pergunta direta. O corpus vira uma enciclopédia bem-comportada — que é
  exatamente o problema que estamos tentando resolver. 8% de web comum é a dose de "língua
  como as pessoas escrevem".
- **5% domínio público** dá prosa longa e literária (períodos compostos, narração em
  primeira pessoa, diálogo com travessão) e é o único componente com licença
  incontestável. Limitado a 5% por causa do OCR sujo e da ortografia pré-1911.

### Mistura B — "Licença blindada" (se o modelo puder virar produto), alvo 1,5B tokens

| % | Dataset | Licença |
|---:|---|---|
| 35% | `Madras1/corpus-ptbr-v2` (subset `real`) | ODC-By 1.0 |
| 25% | `Boakpe/historinhas` | Apache-2.0 |
| 20% | `wikimedia/wikipedia` `20231101.pt` | CC-BY-SA-3.0 |
| 12% | `thegoodfellas/blogset-br` | Apache-2.0 |
| 8% | `PleIAs/Portuguese-PD` | domínio público |

Perde qualidade em relação à Mistura A (não há score educacional pré-computado em nenhum
componente), mas todo item tem licença declarada e permissiva.

### Mistura C — "Mínimo esforço", alvo 1,0B tokens

70% `Polygl0t/gigaverbo-v2` com `edu_int_score >= 3` + 30% `Boakpe/historinhas`.
Duas chamadas de `load_dataset`, nenhuma gate, e já resolve ~80% da distância entre o
estado atual e a Mistura A. Se houver tempo para uma única mudança, é esta.

---

## 5. Notas de implementação para o job de GPU de ~1h

**1. Não tokenize dentro do job de GPU.** O download+filtragem+BPE de 2B tokens é trabalho
de CPU e rede, e queimá-lo em GPU alugada é desperdício direto. Preparar o `.bin` no x86
(`bpe_prepare.py`) e subir o artefato pronto. O job de GPU deve começar com `np.memmap`
sobre um arquivo local, não com `load_dataset`.

**2. Se for streamar mesmo assim, cuidado com a amplificação de leitura.** Filtrar
`edu_int_score >= 3` descarta ~63% dos tokens lidos (118,7B de 317,7B), então o stream lê
~2,7 GB para cada 1 GB aproveitado. Não é problema no volume que precisamos (0,8B de 118,7B
disponíveis = 0,7% do subset), mas é fatal se alguém tentar varrer o dataset inteiro.

**3. Interleave em vez de concatenar.** Misturar os datasets com pesos, em vez de
concatenar blocos:
```python
from datasets import load_dataset, interleave_datasets

gv   = load_dataset("Polygl0t/gigaverbo-v2", "default", split="train", streaming=True)
edu  = gv.filter(lambda x: x["edu_int_score"] >= 3 and x["toxic_int_score"] == 1)
web  = gv.filter(lambda x: x["edu_int_score"] <= 2)
syn  = load_dataset("Polygl0t/gigaverbo-v2-synth", "default", split="train", streaming=True)
sto  = load_dataset("Boakpe/historinhas", split="train", streaming=True).rename_column("texto", "text")
wik  = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)
pd_  = load_dataset("PleIAs/Portuguese-PD", split="train", streaming=True)

mix = interleave_datasets(
    [edu, syn, sto, wik, web, pd_],
    probabilities=[0.40, 0.20, 0.15, 0.12, 0.08, 0.05],
    seed=1337,
    stopping_strategy="all_exhausted",
)
```
Concatenar blocos causa deriva de distribuição ao longo do treino: o modelo termina o
treino tendo visto só o último bloco recentemente, e a amostra final reflete isso.
Nenhum dos datasets acima é gated, então **nada disso exige `HF_TOKEN`**.

**4. Shuffle com buffer.** `mix.shuffle(seed=1337, buffer_size=10_000)` — cada arquivo
parquet tem localidade forte (mesmo domínio, mesmo crawl, mesmo gerador).

**5. Retreinar o tokenizer BPE.** O BPE 16k atual foi ajustado só em Wikipedia-pt e está
otimizado para vocabulário enciclopédico. Refazer o merge sobre uma amostra da mistura nova
(uns 500 MB bastam) antes de tokenizar o corpus, ou a taxa de compressão cai e o orçamento
de tokens efetivo encolhe.

**6. Deduplicar entre fontes.** Há sobreposição conhecida: as sementes do
`gigaverbo-v2-synth` incluem Wikipedia e Historinhas, e o `gigaverbo-v2` já absorveu
mC4-pt e FineWeb-2-pt. O texto sintético não é cópia da semente (é reescrita), mas um
dedup exato por hash de documento no final da preparação é barato e vale a pena.

---

## 6. Fontes

- GigaVerbo-v2: https://huggingface.co/datasets/Polygl0t/gigaverbo-v2
- GigaVerbo-v2 Synth: https://huggingface.co/datasets/Polygl0t/gigaverbo-v2-synth
- GigaVerbo-v2 SFT: https://huggingface.co/datasets/Polygl0t/gigaverbo-v2-sft
- Tucano 2 (blog): https://huggingface.co/blog/Polygl0t/tucano2
- Tucano 2 Cool (paper): https://arxiv.org/pdf/2603.03543
- Tucano v1 (paper): https://arxiv.org/abs/2411.07854
- GigaVerbo v1: https://huggingface.co/datasets/TucanoBR/GigaVerbo
- Historinhas: https://huggingface.co/datasets/Boakpe/historinhas
- TinyStories (paper): https://arxiv.org/pdf/2305.07759
- TinyStories (dataset): https://huggingface.co/datasets/roneneldan/TinyStories
- TinyStories Multilingual: https://huggingface.co/datasets/Dxniz/TinyStories-Multilingual
- Wikipedia (Wikimedia): https://huggingface.co/datasets/wikimedia/wikipedia
- Portuguese-PD: https://huggingface.co/datasets/PleIAs/Portuguese-PD
- Common Corpus: https://huggingface.co/datasets/PleIAs/common_corpus
- FineWeb-2: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- FineWeb-2 (pipeline): https://github.com/huggingface/fineweb-2
- FineWeb2-HQ: https://huggingface.co/datasets/epfml/FineWeb2-HQ
- CulturaX: https://huggingface.co/datasets/uonlp/CulturaX
- CulturaX (paper): https://huggingface.co/papers/2309.09400
- OSCAR-2301: https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
- C4 / mC4: https://huggingface.co/datasets/allenai/c4
- mC4-pt-cleaned: https://huggingface.co/datasets/thegoodfellas/mc4-pt-cleaned
- Corpus Carolina: https://huggingface.co/datasets/carolina-c4ai/corpus-carolina
- Corpus Carolina (site): https://sites.usp.br/corpuscarolina/
- brWaC: https://huggingface.co/datasets/UFRGS/brwac
- LegalPT (dedup): https://huggingface.co/datasets/eduagarcia/LegalPT_dedup
- CrawlPT (dedup): https://huggingface.co/datasets/eduagarcia/CrawlPT_dedup
- Canarim: https://huggingface.co/datasets/dominguesm/canarim
- VeraCruz PT-BR: https://huggingface.co/datasets/bastao/VeraCruz_PT-BR
- Corpus PT-BR v2: https://huggingface.co/datasets/Madras1/corpus-ptbr-v2
- BlogSet-BR: https://huggingface.co/datasets/thegoodfellas/blogset-br
- SciELO abstracts: https://huggingface.co/datasets/eduagarcia/scielo_abstracts
- Teses CAPES: https://huggingface.co/datasets/eduagarcia/capes_teses_dissertacoes
- Pt-Corpus (TeenyTinyLlama): https://huggingface.co/datasets/nicholasKluge/Pt-Corpus
- Project Gutenberg (LAION): https://huggingface.co/datasets/laion/Project-Gutenberg
- Portuguese-NLP (lista de recursos): https://github.com/ajdavidl/Portuguese-NLP
- Classificador edu pt: https://huggingface.co/Polygl0t/portuguese-bertimbau-large-edu-classifier
- Classificador toxicidade pt: https://huggingface.co/Polygl0t/portuguese-bertabaporu-large-toxicity-classifier
