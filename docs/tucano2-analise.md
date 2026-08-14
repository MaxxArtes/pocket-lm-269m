# Tucano 2 Cool — análise estratégica (pesquisa + crítica agy, 30/07)

Fonte: arXiv 2603.03543 (mar/2026), org HF Polygl0t (projeto alemão BMBF,
Kluge-Corrêa + Lucie Flek), tudo Apache 2.0 / CC-BY 4.0.

## O cenário competitivo real

- Suíte nova: 1 modelo do zero (Tucano2-0.6B-Base, 670M, CALAME 57,61) + 3
  continual-pretrain sobre pesos Qwen3 (0.5B/1.5B/3.7B, CALAME 58,7-61,1 — fora
  das nossas regras como caminho, mas mostra o teto da rota "atalho").
- **Não há modelo novo do zero na faixa 100-500M** — o comparável de faixa segue o
  Tucano-160m antigo (52,3). MAS (correção do agy, acatada): essa comparação
  conforta e engana. O 0.6B deles treinou em **400B tokens**; nós, 269M em 4,6B.
  Nosso gap real é REGIME DE DADOS, não tamanho — estamos severamente subtreinados
  pros padrões de 2026.

## Achado estratégico principal (agy)

**GigaVerbo-v2 Synth é baixável**: 9,3B tokens sintéticos / 11,2M amostras, gerados
com professores Qwen2.5-32B/14B/7B-Instruct, decontaminados por k-gram, públicos.
Nossa regra proíbe pesos de terceiros, não dado sintético de professor — é
exatamente o que já fazemos na fase de conversa. Eles pagaram a conta (73% da
energia do projeto deles); pra nós é download. A ablação deles mostra o valor:
Edu+Synth vs Non-Edu dá ARC-C 34,4 vs 25,6 e HellaSwag 46,0 vs 38,3.
**VERIFICADO (30/07)**: totalmente utilizável.
- Repo próprio: https://huggingface.co/datasets/Polygl0t/gigaverbo-v2-synth —
  11,2M amostras, 26,6GB Parquet, streaming habilitado.
- Licença limpa: Apache 2.0; os professores (Qwen2.5-7B/14B/32B-Instruct) são todos
  Apache 2.0, escolhidos pelos autores exatamente por permitir redistribuição — a
  cláusula "Built with Qwen" só existe nos 3B/72B, não usados.
- Campos: `text`, `seed` (fonte-semente: Cosmopedia, FineMath, LegalPT, StarCoder,
  contos, WikiHow-style, QA múltipla escolha...), `generator` (qual professor),
  `token_count`. Filtrar por `seed` permite dosar QA vs texto corrido na mistura.
- Decontaminação declarada: k-gram (k 8-32) contra benchmarks + filtro de idioma.
- Sem avaliação independente ainda (community vazia) — a evidência é a ablação dos
  próprios autores (Edu+Synth > Synth-only > Non-Edu).
- Pipeline: `load_dataset("Polygl0t/gigaverbo-v2-synth", streaming=True)` +
  tokenizar `text` com nosso BPE 16k, shard a shard — idêntico ao fluxo ClassiCC.

## O que aproveitar (filtrado pelo nosso orçamento e escala)

1. **Decay 1-sqrt no WSD** (custo zero, só código): usamos decay linear; Hägele et
   al. e agora o Tucano 2 apontam 1-sqrt como forma melhor. Trocar no próximo treino.
2. **Corpus sintético deles na nossa mistura** (custo zero + engenharia): candidato
   a stream "premium" ou mistura na próxima rodada grande de dados.
3. **Checkpoints intermediários públicos** (custo zero): branches no HF a cada X
   tokens — reprodutibilidade + build-in-public que atrai avaliação/SFT de graça
   da comunidade.
4. **Engenharia reversa de hiperparâmetros** do llm-foundry deles
   (github.com/Polygl0t/llm-foundry) antes de qualquer rodada paga.

## Descartado (com razão)

- **Tokenizer ~49k** (recomendação do agente, VETADA pelo agy — acatado): em 269M
  com d=1024, ir de 16k pra 49k de vocabulário roubaria ~33M de parâmetros das
  camadas de atenção/MLP só pra embedding. Nosso 16k está certo pra escala.
- **Ablação própria de misturas**: treinar N vezes pra comparar não cabe em
  orçamento de dezenas de dólares — usar as ablações publicadas DELES como guia.
- Nota: AdamW+Muon e WSD já são nossos desde antes — a novidade é só a forma
  1-sqrt do decaimento.

Relacionado: [[pocket-lm-estado-29-07]], docs/corpora-complementares.md,
docs/continued-pretraining-lr.md.
