# Pocket-LM 269M

Modelos de linguagem pequenos em **português do Brasil**, treinados **do zero**, para rodar
**offline no celular** (llama.cpp / GGUF, CPU). Este repositório é a publicação aberta da
família **269M**: bases de pré-treino, receitas de treino, adapters de produto, medições e
lições. Licença **Apache 2.0** — pesos e código.

- Pesos no Hugging Face: coleção **Pocket-LM 269M** (usuário [`Magurofg`](https://huggingface.co/Magurofg))
- Corpus de pré-treino: 12,3 bi de tokens pt-BR **só com licença declarada** (ODC-By/CC0) —
  ver `docs/licencas-e-risco-juridico.md` e `docs/montagem-corpus.md`

## As três bases (pré-treino do zero, 269M/286M parâmetros)

| base | ideia | CALAME-PT | LAMBADA-PT | intenção (ações) | coerência (conversa) |
|---|---|---|---|---|---|
| `generica-v1` | só corpus limpo, nenhum dado de tarefa | **53,13** | **36,85** | 0,0 | 11,5 |
| `conversa-v1` | fase de decay dominada por diálogo | 48,55 | 32,23 | 0,0 | 69,2* |
| `executor-v1` | fase de decay dominada por ações de celular | 49,86 | 35,01 | **86,4 sem SFT** | 34,6 |

\* n=52, juiz binário. Receita e resultados completos de cada base em `bases/*/RECEITA.json`
e `bases/*/RESULTADOS.json`.

O achado central do `executor-v1`: **intenção 86,4% e saída bem-formada 98,2% saem do
pré-treino, sem SFT nenhum** — o SFT só fecha a extração de campos (F1 38→73).

## Tabela de cobertura (3 bases × 4 tratamentos de SFT)

`medicoes/TABELA_COBERTURA.md` — 12 células, mesma régua. As duas lições mais caras:

- **SFT sem replay destrói a língua**: CALAME 53→12, 50→17 (qualquer base).
- **SFT sequencial destrói a tarefa anterior**: intenção 86→44.

## Adapters de produto (SFT com replay 25%)

`medicoes/ANALISE_ADAPTERS.md` — `adapter-app-v1` e `adapter-whats-v1`, SFT completo sobre
`executor-v1`, dose de conversa 7%, replay 25%:

- Replay 25% recupera a língua **sem custar execução**: CALAME 18→43, 5-shot 19→41,
  intenção 87,2→87,4, F1-parada 74,9→75,5.
- Dose de conversa de 7% compra ~50 pp de coerência por 0,4 pp de intenção.

## A "física" medida

`CONSTANTES.md` — MFU, custo por run, leis de receita (batch crítico, LR por largura,
épocas úteis de dado premium: aprendizado puro ≤ ~2,7, memorização a partir de ~5, valor
zera em ~40), e quanto confiar em cada instrumento (juiz cego tem ruído de ±9 pp).

## Estrutura

```
bases/        receita + resultados de cada base (JSON)
receitas/     jobs de treino e avaliação como rodaram (Vast.ai, H100/4090)
treino/       model.py, laços de treino, tokenização, export GGUF
avaliacao/    CALAME-PT e LAMBADA-PT (log-verossimilhança e geração)
medicoes/     tabela de cobertura e análise dos adapters
docs/         história, corpus, licenças, receitas, vereditos (QAT, quantização...)
CONSTANTES.md constantes medidas do projeto
```

Os scripts referem-se a caminhos da infra original (R2 privado, `/work` nas instâncias);
publicam a **receita**, não um pipeline turn-key. Dados premium (pares gerados/filtrados
por professores) e o juiz calibrado não fazem parte desta publicação.

## Como carregar

Arquitetura GPT (decoder-only) definida em `treino/model.py`; checkpoints no HF em
safetensors (bf16) + `tokenizer.json` (BPE próprio, treinado no corpus pt-BR). Export
para GGUF: `treino/exporta_gguf.py` (roda em llama.cpp; Q4_K_M testado em Android).

## Citação

Se este material for útil, cite o repositório. Artigo em preparação (versão atualizada
com as 84 medições e a curva de dose).
