# Pocket-LM 269M

Modelos de linguagem pequenos em **português do Brasil**, treinados **do zero**, para rodar
**offline no celular** (llama.cpp / GGUF, CPU). Este repositório é a publicação aberta da
família **269M**: bases de pré-treino, receitas de treino, adapters de produto, medições e
lições. Licença **Apache 2.0** — pesos e código.

- Pesos no Hugging Face: coleção **Pocket-LM 269M** (usuário [`Magurofg`](https://huggingface.co/Magurofg))
- Corpus: a **fase final** de pré-treino da `generica-v1` consumiu **3,08 bi de fichas únicas**
  amostradas de um corpus de 12,31 bi montado só com fontes de licença declarada pelos
  distribuidores (finepdfs e FineWeb2-HQ: ODC-By 1.0, que cobre a base de dados e não
  automaticamente cada documento; HPLT 2.0: CC0). Ela continuou pesos de fases anteriores
  (13,1 bi de fichas) cuja fonte principal é o **ClassiCC-PT, que não declara licença**.
  Total: 16,2 bi de fichas únicas. `conversa-v1` e `executor-v1` foram pré-treinadas do
  zero sem essa fase.
- **Correção de 21/08/2026:** no treino em 4 GPUs, os 4 processos liam o mesmo micro-lote.
  As contagens anteriores mediam fichas processadas, não fichas distintas. **Limitação
  conhecida:** o `treino/train_gpu.py` publicado aqui ainda monta a mesma permutação em
  todos os processos. Para usar em várias GPUs, fatie a ordem da época por processo
  (`ordem[RANK::MUNDO]`) antes de treinar.
- **Modelo recomendado hoje:** `pocket-assistente-v2-q4_k_m.gguf` (28/08/2026) em
  [Magurofg/pocket-lm-269m](https://huggingface.co/Magurofg/pocket-lm-269m). As seções
  abaixo citam o v8, que era o oficial até essa data.
- **Placar vivo** (toda versão × toda métrica, regenerado a cada rodada):
  https://pocket.omnigen.ia.br/placar/ — versão em markdown em `docs/PLACAR.md`
- Aplicações comerciais construídas sobre a família: https://pitaia.ia.br

## As três bases (pré-treino do zero, 269M/286M parâmetros)

| base | ideia | CALAME-PT | LAMBADA-PT | intenção (ações) | coerência (conversa) |
|---|---|---|---|---|---|
| `generica-v1` | fase final com fontes de licença declarada (fases anteriores com ClassiCC-PT), nenhum dado de tarefa | **53,13** | **36,85** | 0,0 | 11,5 |
| `conversa-v1` | fase de decay dominada por diálogo | 48,55 | 32,23 | 0,0 | 69,2* |
| `executor-v1` | fase de decay dominada por ações de celular | 49,86 | 35,01 | **86,4 sem SFT** | 34,6 |

\* n=52, juiz binário. Receita e resultados completos de cada base em `bases/*/RECEITA.json`
e `bases/*/RESULTADOS.json`.

CALAME-PT: uma varredura por 13-gramas no material de treino do projeto achou correspondência
para 155 das 2.076 frases (08/09/2026). O número nas 1.921 sem correspondência detectada
ainda não foi calculado.

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
- **McNemar pareado contra o adapter oficial de produto (v8, n=1035):** intenção 87,92
  (adapter-app-v1) × 87,73 (v8) — cerca de 2 itens, sem significância (o que não
  demonstra equivalência); F1 de campos 74,84 × 77,96 — 3,1 pp a favor do v8. O v8
  permaneceu oficial até 28/08, quando foi substituído pelo `pocket-assistente-v2` (média
  de pesos, sem retreino). Detalhe em `docs/PLACAR.md`.

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

Nota de reprodução — **fusão de LoRA para comparação**: para comparar um adapter LoRA
com um SFT completo, funda o adapter na base antes de medir (mesma matemática do export
GGUF: `W += B@A · α/r`, mapeando as chaves da convenção gguf para o `state_dict` —
ver `receitas/job_mcnemar_v8.sh`) e valide com um **portão de sanidade**: o modelo
fundido precisa de ≥60 de intenção em 50 itens antes de qualquer medição valer. Fusão
silenciosamente incompleta produz números falsos — foi assim duas vezes aqui antes de a
regra existir.

## Citação

Se este material for útil, cite o repositório. A tabela de cobertura tem 96 medições
(12 células × 8 métricas). Um artigo está em rascunho, ainda não submetido.
