# A tese do produto validada com número — 01/08/2026

**A tese:** um modelo de 269M não deve SABER, deve PROCESSAR o texto que recebe. O
conhecimento vem do prompt, não dos pesos (que comportam ~67 MB, por 2 bits/parâmetro).

**A tese se sustenta, e o dado é forte:**

| Modelo | SQuAD 2.0 (F1) |
|---|---|
| **RoBERTa 124M ajustado** | **82,43** |
| Llama-3.1-70B-Instruct | 73,68 |
| GPT-4 Turbo | 59,56 |
| Llama-3.2-1B **zero-shot** | **28,28** |

**565× menos parâmetros INVERTE o resultado — quando treinado para a tarefa.** Sem
treinar, um modelo abaixo de 1B é um leitor ruim. A especialização é tudo.

**E existe prova no nosso tamanho exato:** o OCC-RAG-0.6B (arXiv:2606.00683) faz
mid-training de um Qwen3-0.6B com 3,25 M exemplos sintéticos e vai de **6,3 → 86,9 em
acurácia de recusa** e de **59,7 → 79,9 em fidelidade** — **superando o Qwen3-4B nas
duas**. É exatamente a receita que temos condição de executar.

## A receita

**1. O formato de saída decide tudo.** Emitir `STATUS: RESPONDIDA|AUSENTE`, depois a
**citação literal** do trecho, depois a resposta. O RAFT ensina o modelo a copiar verbatim
com marcadores e ganha 73,3 contra 71,6 (PubMed) e **35,3 contra 4,4** (HotpotQA) sobre
SFT+RAG comum. **Cópia literal é a operação mais barata que um 269M sabe fazer** — usar
isso como âncora anti-alucinação.

**2. Mistura de SFT:** 55% QA extrativo com contexto único · 25% com **distratores**
(1 documento correto + 3-4 errados) · **15-20% negativos** (resposta ausente) · 5-10%
tarefas irmãs (resumo, extração de campo, reescrita), no mesmo formato.

Gerar o negativo **apagando o trecho da evidência** (truque do ChatQA) é o mais barato e
o mais fiel. Começar em 20% e fazer ablação em 10/20/33%.

**3. Anti-recusador-compulsivo — métrica obrigatória.** No SQuAD 2.0, **"sempre abster"
já dá 48,9 F1**. Um número agregado NÃO detecta o modelo degenerado. Reportar sempre o
**par**: acurácia nos respondíveis e acurácia nos ausentes, mais taxa de falsa abstenção.

**4. Contexto: estender para 2048 vale.** O termo quadrático sai de ~8% para ~25% das
FLOPs de camada; custo total **+23%**, cache de 50 para 200 MB. Método: RoPE com YaRN/NTK
mais PoSE. **Mas o nosso usa posição APRENDIDA** — exige trocar por RoPE e re-treino curto.
**Não** usar StreamingLLM: por admissão dos autores, não estende janela.

Alerta: em "Lost in the Middle" só modelos de 13B/70B mostram a curva em U; o Llama-2-7B
era **puramente enviesado para o fim**. Esperar que o nosso 269M **ignore o começo do
contexto** — validar com QA real, nunca com agulha-no-palheiro.

**5. Medir alucinação:** MiniCheck-FT5 (770M) dá **74,7% de acurácia balanceada contra
75,3% do GPT-4, a US$ 0,24 contra US$ 107**. Em português não há benchmark de fidelidade —
usar mDeBERTa-v3-base-xnli como juiz e calibrar com 100-150 exemplos anotados à mão.

**6. RAG local no celular, medido:** MobileRAG num Galaxy S24 — embedder GTE-Small de 33M
mais Qwen2.5-0.5B dá **56,5% no SQuAD**, com tempo até o primeiro token 10-26% menor e
energia 24-40% menor.

## CONTRADIÇÃO entre dois levantamentos nossos — registrar

Este relatório recomenda **FaQuAD-NLI como benchmark primário**. O outro levantamento do
mesmo dia mediu, puxando os resultados brutos do leaderboard, que o **FaQuAD-NLI é
DEGENERADO**: os quatro Tucanos (160m, 630m, 1b1, 2b4) marcam **43,97 idêntico**, que é
exatamente o macro-F1 da classe majoritária (78,46% do split).

**Vale a medição, não a recomendação.** FaQuAD-NLI não serve como métrica primária. Para
abstenção, usar **Pirá 2.0**, que é o único dataset em português com perguntas sem
resposta e tarefa de disparo de resposta.

## Ordem de execução

1. Construir o eval pt-BR com split respondível/ausente — **antes** de treinar.
2. SFT com citação literal, distratores e 20% de negativos.
3. Só então estender para 2048 e re-medir.
