# Varredura de técnicas recentes de treino pequeno (30/07/2026, janela jun-jul/2026)

## Achados aplicáveis

1. **SCALE (memory-efficient optimizer)** — arXiv 2506.16659, aceito ICML 2026.
   Normalização column-wise + momentum só na última camada; iguala/supera Adam em
   60M-1B params usando 35-45% da memória, supera GaLore/Fira/APOLLO. **Alta
   prioridade**: nossa faixa de parâmetros exata; memória menor = GPU mais barata
   pro mesmo throughput. Vale um teste curto comparando contra Muon.
2. **Data-Constrained Pretraining / MIR** — arXiv 2606.06888. Masked-input
   regularization + weight decay forte equivale a ~1,3x mais dado único ao reciclar
   corpus em múltiplas épocas; lei de escala "SoftQ" (melhor que Chinchilla aditivo
   pra regime de repetição). Testar MIR é barato (CPU/preprocessamento) antes de
   gastar orçamento de GPU em dado novo.
3. **L20-Edu-135M** — arXiv 2606.22189. Metodologia auditável em GPU única,
   orçamento parecido com o nosso (134.5M params, 13B tokens). Achado negativo
   importante: RLVR direto em GSM8K PIOROU o modelo — não vale tentar isso na nossa
   escala sem mistura curada antes.
4. **OpenLanguageModel** — arXiv 2607.16669. Framework de engenharia, não técnica
   nova — prioridade baixa/média, só reduz atrito se formos reescrever a esteira.

## Fora de escopo (não aplicável)

- **SOAP/Muon/beyond em escala grande** (arXiv 2607.20548) — multi-bilhões de
  parâmetros, batch de 100M tokens. Não bate com nosso porte.

## Alerta de concorrência

**Tucano 2 Cool** (arXiv 2603.03543, mar/2026) — suíte 0.5-3.7B com novo corpus
GigaVerbo-v2 + dado sintético. Fora da janela de 4-8 semanas mas relevante: a régua
de comparação (nossa referência acadêmica mais próxima) está subindo. Reavaliar o
gap antes de anunciar.

## Sem sinal

Nenhum modelo pt-BR/multilíngue novo na faixa 100M-1B com receita documentada nas
últimas 4-8 semanas, além do Tucano 2. Nenhuma técnica nova de overtraining
além-Chinchilla específica pra escala pequena nesse período.

Relacionado: [[pocket-lm-estado-29-07]], [[pocket-lm-pesquisa-treino]],
[[pesquisa-sempre-ativa]].
