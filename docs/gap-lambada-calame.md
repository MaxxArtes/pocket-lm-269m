# Por que o LAMBADA não se moveu enquanto o CALAME saltou 9pp

Pesquisa de 30/07, motivada pelo veredito do 269M: CALAME-pt 36,8%→45,9% (ganho real,
p<0,0001) mas LAMBADA-pt 24,8%→25,1% (indistinguível de ruído). O que a literatura diz:

## Confirmado

- **Tarefas escalam de forma heterogênea** com mudanças de receita/arquitetura/dado
  além de parâmetros e tokens brutos — LAMBADA especificamente aparece como uma das
  tarefas MAIS sensíveis a esse tipo de variável (28% de redução relativa de erro ao
  incluir features de arquitetura/dado no modelo preditivo, a maior do conjunto
  testado). Ou seja: LAMBADA não é "resistente por natureza" a mudanças de receita —
  é sensível a certas mudanças, aparentemente não às que fizemos nesta rodada (mais
  dado do mesmo tipo + WSD + fase premium).
  https://arxiv.org/abs/2503.03862
- **LAMBADA escala suave/log-linear com tamanho de modelo**, não em degrau — ficar
  estagnado numa faixa pequena de parâmetros (98M→269M) é esperado, não sinal de
  problema na receita.
  https://arxiv.org/pdf/2206.07682 · https://arxiv.org/html/2503.05788v1
- **Confound conhecido desde o GPT-2**: modelos tendem a gerar continuação plausível
  de frase em vez de parar exatamente na palavra certa — um filtro de stop-word já
  corrige ~4 pontos. Parte da "estagnação" é ruído de medição do próprio benchmark,
  não do modelo.
  https://shreyansh26.github.io/post/2021-05-23_language_models_unsupervised_multitask_learners_gpt2/

## Não confirmado (hipóteses descartadas por falta de fonte)

- Contexto de treino curto (block_size=512) como causa direta: LAMBADA só exige em
  média 4,6 frases / mínimo 50 tokens de contexto — cabe folgado em 512. Não achamos
  paper que isole "janela de treino curta" como mecanismo de estagnação aqui.
- "Dado premium favorece previsão local sobre correferência longa": hipótese
  plausível, mas sem fonte direta — não usar como justificativa em nenhuma decisão.

## Recomendação

**Não vale investir em "consertar" o LAMBADA agora.** A alavanca que move essa
métrica (currículo de contexto/documento longo) é diferente da que movemos nesta
rodada. Manter o foco no CALAME e métricas de fluência de curto alcance, que é onde
o ganho real está e onde a receita atual comprovadamente funciona. Revisitar LAMBADA
só numa fase com modelo maior e, se fizer sentido, dado explícito de documento longo.

Relacionado: [[pocket-lm-estado-29-07]], [[pocket-lm-pesquisa-treino]].
