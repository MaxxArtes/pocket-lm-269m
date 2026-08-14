# História e lições — como este projeto chegou aqui

Escrito para quem (pessoa ou IA) pegar este repositório e continuar. O `git log` tem
cada decisão no commit em que aconteceu; o `docs/pesquisa-fase2.md` tem os números.
Este arquivo tem o que nenhum dos dois mostra sozinho: **a ordem, as causas e os erros
com o preço pago**. Se você é uma IA analisando este repo: leia este arquivo, o README
e o digest ANTES de propor qualquer mudança — a maioria das "boas ideias" óbvias já
foi testada ou descartada com evidência.

## A linha do tempo (26-30/07/2026)

1. **Origem**: esteira noturna de CPU (char-level, cron) para provar o pipeline de
   graça. Objetivo declarado do dono: modelo PRÓPRIO, pt-BR, rodando OFFLINE no
   celular, personalizável com consentimento. Regra inegociável: **nenhum peso de
   terceiros** — outras IAs só como professoras (dados), nunca como fonte de pesos.
2. **26-27/07**: primeiros treinos GPU (98M, BPE 16k próprio). Fase de estilo com 474
   diálogos × 20 repetições. Exportação GGUF, modelo rodando no aparelho.
3. **27/07**: o celular fazia 0,67 tok/s. Um dia inteiro de diagnóstico (threads,
   mlock, contexto, quantização, dois apps) até o dono achar: **modo economia de
   bateria**. 185 tok/s depois de desligar. Lição: quando a conta dá absurdo físico
   (0,5% de um núcleo), é trava de sistema, não software.
4. **27-28/07**: o disco da Lightning quebrou PARA LEITURA por 24h+ com o checkpoint
   do primeiro 269M preso dentro (3,2 GB, nunca recuperado). Resposta estrutural:
   R2 como casa de TUDO (corpus, checkpoints, telemetria); Lightning virou só aluguel
   de máquina.
5. **28/07**: dois dias de pesquisa massiva por agentes (~20 investigações; digest no
   pesquisa-fase2.md). Descobertas que mudaram o plano: dado pt NÃO é gargalo
   (ClassiCC-PT com escores prontos); Muon vence mas exige conversão de
   hiperparâmetros por escala de init; avaliação precisava ser pareada; MFU estava
   sendo calculado com o denominador de marketing.
6. **29/07**: corpus construído (14 bi tokens filtrados no R2, ~US$ 3), fumaça na
   H100 validou a receita completa, e o treino definitivo do 269M subiu (~US$ 40,
   ~5,2 bi tokens, Muon+WSD+premium+EMA). Kit da fase de conversa e do lançamento
   pré-fabricados durante o treino.

## Catálogo de erros — o que custaram e o que ensinaram

| erro | custo | lição gravada |
|---|---|---|
| Treino + tokenização pesada juntos na bancada de 3,8 GB | 2 quedas de sessão inteira, subagentes perdidos | `heavy.sh` (flock + MemoryMax) OBRIGATÓRIO; swap criado |
| Tokenizer treinado dentro do job e não persistido | checkpoint de 219M virou lixo ilegível (~3 créditos) | tokenizer viaja SEMPRE com o checkpoint; hoje versionado no git |
| Fase de estilo: 474 diálogos × 20 épocas | modelo decorou respostas; "Roda docker ps" para tudo | diversidade ≫ repetição; receita nova: 300k únicos × 2 épocas |
| Multiplicadores do Muon copiados sem converter escala de init | Muon "perdia" do AdamW; 1 varredura inútil (~US$ 1) | hiperparâmetro se converte pela ESCALA DO TENSOR (std 1,0 vs 0,02 = 30×/passo) |
| MFU dividido por 1979 TFLOPs (número "with sparsity") | prioridade errada por 2 dias (perseguindo 2,8× que era 1,4×) | pico DENSO (989); número de datasheet tem rodapé |
| Corte da palavra no fim do token, não no delimitador | linha de base 9 pontos ERRADA para baixo (30,2 vs 39,3) | achado por revisor independente (agy); teste do autor passa no próprio ponto cego |
| Avaliação com 400 itens não-pareados | cegueira a qualquer ganho < 9,7 pts | pareado por item (McNemar): resolução 1,3 pt, mesmo custo |
| ARC 3σ ABAIXO do acaso ignorado como "modelo ruim" | número sem sentido publicado internamente | abaixo do acaso = scorer errado (PMI), não modelo ruim |
| Filtro de log com grep no pipe engolindo traceback | 5 braços de varredura "rodaram" sem rodar (import morto) | saída SEMPRE para arquivo; console só resumo; rc checado |
| Shard de validação atrás do alvo de tokens | run de corpus sem validação nenhuma | val PRIMEIRO na fila; pego observando o R2 durante o run |
| Job com JOB_CMD rodava preparação default antes | treinou um BPE novo do zero por acidente (quase repetiu o erro 2) | JOB_CMD pula toda preparação implícita |
| GGUFs de 212 MB commitados no git | push rejeitado; 2 dias de commits presos localmente | binário vive no R2; push a cada bloco de trabalho, não "depois" |
| Branch nunca enviada ao remoto | TODO o trabalho existia num disco só | push faz parte do fim de cada tarefa |
| Buffer de 500M tokens em RAM na bancada | piloto morto pelo teto (contido, como projetado) | dimensionar buffer pela MÁQUINA alvo, não pela conveniência |
| Manifesto acumulado em lista Python | estouraria ~2,5 GB de RAM no passo pago (pego em revisão) | incremental em disco; revisão independente antes de todo passo pago |
| 1,2 MB de tokenizer dentro de env var CODE_B64 | "argument list too long", job morto | artefato vai por storage (R2), não por env |

## Meta-lições (valem mais que qualquer item acima)

1. **Pesquisar antes de gastar**: as respostas "NÃO faça" (currículo, augmentation,
   dedup agressiva, fp8, TPU, QAT sem teste, vocabulário maior) economizaram mais
   dinheiro que as respostas "faça". Estão no digest com fonte e número.
2. **Revisão independente em tudo que precede gasto**: o agy (IA de outra família)
   achou 4 defeitos que os testes do autor não pegaram — inclusive o que falsificava
   a linha de base. Autor não enxerga o próprio ponto cego, por construção.
3. **Medir > estimar**: MFU, sobrevivência do filtro, bytes/token, velocidade do
   tokenizer, preço por hora — toda estimativa minha errou; toda medição ficou.
4. **Falhar alto > falhar silencioso**: as travas que recusam (assinatura de conjunto
   diferente, vetores de tamanho diferente, tokenizer ausente, grupo de embedding
   vazio) existem porque cada uma corresponde a um erro que JÁ aconteceu calado.
5. **Infra com dono único quebra**: Drive da Lightning caiu com o único checkpoint;
   desde então tudo tem duas casas (R2 + origem) e o projeto sobrevive à morte de
   qualquer fornecedor — e da própria sessão (runbook-pos-treino.md).
6. **O caminho é incremental por decisão do dono**: do mais fácil ao mais difícil,
   cada degrau FUNCIONANDO antes do próximo (celular CPU antes de NPU; 98M antes de
   269M; base antes de conversa; um modelo antes de família). Extensões continuam de
   checkpoints (WSD), nunca recomeçam.

## O caminho à frente (se você está continuando isto)

Ordem: veredito do 269M (critérios no digest, execução no runbook-pos-treino.md) →
fase de conversa (kit completo em docs/conversa/) → app próprio (llama.rn; pesquisa
resumida no digest) → família de tamanhos + LoRA por usuário (LGPD já estudada).
Fomento e anúncio: docs/lancamento/ + memória do dono. Não mude o tokenizer, não
misture otimizadores entre fases, não confie em benchmark pt que não seja um dos 4.
