# Custo real da fase de conversa — pesquisa de preços (30/07)

O plano original estimava ~US$25 total. Duas pesquisas de preço corrigiram e depois
resolveram a conta.

## A conta que estourou

200k diálogos (professor gpt-oss-120b: ~240M tokens in / 80M out) + 200k julgamentos
(juiz): com Qwen3-32B de juiz, total real **US$36-45** — o juiz custava quase igual
ao professor. Groq batch (-50%) não compete (base cara: 0,15/0,60); prompt caching
não é a alavanca (few-shot de 600 tokens × 200k ≈ só US$4,4 a preço cheio).

## A solução

**Professor**: DeepInfra tier Flex (0,8×: US$0,030/0,136) ≈ **US$18**.
https://deepinfra.com/openai/gpt-oss-120b (OpenRouter floor 0,03/0,17 é equivalente)

**Juiz** (troca do Qwen3-32B por candidato barato de família ≠ OpenAI):

| candidato | custo 200k | nota |
|---|---|---|
| Gemini 2.5 Flash-Lite (Batch -50%) | **~US$9** | free tier (~1.500 req/dia) cobre a calibração de graça; pt-BR forte |
| DeepSeek V4-Flash | **~US$11,4** | cache de prefixo a US$0,0028/M — rubrica fixa quase grátis; família bem distinta |
| Qwen3-14B | ~US$16 | único com kappa publicado como juiz (SLMJury κ=0,519) |
| Gemma 3 27B | ~US$12,8 | reserva |

Literatura: juiz ~8B sofre "kappa deflation" (consistente porém enviesado —
arXiv 2606.19544); ponto doce barato é 14B-27B (SLMJury, arXiv 2606.07810). Nossa
rubrica simples (0-2, JSON) reduz a exigência.

**Decisão empírica**: rodar os 3 candidatos nos 80 rótulos humanos do dono
(<US$0,05 total) e escolher pelo kappa; empate → o mais barato. O humano é o
árbitro, não o preço nem o benchmark.

**Refinamentos do agy (acatados)**:
1. **Few-shot no prompt do juiz**: 2-3 diálogos já julgados dentro do prompt — é a
   melhor alavanca de calibração pra juiz pequeno e sai quase grátis (cache do
   DeepSeek / preço do Flash-Lite).
2. **Kappa sozinho não decide entre candidatos próximos** (80 itens = ruído de ±
   alguns pontos): olhar TAMBÉM o formato do erro — falso positivo em
   alucinação/informação errada pesa mais que confusão entre "bom" e "ótimo".
3. **Riscos por candidato**: Flash-Lite pode ter "kappa deflation" de modelo lite
   (consistente porém raso); DeepSeek pode julgar mecânico em gíria/nuance BR;
   Qwen3-14B tem κ publicado mas em benchmark inglês ("imposto idiomático").
   Nenhum é eliminatório a priori — é exatamente o que o teste nos 80 mede.
4. Tolerância a prompt: se um candidato falhar no formato JSON, uma iteração de
   prompt por modelo é permitida antes de descartar (arquiteturas diferentes
   reagem diferente ao mesmo prompt).
5. **Viés de verbosidade**: professor de 120B tende a prolixo, e juiz pequeno tem
   viés documentado de premiar resposta longa — a rubrica já pune comprimento
   (juiz.md), manter isso explícito no prompt de todos os candidatos.
6. Ordem de teste do agy: Flash-Lite → **Gemma 3 27B** (promovida: tamanho no ponto
   doce da literatura + pt-BR do Google, ~US$12,8) → DeepSeek → Qwen3-14B por último.

**Total revisado da fase: ~US$27-29** (professor ~18 + juiz ~9-11), de volta ao
planejado. SFT continua grátis (Kaggle).

Relacionado: docs/conversa/LEIA.md, [[pocket-lm-estado-29-07]].


## DECISÃO (30/07, calibração executada)

Os 3 candidatos foram testados contra o gabarito humano do dono (92 itens: 80
aceites reais + 12 sabotados que ele rejeitou 12/12, com 8/8 de consistência nas
duplicatas):

| candidato | kappa | sabotados que passaram | bons rejeitados |
|---|---|---|---|
| **DeepSeek V4-Flash** | **0,954** | 0 | 1 |
| Qwen3-14B | 0,904 | 1 (pt-PT) | 1 |
| Gemini Flash-Lite | 0,832 | 0 | 4 |

**Juiz escolhido: DeepSeek V4-Flash** (kappa quase perfeito, zero erro crítico,
~US$11 pros 200k com cache de prefixo). Gemini Flash-Lite fica como segunda opinião
gratuita (free tier) pra casos borderline. Qwen descartado pelo falso aceite de
pt-PT — o formato de erro que mais pesa (critério do agy). Script:
calibra_juiz.py; resultados em conversa_out/calibra_*.json.
