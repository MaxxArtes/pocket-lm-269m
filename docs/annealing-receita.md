# Fase de decaimento (annealing): o que está medido — 01/08/2026

É a alavanca mais barata que temos: acontece dentro de um treino que já vamos pagar, e os
ganhos publicados são grandes.

## Ganhos medidos nas fases finais

| Modelo | Duração | Composição do lote | Ganho MEDIDO |
|---|---|---|---|
| **OLMo 2 7B** | 3×50 bi, LR→0, **média dos 3 pesos** | ~47% web filtrada + 53% premium | GSM8K 24,1→**67,5**; MMLU 59,8→63,7; média OLMES 50,6→**61,2** |
| OLMo 2 13B | 3×100 bi + 1×300 bi | ~50-52% web | GSM8K 37,3→75,1; média 56,5→66,8 |
| **Llama 3 8B** | últimos ~40 bi | "upsample de fontes de altíssima qualidade" (proporção não publicada) | GSM8k **+24%**, MATH +6,4%. **No 405B o ganho é ~zero** |
| **MiniCPM** | decay do WSD | pré-treino + SFT (UltraChat, SlimOrca, EvolInstruct, didático K-12) | C-Eval 40,0→**52,6**; MMLU 44,6→50,9 |
| **SmolLM2** | 1 tri (10% do total) | web 58% · Stack-Edu 24% · **matemática 14%** · Cosmopedia 4% | matemática 7,27→**22,07**; código 16,75→23,21 |
| Gemma, Qwen | — | **não publicam** | — |

**A metade premium do OLMo 2, decomposta:** ≈64% artigo científico, 19% instrução,
12% matemática, 4% enciclopédia, 1% QA técnico.

## O padrão que importa: annealing move FORMATO DE TAREFA, não conhecimento difuso

Não há ablação publicada comparando tipos de dado lado a lado — o agente procurou e não
achou. Mas o padrão dos quatro papers é consistente: no OLMo 2 o GSM8K sobe **43 pontos**
enquanto o MMLU sobe 3,9. No SmolLM2 os ganhos grandes estão exatamente nos domínios que
foram aumentados. **O que se coloca na fase final é o que melhora.**

Ordem inferida (não medida): matemática/raciocínio em prosa > instrução/QA >
científico-didático > enciclopédia > web filtrada > livro literário.

## Os 50% do OLMo 2 são restrição de oferta, não ótimo

Insight do agente: o pool Dolmino tem 91 bi de material não-web; três rodadas de 50 bi
exigem ~26 bi premium cada — passar de 53% forçaria repetir demais. **O 0,5 que copiamos é
racionalização de escassez.** Blakeney mediu a *duração* ótima (10-20%), mas ninguém
varreu a *fração do lote*. É ablação barata e original disponível para nós.

## Repetição: até 4 épocas é seguro, mas há coisa melhor

Muennighoff mediu perda desprezível até **4 épocas** de dado repetido. O OLMo 2 evita o
problema por outra via: roda o decaimento **3-4 vezes com ordens diferentes e faz a média
dos pesos** (souping). Recomendação: **≤2 épocas + souping**.

## Sintético na fase final é o item dominante

Toda a matemática do Dolmino é sintética; o decay do MiniCPM é SFT sintético — e é a
**medição mais limpa que existe** do valor do sintético na fase final, porque é a única
variável que muda entre C-Eval 40,0 e 52,6. Nemotron-CC ganhou +5,6 de MMLU com reescrita
sintética.

## A OMISSÃO que o próprio agente admitiu, e que conflita com uma decisão nossa

A mistura proposta tem **~0% de código e ~0% de matemática** — enquanto o SmolLM2 põe
24% + 14% e **é exatamente aí que estão os maiores ganhos medidos**.

**Isso tensiona uma decisão que tomamos:** no desenho do ext3, o agy vetou matemática e
jurídico ("269M não absorve; o produto é conversa") e nós acatamos. A evidência de
annealing diz o contrário — matemática na fase final é o que mais move benchmark.

**A reconciliação honesta:** as duas coisas podem ser verdade. Matemática na fase ESTÁVEL
de um 269M provavelmente é desperdício; matemática na fase de DECAIMENTO é onde o ganho
aparece. São fases diferentes com funções diferentes. **Decidir com ablação, não com
argumento** — e a nossa métrica alvo (CALAME, que é cloze) pode não capturar isso, o que
torna a escolha dependente do que queremos: pontuar no CALAME ou ter capacidade.

## Mistura proposta, com a origem de cada número declarada

| Bloco | Fração | Origem |
|---|---|---|
| web filtrada (FineWeb2-HQ + GigaVerbo) | 40% | **ancorado** nos 47-52% do OLMo 2; puxado para baixo por julgamento |
| livro/artigo (FinePDFs, SciELO, Gutenberg) | 32% | **ancorado**: artigos são 64% da metade premium do OLMo 2 |
| Wikipédia | 15% | **julgamento** — OLMo usa 4%; subiu porque pt não tem equivalente de FLAN |
| LegalPT | 10% | **julgamento puro** — nenhum modelo aberto usa jurídico no annealing |
| provas/instrução (ENEM, BLUEX, Essay-BR), repetido 3-5× | 3% | **ancorado** no FLAN (19% do premium OLMo); fração baixa é julgamento |

## `gigaverbo-v2-synth` herda o problema de licença? — resposta

**O gerador está limpo.** O card declara Apache 2.0, e os modelos que geraram
(Qwen2.5-7B/14B/32B-Instruct) são **Apache 2.0, que não tem cláusula sobre as saídas do
modelo** — ao contrário da licença do Llama 2, que restringe explicitamente usar saída para
treinar outro modelo. **Não há contaminação vinda do gerador.**

**O risco residual é a semente.** São 14 datasets-semente (Cosmopedia-v2, FineMath,
Wikipedia, LegalPT, CodeParrot, StarCoder, teses CAPES/BDTD, resumos SciELO, blogs). O card
**não lista a licença de cada uma** nem faz declaração de conformidade. As sementes nomeadas
não incluem os componentes NC do GigaVerbo original.

**Recomendação:** risco materialmente menor que no GigaVerbo original, **não zero**. Para
**treino de pesquisa e publicação aberta, usar**. Para **uso comercial, pedir ao publicador
o manifesto de licenças das 14 sementes** antes de tratar como Apache 2.0.
