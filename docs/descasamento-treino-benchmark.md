# Onde o nosso dado de treino difere dos benchmarks reais

Medido em 02/08/2026, enquanto a avaliação externa rodava. Serve para dois fins: prever
onde o modelo vai falhar fora do nosso conjunto, e corrigir o gerador na próxima rodada.

## O que bate e o que não bate

| | nosso treino | FaQuAD (externo) | veredito |
|---|---|---|---|
| palavras na resposta, mediana | 4 | 4 | bate |
| palavras na resposta, **média** | **4,0** | **8,6** | **NÃO bate** |
| caracteres na passagem, mediana | 466 | 778 | aperta |
| caracteres na passagem, faixa | 350–900 | 226–1.964 | **NÃO bate nas duas pontas** |
| palavras na pergunta, mediana | 7 | 8 | bate |
| tipo mais comum | `qual` 31% | `o que` 23% | difere |

## O descasamento que mais custa: teto de resposta

O prompt do gerador diz literalmente *"A resposta deve ser CURTA: no máximo 8 palavras"*.
O FaQuAD tem **média de 8,6 palavras** — ou seja, mais da metade das respostas dele é mais
longa que o teto que impusemos ao treino inteiro.

Consequência: o modelo responde com ~4 palavras onde se espera ~9, e o F1 cai **por
construção**, sem relação com ele entender ou não a pergunta. É erro de projeto do dado,
não de capacidade do modelo.

**Correção para a próxima rodada:** subir o teto de 8 para ~20 palavras, e deixar a
distribuição sair do texto em vez de ser imposta. Custo: zero GPU, só regerar.

## A janela de passagem aperta dos dois lados

Todas as passagens de treino ficaram entre 350 e 900 caracteres, por causa do filtro do
`colhe_passagens.py`. Duas consequências medidas:

- **Abaixo de 350 o modelo recusa tudo.** Testado com o MESMO texto em duas versões: com
  700 caracteres responde certo, com 270 recusa as três perguntas, inclusive as
  respondíveis. É defeito de produto — quem cola um parágrafo curto conclui que o modelo
  não presta.
- **Acima de 900 ele nunca viu.** O FaQuAD tem mediana de 778 e chega a 1.964.

**Correção:** alargar para 150–1.500 caracteres.

## A hipótese que a medição DERRUBOU

Eu previ que o FaQuAD seria mais difícil por usar menos as palavras do texto na pergunta.
**É o contrário:**

| conjunto | n | média | mediana | acima de 0,60 | sobreposição zero |
|---|---|---|---|---|---|
| nosso treino: normal | 1.923 | 0,557 | 0,600 | 45,8% | 9,7% |
| nosso treino: adversarial | 380 | 0,140 | 0,000 | 1,8% | 57,9% |
| **FaQuAD** | 62 | **0,640** | **0,667** | **53,2%** | **0,0%** |

O FaQuAD reusa **mais** o vocabulário do texto (0,640 contra 0,557) e não tem um único item
com sobreposição zero. Anotador humano lendo um texto e escrevendo pergunta sobre ele pega
naturalmente o vocabulário que acabou de ler.

**Implicação:** o conjunto adversarial que geramos (0,140 de sobreposição) é mais difícil
que qualquer coisa no FaQuAD. Se o F1 externo vier baixo, **não é por paráfrase** — essa
hipótese está descartada por medição, e sobra o teto de resposta.

Vale como achado para o paper: gerar variante adversarial é barato e treina o modelo acima
da dificuldade que os benchmarks humanos apresentam.

## Resumo das correções, em ordem de impacto esperado

1. **Teto de resposta de 8 → 20 palavras.** Explica a maior parte da perda de F1 esperada.
2. **Faixa de passagem 350–900 → 150–1.500 caracteres.** Corrige recusa em texto curto e
   cobre o comprimento real dos benchmarks.
3. **Replay 5% → 25%.** Recupera os 12,3 pontos de CALAME perdidos — ver
   `docs/publicacao-fomento-02-08.md` e a medição em `avaliacao/calame.json`.
4. Equilibrar tipos de pergunta (hoje `qual` domina com 31%).

Nenhuma das quatro custa GPU extra além do retreino; todas são mudanças no gerador de dados.
