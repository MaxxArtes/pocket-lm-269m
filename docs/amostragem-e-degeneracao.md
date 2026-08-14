# Amostragem e portão de degeneração (31/07)

## Defaults recomendados

```
temperature 0.3   top_p 0.9   top_k 40   min_p 0.05
penalty_repeat 1.08   penalty_last_n 128   max_tokens 384
```

**Temperatura 0.3 é o parâmetro que mais protege contra loop**, e 0.3 fica dentro da faixa
que modelos pequenos reais usam — não é chute nosso. Lido dos arquivos publicados:

| Modelo | temp | top_p | top_k | rep_pen |
|---|---|---|---|---|
| SmolLM2-135M/360M-Instruct | 0.2 | 0.9 | — | — |
| Qwen2.5-0.5B-Instruct | 0.7 | 0.8 | 20 | 1.1 |
| **Tucano-1b1-Instruct (pt-BR)** | 0.1 | 1.0 | 50 | 1.2 |
| llama.cpp (default do binário) | 0.8 | 0.95 | 40 | 1.0 (desligado) |

O padrão: **quanto menor o modelo, menor a temperatura que os próprios autores enviam.**
Nenhum modelo pequeno de verdade usa os 0.8 do default do llama.cpp.

Penalidade em **1.08, não 1.2**: `repetition_penalty` pune o token pela identidade, sem
olhar contexto — em português isso castiga "de/que/a/não" e **quebra citação literal de
texto colado**, que é exatamente o que o bloco de compreensão do nosso dataset treina.
1.08 quebra loop sem esse dano. `no-repeat-ngram` está descartado: torna citação verbatim
impossível. DRY seria melhor em princípio (só pune o que estende repetição existente) mas
**o wllama não expõe `dry_*`** — verificado em `src/types/types.ts`.

## min-p: a evidência foi refutada, não usar como estratégia

O paper do min-p (arXiv:2407.01082, ICLR 2025 Oral) alega superar top-p em temperatura
alta. **Existe refutação formal**: arXiv:2506.13681 ("Min-p, Max Exaggeration") mostra que
a avaliação humana omitiu dados e usou teste estatístico errado; controlando o número de
hiperparâmetros na varredura, o ganho desaparece; as alegações de adoção em massa foram
retiradas. Os próprios autores originais refizeram a avaliação humana e também não
acharam ganho. Além disso **nada abaixo de 1B foi medido** — para o nosso porte qualquer
afirmação é extrapolação. Mantemos min_p 0.05 porque é o default barato do llama.cpp e
serve de piso, **não** porque resolve algo. Em especial: não subir a temperatura para 0.8+
"porque o min-p protege".

## Portão automatizado — `portao_degeneracao.py`

Suíte fixa de 50 prompts pt-BR, seed fixa, endpoint compatível com OpenAI. Reprova se:

- **mediana de seq-rep-4 > 0,05** ou **alguma resposta > 0,20**
- **menos de 95% das respostas param sozinhas** (bater no teto de tokens conta como loop)
- qualquer janela de 20 tokens repetida 3× ou mais

Âncoras medidas na literatura para seq-rep-4 (Welleck, arXiv:1908.04319): texto **humano
0,005-0,006**; GPT-2 no greedy **0,506**; top-k 3 **0,099**. O corte de 0,05 fica uma ordem
acima do humano e uma ordem abaixo do greedy. Self-BLEU foi descartado de propósito: mede
diversidade entre amostras, custa caro e não responde "está em loop".

## Linha de base medida (31/07, modelo publicado, ANTES do SFT)

`PASSA`: seq-rep-4 mediana **0,0**, pior caso 0,023, distinct-2 0,995, **100% pararam
sozinhas**, zero loops duros. 20 prompts, temp 0.3, teto de 200 tokens.

**Ressalva que impede comemorar:** a média foi de **30,5 tokens por resposta**. Degeneração
não é o nosso problema hoje — o problema é conteúdo. As respostas saem no formato e no tom
certos de assistente, mas erram o assunto: "explica inflação" veio com descrição de API,
"mensagem de aniversário" devolveu a própria instrução parafraseada, "arroz soltinho" veio
com instrução de descongelar em saco plástico. Ou seja: o portão está aferido e serve de
guarda de regressão, mas **passar nele não é sinal de qualidade** — respostas curtas passam
fácil. Ele só ganha valor depois do SFT, quando as respostas alongarem e o risco de loop
aparecer de verdade. Julgamento de conteúdo continua sendo trabalho do juiz calibrado.

## Armadilha de entrega no GGUF (verificada no código)

As chaves `general.sampling.*` existem no GGUF e o llama.cpp as lê
(`common_init_sampler_from_model`), mas **só valem para o que o usuário não setou** — e,
pior, **`convert_hf_to_gguf.py` não lê `generation_config.json`**. Se não gravarmos as
chaves na mão, elas simplesmente não existem no arquivo. O `tokenizer.chat_template` viaja
e o wllama lê (`getChatTemplate()`), mas o llama.cpp dele é submódulo pinado.
**Conclusão operacional: mandar todos os parâmetros explícitos no JS do demo**, não confiar
que o default viaja.

## O que não fazer

Subir temperatura para 0.8+ apostando no min-p; usar `no-repeat-ngram`; empilhar
`rep_pen ≥ 1.2` num demo que aceita texto colado; contar com defaults viajando no GGUF;
**usar amostragem para esconder problema de SFT** — se o portão só passa com temp 0.1, o
problema está nos pesos, não na amostragem.

Fontes: arxiv.org/abs/2407.01082 · arxiv.org/abs/2506.13681 (refutação) ·
arxiv.org/abs/1908.04319 (seq-rep-n) · arxiv.org/abs/1904.09751 (degeneração) ·
generation_config.json de Qwen2.5-0.5B-Instruct e Tucano-1b1-Instruct ·
github.com/ggml-org/llama.cpp (common/common.h) · github.com/ngxson/wllama (types.ts)
