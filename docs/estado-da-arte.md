# Estado da arte — o que já existe (e onde o Pocket-LM se encaixa)

Levantamento de julho/2026. Objetivo: saber **o que já está publicado** antes de gastar
GPU e semanas de engenharia. Nada aqui invalida o projeto — mas muda o que faz sentido
chamar de "entrega" e o que faz sentido chamar de "aprendizado".

> Contexto do projeto: `model.py` é um GPT decoder-only estilo nanoGPT, alvo 50–90M de
> parâmetros, tokenizer BPE byte-level, treinado do zero em pt-BR, com destino final a NPU
> do MediaTek Dimensity 9500s (ver `docs/npu-deploy.md`).

---

## TL;DR

1. **Já existe LM pt-BR treinado do zero e aberto, na faixa exata que o Pocket-LM mira:**
   **Tucano-160m** (162M params, Apache 2.0, 200B tokens de GigaVerbo) e
   **TeenyTinyLlama-160m/460m** (Apache 2.0, 6,2B tokens). São o piso de comparação honesto.
   O próprio card do Tucano diz que os modelos **não são destinados a deployment**.
2. **A maioria do que o Brasil publicou não é "do zero"** — Sabiá, Bode, Cabrita e Gervásio
   são *continued pretraining* ou *fine-tune* de modelos em inglês (LLaMA, Llama 2/3, OpenLLaMA,
   Pythia). Os modelos comerciais bons (Sabiá 4 / Amazônia) são **fechados, só API**.
3. **Multilíngue pequeno já ganha de pt-BR pequeno em quase tudo**: Qwen3-0.6B, Gemma-3-270M,
   LFM2.5-350M, SmolLM2-360M, EuroLLM-1.7B — todos com português no treino e com artefatos
   quantizados prontos (GGUF, `.litertlm`, ExecuTorch).
4. **Já existe um `.litertlm` pré-compilado pra NPU MediaTek MT6993** (= Dimensity 9500):
   `Qwen3-0.6B.mediatek.mt6993.litertlm`. Ou seja: o alvo de hardware da fase 3 **já tem um
   modelo rodando nele**, publicado pela própria comunidade LiteRT. Ressalva importante na §3.
5. **Caminho mais curto pro celular esta semana:** instalar **Google AI Edge Gallery** ou
   **PocketPal AI** da loja e carregar um GGUF/`.litertlm` pronto. Uma tarde, zero código.
   Isso separa "provar que o aparelho roda LLM" de "provar que o meu modelo roda".
6. **Resposta franca (§4):** treinar do zero **não** produz um produto competitivo em chat
   pt-BR. Produz (a) aprendizado real de ponta a ponta, (b) controle de licença e dados,
   (c) possibilidade de vencer em **domínio estreito e tamanho ultra-pequeno** (<50M), que é
   onde ninguém publicou nada em português. Fora disso, perde para um fine-tune de
   Gemma-3-270M/Qwen3-0.6B por uma margem grande.

---

## 1. Modelos em português já publicados

### 1.1 Treinados **do zero** nativamente em português (BASE de verdade)

| Modelo | Params | Quem | Dados | Licença | Qualidade relatada |
|---|---|---|---|---|---|
| [TucanoBR/Tucano-160m](https://huggingface.co/TucanoBR/Tucano-160m) | 162.417.408 | Nicholas Kluge Corrêa et al. (Univ. Bonn) | [GigaVerbo](https://huggingface.co/datasets/TucanoBR/GigaVerbo), 200B tokens pt (780 GB) | Apache 2.0 | média 35,14 no harness pt; ctx 2048; 320k passos em 8×A100-80GB, ~44h |
| [TucanoBR/Tucano-630m](https://huggingface.co/TucanoBR/Tucano-630m) | ~630M | idem | idem | Apache 2.0 | melhor que 160m, ainda abaixo de multilíngues modernos de mesmo porte |
| [TucanoBR/Tucano-1b1](https://huggingface.co/TucanoBR/Tucano-1b1) | ~1,1B | idem | idem | Apache 2.0 | "igual ou superior a outros modelos pt e multilíngues de porte similar" ([paper](https://arxiv.org/abs/2411.07854)) |
| [TucanoBR/Tucano-2b4](https://huggingface.co/TucanoBR/Tucano-2b4) | ~2,4B | idem | idem | Apache 2.0 | topo da família; versões `-Instruct` disponíveis |
| [nicholasKluge/TeenyTinyLlama-160m](https://huggingface.co/nicholasKluge/TeenyTinyLlama-160m) | ~162M | Corrêa et al. (2024) | Pt-Corpus Instruct, 6,2B tokens pt-BR | Apache 2.0 | antecessor do Tucano; [paper](https://www.sciencedirect.com/science/article/pii/S2666827024000343) |
| [nicholasKluge/TeenyTinyLlama-460m](https://huggingface.co/nicholasKluge/TeenyTinyLlama-460m) | 468.239.360 | idem | 6,2B tokens; 1,2M passos; 1×A100-40GB, ~280h | Apache 2.0 | perplexidade final 11,77; tem variantes `-awq` e [GGUF de terceiros](https://huggingface.co/afrideva/TeenyTinyLlama-460m-GGUF) |
| [NOVA-vision-language/GlorIA-1.3B](https://huggingface.co/NOVA-vision-language/GlorIA-1.3B) | 1,3B | NOVA Lisboa | ~35B tokens **PT-PT** (ArquivoPT, Europarl, OpenSubtitles) | ver card | [paper PROPOR 2024](https://arxiv.org/abs/2402.12969): supera decoders PT abertos existentes. **PT-PT, não pt-BR** |

**Observações que importam:**

- O card do Tucano é explícito: *"Tucano models are **not intended for deployment**"* e
  *"for the Portuguese language only"*. São artefatos de pesquisa, não produtos.
- O [paper do Tucano](https://arxiv.org/abs/2411.07854) traz o achado mais desconfortável do
  campo: **o desempenho nos benchmarks pt usados pela comunidade tem pouca ou nenhuma
  correlação com a quantidade de tokens ingeridos no treino**. Traduzindo: os benchmarks
  brasileiros disponíveis não medem bem o que a gente quer medir. Isso vale como aviso pra
  qualquer avaliação que o Pocket-LM venha a fazer.

  > **Correção (27/07/2026): desatualizado.** O **PoETa v2** (arXiv 2511.17808, nov/2025)
  > encontra **correlação positiva clara** entre custo computacional e performance em
  > português, com os outliers explicados por arquitetura e composição de dados. A leitura
  > corrigida é: **o problema era o benchmark antigo, não a ausência de sinal**. Isso derruba
  > uma das justificativas para não investir em avaliação — e a avaliação foi definida:
  > **BPB congelado + CALAME-PT (acc) + CALAME-PT-NLL**, com ASSIN/Belebele só como *null
  > check*. Ver `varredura-institucional.md` §7.1/§10.1, `plano-avaliacao.md` §1 e
  > `PLANO.md` D1/D2/D3.
- **Escala de referência:** o menor modelo pt-BR "de verdade" (Tucano-160m) custou 200B tokens
  e ~350 GPU-horas de A100. O TTL-460m custou 280h de A100 pra 6,2B tokens. Um treino de
  poucas horas em L4 não chega perto — é outra ordem de grandeza.

### 1.2 Fine-tune / continued pretraining sobre modelos em inglês (NÃO são "do zero")

| Modelo | Params | Base | Quem | Licença |
|---|---|---|---|---|
| [maritaca-ai/sabia-7b](https://huggingface.co/maritaca-ai/sabia-7b) | 7B | LLaMA-1-7B + 10B tokens do ClueWeb22-pt | Maritaca AI | licença LLaMA-1 — **só pesquisa** |
| Sabiá-2 small/medium, [Sabiá 4 / Sabiazinho 4 / Sabiá 4 Thinking](https://www.maritaca.ai/) | não divulgado | fechado | Maritaca AI | **API-only, sem pesos** |
| [recogna-nlp/bode-7b-alpaca-pt-br](https://huggingface.co/recogna-nlp/bode-7b-alpaca-pt-br) / [13b](https://huggingface.co/recogna-nlp/bode-13b-alpaca-pt-br) | 7B/13B | Llama 2 + Alpaca traduzido | Recogna/UNESP ([paper](https://arxiv.org/pdf/2401.02909)) | herda Llama 2; tem [GGUF](https://huggingface.co/recogna-nlp/bode-7b-alpaca-pt-br-gguf) |
| openCabrita-3B ([paper](https://arxiv.org/abs/2308.11878)) | 3B | OpenLLaMA + continued pretraining pt + tokenizer novo | Celio Larcher et al. | aberta (OpenLLaMA) |
| [PORTULAN/gervasio-ptbr-base](https://huggingface.co/PORTULAN/gervasio-ptbr-base) | ~1B | **Pythia** (família GPT) + instrução em pt | PORTULAN CLARIN | ver card |
| [gervasio-8b-ptpt](https://huggingface.co/PORTULAN/gervasio-8b-portuguese-ptpt-decoder) / [70b](https://huggingface.co/PORTULAN/gervasio-70b-portuguese-ptpt-decoder) | 8B/70B | Llama 3.1 / 3.3 Instruct | PORTULAN | herda Llama |
| [nicholasKluge/Aira-2-portuguese-124M](https://huggingface.co/nicholasKluge/Aira-2-portuguese-124M) | 124M | GPT-2 + instruction tuning sintético | N. Kluge | aberta |
| [Latam-GPT](https://www.euronews.com/next/2026/02/12/what-is-latam-gpt-latin-americas-spanish-and-portuguese-ai-model) | 70B | Llama 3.1 + 300B tokens es/pt | CENIA (Chile) + 60 orgs, 15 países | aberta com restrições a big tech |
| [Amazônia IA / Amazônia 360](https://abes.org.br/en/apos-sucesso-do-amazonia-ia-widelabs-lanca-amazonia-360-primeira-plataforma-soberana-de-ia-do-brasil/) | n/d | n/d | WideLabs | **fechado, plataforma comercial** |

### 1.3 Encoders (não geram texto — citados só pra não confundir)

- [Albertina PT-BR/PT-PT](https://huggingface.co/PORTULAN/albertina-900m-portuguese-ptbr-encoder) (100M / 900M / 1.5B), família BERT/DeBERTa, PORTULAN.
- BERTimbau (usado inclusive pelo próprio time do Tucano como
  [filtro de qualidade de texto](https://huggingface.co/TucanoBR)).
- **Não servem** pro Pocket-LM: são encoders, não geram texto autoregressivamente.

### 1.4 O buraco real do português

Nesta faixa, ordenado por tamanho:

```
<50M   → NADA publicado em pt-BR. Buraco genuíno.
~124M  → Aira-2-portuguese-124M (GPT-2 fine-tune, não base)
~160M  → Tucano-160m, TeenyTinyLlama-160m   (base, do zero, Apache 2.0)
~460M  → TeenyTinyLlama-460m
~630M  → Tucano-630m
1B+    → Tucano-1b1/2b4, GlórIA-1.3B, Gervásio-1B, ...
```

O Pocket-LM (50–90M) cai **exatamente no buraco de <100M**. Isso é interessante — e é o
único argumento de ineditismo que sobrevive a este levantamento. Ver §4.

Fontes de acompanhamento: [Open Portuguese LLM Leaderboard](https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard),
[lista Portuguese-NLP](https://github.com/ajdavidl/Portuguese-NLP),
[modelos de linguagem para português — USP](https://sites.usp.br/keml/en/language-models-for-portuguese/).

---

## 2. Multilíngues pequenos que já falam português e cabem no celular

| Modelo | Params | Português? | Artefato on-device pronto |
|---|---|---|---|
| [Qwen3-0.6B](https://huggingface.co/litert-community/Qwen3-0.6B) | 0,6B | sim (multilíngue amplo) | **`.litertlm` INT8 (586 MB), mixed-INT4 (475 MB) e `mediatek.mt6993` a16w8 (992 MB)** + GGUF em toda parte |
| [Gemma-3-270M-it](https://huggingface.co/litert-community/gemma-3-270m-it) | 270M | vocab 256k, multilíngue por design; **feito pra fine-tune, não pra chat** | `.litertlm` e `.task` q8 (304 MB); [GGUF](https://huggingface.co/bartowski/google_gemma-3-270m-it-GGUF) |
| Gemma-3-1B | 1B | sim | `.litertlm`, GGUF, ExecuTorch |
| [Gemma 4 E2B / E4B](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) | 2B/4B efetivos | sim | `.litertlm` oficial; é o que o AI Edge Gallery usa |
| [SmolLM2 135M / 360M / 1.7B](https://huggingface.co/collections/HuggingFaceTB/smollm2) | 135M–1,7B | **português é uma das 6 línguas suportadas** | GGUF, ONNX, ExecuTorch |
| [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 3B | idem, +long context | `.litertlm` na litert-community |
| [LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) / [LFM2.5-350M](https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind) | 350M | multilíngue; 28T tokens de treino no 2.5 | GGUF, [ONNX](https://huggingface.co/onnx-community/LFM2-350M-ONNX); <500 MB |
| [EuroLLM-1.7B(-Instruct)](https://huggingface.co/utter-project/EuroLLM-1.7B-Instruct) | 1,7B | **português é língua de primeira classe** (projeto UE, IST Lisboa/Unbabel), 4T tokens, Apache 2.0 | GGUF de terceiros |
| Llama-3.2-1B, Qwen2.5-0.5B/1.5B, Phi-4-mini | 0,5–4B | sim | GGUF/`.litertlm`/MLC |
| TinyLlama-1.1B | 1,1B | fraco, inglês-centrado | GGUF — **datado, não usar como referência em 2026** |

**O ponto duro:** um modelo de 350M–600M multilíngue de 2025/26, treinado em trilhões de
tokens, escreve português melhor que qualquer coisa que 90M params treinados em alguns
bilhões de tokens vão escrever. Não é opinião — é orçamento de compute.

**Meio-termo mais rentável:** o [Gemma-3-270M foi explicitamente publicado para fine-tune
de domínio e de idioma](https://developers.googleblog.com/en/introducing-gemma-3-270m/)
(vocab de 256k justamente pra absorver tokens raros de outras línguas). Um LoRA em pt-BR
sobre ele custa horas, não semanas, e sai com artefato `.litertlm` pronto pro celular.

---

## 3. Projetos abertos de LLM on-device — e o caminho mais curto

| Stack | Formato | Android | NPU MediaTek | Status jul/2026 |
|---|---|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF | maduro (Termux, JNI, apps) | **não** — backends de NPU são [Hexagon/Qualcomm](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md); em MediaTek fica CPU/GPU | maior ecossistema, menor atrito |
| [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) | `.litertlm` / `.tflite` | oficial (Kotlin/C++) | **sim** — [NeuroPilot Accelerator](https://developers.googleblog.com/mediatek-npu-and-litert-powering-the-next-generation-of-on-device-ai/) | rota de produção pra NPU MediaTek |
| [ExecuTorch](https://github.com/pytorch/executorch) | `.pte` | AAR oficial | backend MediaTek em **alpha/beta** ([doc](https://docs.pytorch.org/executorch/stable/backends-mediatek.html)) | [1.0 em 2025](https://pytorch.org/blog/introducing-executorch-1-0/), série 1.3 em 2026 |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | compilado TVM | app MLCChat | não | ativo, bom em GPU |
| [MediaPipe LLM Inference](https://developers.google.com/edge/mediapipe/solutions/genai/llm_inference/android) | `.task` | sim | via LiteRT | **modo manutenção** — Google recomenda migrar pra LiteRT-LM |

### 3.1 O caminho mais curto pra ter LLM rodando no POCO **esta semana**

Em ordem de esforço crescente. Os dois primeiros não exigem escrever uma linha de código.

1. **Google AI Edge Gallery** (Play Store, app open-source do Google) — baixa Gemma 4 / Qwen /
   Phi e roda offline, com aba de benchmark por hardware.
   https://github.com/google-ai-edge/gallery/releases
   → **Isto é o "roda LLM no meu celular" em ~20 minutos.**
2. **PocketPal AI** (https://github.com/a-ghorbani/pocketpal-ai) ou
   [ChatterUI](https://github.com/Vali-98/ChatterUI) / [SmolChat](https://github.com/shubham0204/SmolChat-Android)
   — front-ends llama.cpp que aceitam **qualquer GGUF do HuggingFace**. É por aqui que dá pra
   rodar **um modelo pt-BR de verdade hoje**, ex.:
   [tensorblock/Tucano-1b1-GGUF](https://huggingface.co/tensorblock/Tucano-1b1-GGUF).
   → **E é por aqui que o Pocket-LM chega ao celular mais cedo:** converter o checkpoint pra
   GGUF (`convert_hf_to_gguf.py` + `llama-quantize`) e abrir no PocketPal. Sem NPU, sem SDK
   proprietário, sem app novo.
3. **LiteRT-LM CLI** — `litert-lm run --from-huggingface-repo=litert-community/Qwen3-0.6B`.
   Roda em desktop e Android, e é o caminho pro `.litertlm` da NPU.
4. **ExecuTorch XNNPACK int4** — a rota que o `docs/npu-deploy.md` já recomenda pra validar a
   esteira do *nosso* modelo em CPU antes de tentar NPU.
5. **NPU MediaTek** — só depois dos anteriores.

### 3.2 Realidade da NPU: dois avisos concretos

- **Bom:** existe `Qwen3-0.6B.mediatek.mt6993.litertlm` (a16w8, ctx 4096) publicado na
  [litert-community](https://huggingface.co/litert-community/Qwen3-0.6B). **MT6993 é o
  part number do Dimensity 9500** ([Geekbench/Gizmochina](https://www.gizmochina.com/2025/06/16/dimensity-9500-mt6993-geekbench-leak/)).
- **Ruim, e é preciso registrar:** o Dimensity **9500s** ([página MediaTek](https://www.mediatek.com/products/smartphones/mediatek-dimensity-9500s),
  anunciado jan/2026) é uma **variante cortada** — a [Notebookcheck](https://www.notebookcheck.net/Mediatek-Dimensity-9500s-Processor-Benchmarks-and-Specs.1314460.0.html)
  o descreve como compartilhando arquitetura com a geração 9400+, com NPU 890, e **não
  confirma que o part number seja MT6993**. Ou seja: **não está garantido que o artefato
  `mt6993` carregue no 9500s.**
- E mesmo no 9500 legítimo há atrito real: [LiteRT-LM issue #1437](https://github.com/google-ai-edge/LiteRT-LM/issues/1437)
  — `gemma3-270m` falhou na NPU de um OPPO Find X9 (MT6993) com
  `Failed to restore model from compiled network` / `unresolved custom op: DISPATCH_OP`,
  enquanto os **mesmos binários funcionaram num MT6991**. Binários de NPU são específicos por
  SoC e quebram.
- **Conclusão prática:** o primeiro teste no aparelho deve ser **CPU/GPU**, com um modelo
  pronto, pra descobrir logo em que categoria de compatibilidade o 9500s cai. Custa uma tarde
  e derruba (ou confirma) a maior incerteza da fase 3.

---

## 4. A pergunta estratégica — vale treinar do zero?

Resposta curta: **não, se o objetivo for um produto de chat pt-BR. Sim, se o objetivo for
aprendizado, controle, ou um nicho que ninguém ocupou.** Os dois são objetivos legítimos;
o erro é confundi-los.

### 4.1 Onde treinar do zero **ganha**

1. **Aprendizado — e é o argumento mais forte.** Dados → tokenizer → treino → avaliação →
   quantização → export → runtime Android. Ninguém aprende isso baixando GGUF. O Pocket-LM já
   produziu esteira de GPU headless, checkpoint retomável, cron noturno e um documento de NPU
   que a maioria dos "praticantes de IA" não saberia escrever. **Isso é o entregável real.**
2. **Controle total de licença e proveniência.** Todo modelo pt-BR aberto e bom hoje ou é
   Apache 2.0 mas "não destinado a deployment" (Tucano), ou herda licença Llama/LLaMA
   (Sabiá-7b é **research-only**), ou é fechado (Sabiá 4, Amazônia). Um modelo próprio, com
   corpus documentado, é limpo para uso comercial sem cláusula de terceiro. Vale quando o
   cliente é jurídico/governo/dados sensíveis.
3. **Tamanho ultra-pequeno (<100M) — o buraco real do português.** Ninguém publicou base pt-BR
   abaixo de ~124M. Se o requisito é `<50 MB` de artefato, latência de milissegundos e RAM
   mínima, não existe prateleira: Gemma-3-270M q8 são 304 MB, Qwen3-0.6B INT4 são 475 MB.
   Um modelo de 50M em int4 fica na casa de **dezenas de MB**. Aí não é competição, é o único
   candidato.
4. **Domínio estreito.** É a lição do [TinyStories](https://arxiv.org/pdf/2305.07759): abaixo de
   10M–33M de parâmetros já se obtém texto fluente e coerente **quando o domínio e o vocabulário
   são restritos**. Aplicado a pt-BR: autocomplete de campo de formulário, normalização de
   endereço, geração de descrição de produto num template, classificação/extração num nicho —
   coisas onde um modelo generalista de 600M é desperdício. Aqui um modelo próprio de 30–90M
   **ganha em latência, custo e previsibilidade**.
5. **Tokenizer 100% pt-BR.** É o achado do [Cabrita](https://arxiv.org/abs/2308.11878):
   tokenizer próprio reduz muito a contagem de tokens por texto em português. Menos tokens =
   menos compute por frase = mais tokens/s no celular. Vantagem pequena mas real, e some se
   você usa modelo de prateleira.
6. **Pesquisa/publicação/portfólio.** Tucano e TeenyTinyLlama viraram paper. É um caminho
   aberto e barato de credibilidade técnica.

### 4.2 Onde treinar do zero **claramente perde**

1. **Chat de propósito geral.** Sem chance. Qwen3-0.6B viu trilhões de tokens; um treino de
   horas em L4 vê alguns bilhões. A diferença aparece na primeira frase.
2. **Conhecimento de mundo e seguir instrução.** Emergem com escala de dados, não com
   arquitetura esperta. 90M não chega lá.
3. **Custo-benefício contra fine-tune.** LoRA de Gemma-3-270M ou Qwen3-0.6B em pt-BR: horas de
   GPU, artefato `.litertlm` pronto, qualidade muito superior. Treinar do zero pra atingir
   qualidade menor é gastar mais pra entregar pior — em qualquer tarefa não-nichada.
4. **Deploy na NPU.** Justamente onde parecia haver mérito próprio, já existe pré-compilado
   (`mediatek.mt6993`). E a parte difícil da fase 3 (SDPA decomposto, KV-cache estático,
   PTQ int4, runner Android) **é a mesma** com modelo próprio ou de terceiro — só que com
   modelo de terceiro dá pra debugar contra um baseline que funciona.
5. **Benchmarking honesto.** O próprio paper do Tucano avisa que os benchmarks pt disponíveis
   quase não correlacionam com escala de treino. Sem métrica confiável, "meu modelo ficou bom"
   é achismo. Se for treinar do zero, **defina antes** uma métrica de tarefa própria (perplexidade
   no seu domínio + avaliação humana em N amostras), não um leaderboard.

### 4.3 Recomendação

Rodar **duas trilhas paralelas**, sem misturar:

- **Trilha A (destrava a fase 3, custo ~1 dia):** instalar AI Edge Gallery/PocketPal no POCO,
  rodar Qwen3-0.6B e um Tucano-1b1 GGUF, medir tok/s em CPU e GPU, e tentar o
  `.litertlm mt6993` na NPU. Isso responde *"o 9500s aceita artefato de NPU?"* — a maior
  incerteza do projeto — antes de qualquer engenharia de export.
- **Trilha B (o Pocket-LM em si):** manter o treino do zero, mas **reposicionar o alvo**: em vez
  de "GPT pt-BR de uso geral", mirar **modelo de domínio estreito, <50M, que cabe onde nada
  cabe**. Escolher a tarefa antes de treinar mais. É onde os §4.1(3) e §4.1(4) se somam e o
  projeto deixa de competir num terreno já ocupado.
- **Se em algum momento aparecer prazo ou cliente:** fine-tune de Gemma-3-270M em pt-BR é o
  caminho pragmático, com `.litertlm` de saída. Não é a mesma coisa que treinar do zero — e
  não precisa ser.

---

## Fontes

**Modelos pt-BR**
- Tucano (paper): https://arxiv.org/abs/2411.07854 · modelos: https://huggingface.co/TucanoBR
- GigaVerbo: https://huggingface.co/datasets/TucanoBR/GigaVerbo
- TeenyTinyLlama: https://huggingface.co/nicholasKluge/TeenyTinyLlama-460m ·
  https://www.sciencedirect.com/science/article/pii/S2666827024000343
- Sabiá-7b: https://huggingface.co/maritaca-ai/sabia-7b · Maritaca (Sabiá 4): https://www.maritaca.ai/
- Bode: https://huggingface.co/recogna-nlp/bode-7b-alpaca-pt-br · https://arxiv.org/pdf/2401.02909
- Cabrita: https://arxiv.org/abs/2308.11878
- Gervásio / Albertina (PORTULAN): https://huggingface.co/PORTULAN/gervasio-ptbr-base ·
  https://portulanclarin.net/models/
- GlórIA: https://arxiv.org/abs/2402.12969 · https://huggingface.co/NOVA-vision-language/GlorIA-1.3B
- Aira-2-portuguese-124M: https://huggingface.co/nicholasKluge/Aira-2-portuguese-124M
- Latam-GPT: https://www.euronews.com/next/2026/02/12/what-is-latam-gpt-latin-americas-spanish-and-portuguese-ai-model
- Amazônia/WideLabs: https://abes.org.br/en/apos-sucesso-do-amazonia-ia-widelabs-lanca-amazonia-360-primeira-plataforma-soberana-de-ia-do-brasil/
- Open Portuguese LLM Leaderboard: https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard
- Lista Portuguese-NLP: https://github.com/ajdavidl/Portuguese-NLP

**Multilíngues pequenos**
- Qwen3-0.6B (LiteRT): https://huggingface.co/litert-community/Qwen3-0.6B
- Gemma-3-270M: https://developers.googleblog.com/en/introducing-gemma-3-270m/ ·
  https://huggingface.co/litert-community/gemma-3-270m-it
- Gemma 4 (edge): https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/
- SmolLM2/SmolLM3: https://huggingface.co/blog/smollm3 · https://github.com/huggingface/smollm
- LFM2 / LFM2.5-350M: https://huggingface.co/LiquidAI/LFM2-350M · https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind · https://arxiv.org/html/2511.23404v1
- EuroLLM: https://huggingface.co/utter-project/EuroLLM-1.7B-Instruct · https://arxiv.org/pdf/2409.16235
- TinyStories: https://arxiv.org/pdf/2305.07759

**On-device**
- llama.cpp: https://github.com/ggml-org/llama.cpp · backend Snapdragon/Hexagon:
  https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md
- LiteRT-LM: https://github.com/google-ai-edge/LiteRT-LM · issue MT6993:
  https://github.com/google-ai-edge/LiteRT-LM/issues/1437
- Google AI Edge Gallery: https://github.com/google-ai-edge/gallery/releases
- PocketPal AI: https://github.com/a-ghorbani/pocketpal-ai · SmolChat: https://github.com/shubham0204/SmolChat-Android
- ExecuTorch: https://github.com/pytorch/executorch · https://pytorch.org/blog/introducing-executorch-1-0/ ·
  backend MediaTek: https://docs.pytorch.org/executorch/stable/backends-mediatek.html
- MLC-LLM: https://github.com/mlc-ai/mlc-llm
- MediaPipe LLM Inference (manutenção): https://developers.google.com/edge/mediapipe/solutions/genai/llm_inference/android
- MediaTek NPU + LiteRT: https://developers.googleblog.com/mediatek-npu-and-litert-powering-the-next-generation-of-on-device-ai/
- Dimensity 9500 = MT6993: https://www.gizmochina.com/2025/06/16/dimensity-9500-mt6993-geekbench-leak/
- Dimensity 9500s: https://www.mediatek.com/products/smartphones/mediatek-dimensity-9500s ·
  https://www.notebookcheck.net/Mediatek-Dimensity-9500s-Processor-Benchmarks-and-Specs.1314460.0.html
- Awesome Mobile LLM: https://github.com/stevelaskaridis/awesome-mobile-llm
