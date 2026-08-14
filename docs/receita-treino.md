# Receita de treino — mais qualidade com o mesmo orçamento de GPU

**Status: COMPLETO** (27/07/2026).

Objetivo: dado o mesmo budget (~50 min de T4), o que mudar em `model.py` e `train_gpu.py`
para chegar num val loss materialmente melhor. Complementa `docs/estado-da-arte.md`
(o que já existe publicado) e `docs/corpus-ptbr.md` (de onde vêm os tokens);
restrições de export estão em `docs/npu-deploy.md`. Não repete o conteúdo deles.

**Resumo em uma frase:** metade do ganho disponível não é ciência — é um bug de schedule
e ~15% de MFU; a outra metade não é arquitetura — é a falta de tokens (25× abaixo de
Chinchilla). Arquitetura e otimizador valem ~2–3,5× de compute; os dados valem ~25×.

---

## 0. Baseline medido e o que dá pra ler dele

| Item | Valor |
|---|---|
| Params | 33.870.848 (8L × 8H × 512d, block 512, vocab 16.384, tied) |
| Não-embedding | ~25,2 M (embedding = 8,39 M = **24,8% do modelo**) |
| Corpus | 27 M tokens |
| Batch efetivo | 512 × 24 × 4 = **49.152 tokens/step** |
| Steps | 2.810 em 50 min → **138,1 M tokens processados = 5,1 épocas** |
| GPU | T4 (Turing: **sem bf16**, fp16 + GradScaler) |
| val loss | 9,82 → **3,63** (ppl 37,7; 5,24 bits/token) |
| Sintoma | português gramatical, factualmente sem sentido |

Três leituras imediatas, antes de qualquer paper:

**(a) MFU ~16%.** FLOPs por token ≈ `6N + 12·L·T·d` = 6·33,87e6 + 12·8·512·512 = 2,28e8.
Total = 2,28e8 × 1,381e8 = **3,15e16 FLOPs em 3.000 s = 10,5 TFLOP/s**. Um T4 faz
~65 TFLOP/s em fp16 tensor core → **MFU ≈ 16%**. Ou seja: **~2,5× do orçamento de GPU
já pago está sendo jogado fora** em kernel ruim de atenção, data loader síncrono e
grad-accum desnecessário. É ganho de qualidade "de graça", sem tocar em ciência.

**(b) O LR nunca decaiu.** `MAX_STEPS` default é 100.000, mas o run para em 2.810 pelo
`TRAIN_BUDGET_S`. O cosseno é calculado contra `MAX_STEPS`, então em `lr_at(2810)`:
`prog = (2810−200)/99800 = 0,026` → `lr = 0,958 × LR`. **O treino inteiro rodou a 96% do
LR de pico, sem fase de decaimento.** É onde mora a maior queda de loss de qualquer run.
Detalhe em §2.3 — é o bug mais barato de corrigir do projeto.

**(c) "Gramatical mas sem sentido" não é problema de arquitetura.** É o resultado esperado
de 27 M tokens. Gramática emerge cedo e barato; fato e coerência de mundo são função de
dados. Ver §4 e §5.

---

## 1. Arquitetura

Notação para a coluna NPU: as restrições vêm de `docs/npu-deploy.md` §5 — a APU MediaTek
é **integer-only** e **static-shape**, não tem exponencial nativa (o `exp` do softmax já é
gargalo), e `nn.MultiheadAttention` como bloco único não mapeia bem.

| Mudança | Ganho esperado | Custo | NPU |
|---|---|---|---|
| SDPA no lugar de `nn.MultiheadAttention` | **1,5–2,5× wall-clock** | ~30 linhas | **exigido** pelo export |
| Zero-init das projeções de saída | estabilidade + alguns % | 4 linhas | neutro |
| QK-norm | permite LR 2–3× maior | ~6 linhas | **ajuda** a quantização |
| RoPE no lugar de `nn.Embedding` de posição | poucos % de loss + generaliza comprimento | ~25 linhas | seguro |
| ReLU² no lugar de GELU | ~neutro em loss, mais rápido | 2 linhas | **melhor** (sem transcendental) |
| RMSNorm no lugar de LayerNorm | ~neutro em loss, ~5% de velocidade | 10 linhas | ok, risco baixo |
| `bias=False` nos Linear | marginal | 3 linhas | **ajuda** (menos acumulador int32) |
| SwiGLU | pequeno ganho de loss | ~15 linhas | pior que ReLU² (sigmoid) |
| Untied embeddings | **negativo aqui** | — | +4,2 MB no artefato int4 |
| GQA | **negativo aqui** | — | irrelevante nesse tamanho |

### 1.1 Atenção: `nn.MultiheadAttention` → SDPA decomposto — **fazer primeiro**

Hoje: `nn.MultiheadAttention(batch_first=True)` com `attn_mask` booleana fatiada por `T`.
Dois problemas somados:

1. **Velocidade.** O fast-path fundido do `nn.MultiheadAttention` é desativado quando você
   passa uma `attn_mask` explícita. Resultado: materializa a matriz `B×H×T×T` de atenção em
   memória, com transposições extras do `batch_first`. Com `q,k,v` próprios e
   `F.scaled_dot_product_attention(q, k, v, is_causal=True)` você garante o kernel fundido.
   No T4 (sm_75) o backend FlashAttention do PyTorch **não** roda (exige sm_80+), mas o
   `EFFICIENT_ATTENTION` (mem-efficient) roda e já evita materializar `T×T` — é a maior
   parte do ganho. Numa L4/A10G (sm_86+) entra o Flash de verdade.
2. **Export.** `docs/npu-deploy.md` §5 lista isso como a tarefa central da fase 3: a atenção
   precisa virar Q/K/V lineares + matmul + softmax + matmul decompostos, e a máscara precisa
   ter shape fixo em vez de `mask[:T,:T]`. Fazer agora mata dois coelhos — o `model_export.py`
   deixa de ser uma reescrita e vira só "a mesma coisa com KV-cache".

Junto vem: `bias=False` nas projeções (estilo Llama). Menos params, menos termo de
acumulador int32 na GEMM quantizada, e nenhuma perda de loss documentada.

### 1.2 Inicialização — dois erros silenciosos no `_init` atual

`_init` aplica `std=0.02` a **todo** `nn.Linear`, inclusive as projeções de saída do bloco
(`attn.out_proj`, `mlp[2]`). Com pre-norm e resíduo, a variância do stream residual cresce
com a profundidade, o que na prática age como um LR inicial inflado.

- **Correção clássica (nanoGPT):** escalar o std das projeções residuais por
  `1/sqrt(2·n_layer)` → 0,02/4 = 0,005 para 8 camadas.
- **Correção do speedrun (melhor):** **zero-init** das projeções de saída. Cada bloco começa
  como identidade e "liga" sozinho. No `modded-nanogpt` isso faz parte do pacote
  ReLU² + QK-norm + zero-init que cortou o tempo de **22,3 → 15,2 min (−31,8%)**.
  Custo: 4 linhas.

Terceiro item relacionado: com weight tying, `self.tok.weight = self.head.weight` é
atribuído **antes** do `self.apply(self._init)`, então a matriz compartilhada é inicializada
duas vezes (uma pelo ramo `nn.Linear`, uma pelo `nn.Embedding`). Inofensivo hoje porque os
dois usam `std=0,02` — mas quebra em silêncio no minuto em que você mudar um dos dois
(ex.: init de embedding com `std=1/sqrt(d)`). Vale reordenar ou comentar.

### 1.3 QK-norm — melhor relação ganho/esforço da seção

RMSNorm aplicado em `q` e `k` (sobre `head_dim`) antes do produto escalar.

- **Ganho:** o efeito principal não é loss por si, é **destravar LR muito mais alto sem
  spike**. Impede o crescimento descontrolado dos logits de atenção (*entropy collapse*),
  que é o modo de falha típico ao subir o LR num modelo pequeno. Faz parte do pacote de
  −31,8% do speedrun e é padrão em Gemma-3, Qwen3 e OLMo-2.
- **Custo:** ~6 linhas, sem parâmetro novo se usar RMSNorm sem escala aprendida.
- **NPU:** dois norms a mais por bloco — suportado. E há um bônus real: **QK-norm limita a
  magnitude de `q·k`, que é exatamente o que faz a atenção int8 ficar bem-condicionada.**
  É aliado da quantização, não obstáculo.

### 1.4 RoPE vs embedding de posição aprendida

- **Ganho.** O record #2 do `modded-nanogpt` (45 → 31,4 min) foi "tuned LR + rotary
  embeddings". Isolado, RoPE num modelo de 30–100 M vale poucos por cento de loss a passos
  iguais — mas: (i) libera 262 k params (`512×512`) que hoje só decoram posição absoluta;
  (ii) modela distância **relativa**, que é o que importa; (iii) permite mudar `block_size`
  entre treino e deploy sem reaprender posição, o que vale muito na fase 3.
- **Custo.** ~25 linhas: tabela `cos/sin` pré-computada como buffer e rotação em `q,k`.
- **NPU.** Seguro. `docs/npu-deploy.md` §5 já lista RoPE entre as ops que a NPU aceita bem,
  e **todos** os LLMs pré-compilados pra MediaTek citados em `estado-da-arte.md`
  (Qwen3-0.6B `mt6993`, Gemma-3-270M/1B) usam RoPE. A tabela vira constante de shape
  estático. Ressalva de quantização: manter a rotação em precisão maior (A16 pra q/k) — é
  exatamente o que o `NeuropilotQuantizer` expõe com A16W8/A16W4.
- **Nota:** `npu-deploy.md` §5 diz "não precisamos migrar pra RoPE por causa de hardware".
  Continua verdade — o argumento aqui é **qualidade e flexibilidade de contexto**, não NPU.

### 1.5 RMSNorm vs LayerNorm

- **Ganho:** loss praticamente neutro. O ganho é custo — sem média, sem bias, menos ops.
  Na prática ~5% do tempo de step num modelo pequeno, onde o norm pesa relativamente mais.
- **Custo:** 10 linhas (ou `nn.RMSNorm`, nativo no PyTorch 2.4+).
- **NPU:** Gemma-3-270M roda RMSNorm na NPU MediaTek hoje, então é caminho batido. Risco
  residual: alguns conversores têm op fundida pra LayerNorm e decompõem RMSNorm em
  `mean/rsqrt/mul` — ainda roda, mas se cair pra CPU custa latência. Como o ganho de loss é
  ~zero, **é a mudança mais fácil de reverter se o export reclamar.**
- Variante do speedrun: RMSNorm **sem escala aprendida** (`elementwise_affine=False`).
  Menos params, mais estável e simplifica a quantização.

### 1.6 MLP: GELU 4× vs SwiGLU vs ReLU²

| | Loss | Custo compute | NPU |
|---|---|---|---|
| GELU 4× (atual) | referência | 2 matmuls + `erf`/`tanh` | transcendental, aproximado por LUT |
| SwiGLU (hidden ≈ 8/3·d) | pequeno ganho consistente (Shazeer, *GLU Variants*) | 3 matmuls | `sigmoid` = transcendental |
| **ReLU²** (`relu(x)**2`) | ~empate com os dois em escala pequena | 2 matmuls, ativação trivial | **`max(0,x)` + `mul` — aritmética inteira pura** |

`docs/npu-deploy.md` §5 registra que a APU **não tem exponencial nativa** e que o `exp` do
softmax já é gargalo. GELU e SiLU dependem de transcendental; ReLU² não depende de nada.
E ReLU² é exatamente o que o `modded-nanogpt` usa em produção de record.

**Recomendação:** trocar GELU por **ReLU²** (2 linhas, mantém `4×` e a contagem de params).
Testar SwiGLU depois como experimento isolado — se adotado, usar `hidden = 8/3·d ≈ 1408`
pra manter params, e aceitar que o export fica um pouco pior.

### 1.7 Pre-norm vs post-norm

Já é **pre-norm** (`ln1 → attn → residual`, `ln2 → mlp → residual`) com `lnf` final. Está
certo, não mexer: post-norm exige warmup longo e é instável em modelo pequeno sem truque de
init. Duas variantes valem consideração:

- **Peri-/sandwich norm** (Gemma-2): norm antes **e** depois de cada sublayer. Melhora
  estabilidade em LR alto; custo pequeno. Opcional, e em boa medida redundante com QK-norm.
- **Logit softcap** (`tanh`, valor 15 no speedrun): estabiliza a head. Útil no treino, mas
  `tanh` é transcendental → **remover ou dobrar na quantização no export**. Só adotar se
  aparecer instabilidade depois de subir o LR.

### 1.8 Untied vs tied embeddings — **manter tied**

A conta que decide: `16.384 × 512 = 8,39 M` params, **24,8% do modelo**. Destravar dobra
isso: 33,9 M → 42,3 M (+25% de params, +25% de FLOPs/token, +4,2 MB no artefato int4 — contra
a meta de "dezenas de MB" de `estado-da-arte.md` §4.1.3).

- Untied ganha por passo **quando a embedding é fração pequena do modelo**. Aqui é um quarto.
- Todos os pares de referência do nosso tamanho **amarram**: Gemma-3-270M, Qwen3-0.6B,
  SmolLM2-135M.
- O `modded-nanogpt` desamarra, mas só **no último terço do treino** — trata untying como
  refinamento de fim de run, não como arquitetura.

**Decisão: manter tied**, e recuperar a maior parte do benefício com duas coisas baratas:

1. **Tirar a embedding do weight decay** (hoje `wd=0.1` cai em cima da matriz compartilhada,
   §2.2). Sozinho já é boa parte do que "untied" resolve.
2. **Escala aprendida na saída** (um escalar multiplicando os logits), pra a head não ser
   obrigada a herdar a magnitude que a embedding quer.

**Item a medir separado — tamanho do vocab.** Com 27 M–1 B tokens, um vocab de 16.384 deixa
muitas linhas da embedding vistas pouquíssimas vezes. Cair pra 8.192 reduz a embedding a
4,2 M params e dobra os updates por linha; o custo é mais tokens por texto (mais compute por
frase e menos contexto útil em 512 posições). Não é óbvio dos dois lados — é um experimento
de 25 min, não uma decisão de escrivaninha. (Ver o achado do Cabrita em
`estado-da-arte.md` §4.1.5 e o resultado análogo do *Regional Tiny Stories* em §5.)

### 1.9 GQA — **não fazer agora**

- **Não resolve problema que a gente tenha.** KV-cache estático int8 com 8 camadas,
  512 posições e 512 de dim = `2 × 8 × 512 × 512 B` ≈ **4,2 MB**. Não é gargalo de nada.
- **Custa qualidade, e mais em modelo pequeno.** Ablação em 188 M params: val ppl 41,94 (MHA)
  → **44,16 (GQA-4)**, ~5% pior. Modelos menores sofrem proporcionalmente mais porque o
  KV-cache já é barato — você paga a degradação sem receber a economia.
- **Quando reconsiderar:** `block_size` ≥ 2048 ou modelo acima de ~150 M. Aí GQA/MQA é bem
  suportado na NPU (Gemma-3 usa), mas continua sendo otimização de **deploy**, não de treino.

---

## 2. Otimizador e schedule

### 2.1 Muon — o maior ganho isolado documentado

- **Evidência 1 (speedrun):** trocar AdamW por Muon foi o **maior salto individual** da
  história do `modded-nanogpt`: 31,4 → 24,9 min, **−20,7%**.
- **Evidência 2 (paper controlado):** *Fantastic Pretraining Optimizers and Where to Find
  Them* (arXiv 2509.02046) mede otimizadores matriciais (Muon, Soap, Kron) contra AdamW com
  tuning honesto dos dois lados: **1,4× de speedup em 130 M params**, caindo pra 1,1× em
  1,2 B. **Modelo pequeno é exatamente onde Muon paga mais** — o nosso caso.
- **Evidência 3 (regime):** o mesmo paper mostra Muon dominando em **razão dados/params
  baixa**, e sendo superado por Soap/Kron quando a razão passa de ~8×. Hoje processamos
  138 M/33,9 M ≈ **4,1×** → território do Muon. Se a §4 for seguida e o corpus crescer muito,
  reavaliar (Soap custa mais memória de estado, mas 33 M cabe folgado).
- **Como aplicar:** Muon é para as **matrizes 2D escondidas** (q,k,v,o e MLP). Embedding,
  head e todo parâmetro 1D (norms, bias) continuam em AdamW. Referências de hiperparâmetro:
  Muon lr ~2e-2 (bem maior que AdamW), momentum Nesterov 0,95 com warmup próprio, 5 passos
  de Newton–Schulz, e o reescalonamento "RMS igual ao do AdamW" (`0,2·sqrt(max(n,m))`) que o
  Moonlight popularizou.
- **Custo:** ~60 linhas numa GPU só; há implementações de referência prontas
  (`KellerJordan/Muon`, e a do Moonlight). Overhead <2% normalmente — **no T4 fica pior
  porque não há bf16** e o Newton–Schulz roda em fp32 (estimar 3–5%). Segue muito positivo.
- **Risco:** médio. Precisa de LR próprio e da separação correta de grupos; jogar embedding
  no Muon degrada. É a mudança com maior payoff **e** maior chance de bug desta lista — por
  isso está depois do P0/P1 na §6.

### 2.2 AdamW — três hiperparâmetros errados hoje

Mesmo que Muon fique pra depois, isto vale a pena e custa minutos:

1. **`weight_decay=0.1` está caindo em tudo.** `AdamW(model.parameters(), ...)` aplica decay
   em LayerNorm, bias **e na embedding compartilhada**. O padrão do próprio nanoGPT é
   separar dois grupos: decay só em tensores com `dim >= 2`, `wd=0` no resto. Num modelo onde
   a embedding é 24,8% dos params e é *tied* com a head, decair a embedding é ativamente
   ruim. **8 linhas.**
2. **`betas=(0.9, 0.95)` com batch de 49 k tokens.** 0,95 vem de treinos com batch de
   0,5–4 M tokens. Com batch 10× menor o segundo momento fica ruidoso e precisa de mais
   suavização — a literatura de batch pequeno recomenda **β₂ = 0,98–0,99** nesse regime.
3. **`lr=6e-4` é herança do GPT-2 124 M** (768 de largura, batch 0,5 M tokens). Nosso modelo
   é mais estreito e o batch é 10× menor; modelo pequeno tolera LR bem mais alto,
   especialmente **depois** de QK-norm + zero-init (§1.2, §1.3). Varrer **1e-3 a 3e-3**.
   As leis empíricas de hiperparâmetro (*Predictable Scale* / Step Law, arXiv 2503.04715:
   `η_opt ∝ C^-0.125`, `B_opt ∝ C^0.327`) apontam na mesma direção pro nosso C minúsculo:
   **batch pequeno, LR relativamente alto**.

### 2.3 Schedule — corrigir o bug e trocar cosseno por WSD

**O bug (§0b).** `MAX_STEPS=100000` + parada por `TRAIN_BUDGET_S` ⇒ o cosseno foi percorrido
até 2,6% e o LR terminou em **0,958 × LR**. O treino inteiro rodou em LR de pico. Toda a
literatura de cooldown (WSD, decay-to-zero) mostra que a queda mais rápida de loss acontece
**durante o decaimento** — a gente simplesmente não fez essa fase. Não é sutileza de
0,01 nat: é a diferença entre um run que terminou e um run interrompido no meio.

**Correção mínima:** estimar os steps reais (medir o tempo de step nos primeiros ~50 e
calcular `steps_previstos = BUDGET_S / step_time`) e usar isso como horizonte do cosseno.
Também trocar o piso de `0,1·LR` por ~0 — a evidência de *decay-to-zero* é que decair até
perto de zero bate "cosseno até 10% do pico", com economia de compute relatada de até 60%.

**Correção melhor: WSD (warmup–stable–decay).**

```
warmup  ~2–3% dos steps   (linear 0 → LR)
stable  ~75–80%           (LR constante)
decay   ~20%              (linear ou 1−sqrt, LR → ~0)
```

Por que WSD é a escolha certa **especificamente para este projeto**:

- **Não exige saber o total de steps de antemão.** Nosso treino é limitado por wall-clock
  (`TRAIN_BUDGET_S`), não por steps. Com WSD você roda estável e dispara o cooldown quando
  restar 20% do orçamento de **tempo**. O cosseno é estruturalmente incompatível com um
  budget em segundos — foi exatamente isso que gerou o bug.
- **Checkpoint do meio do run é reaproveitável.** Dá pra retomar do fim da fase estável e
  fazer outro cooldown, com outros dados. Com cosseno, retomar exige refazer o schedule.
  Isso conversa direto com o `ckpt.pt` retomável que a esteira já tem.
- **É como o MiniCPM mediu lei de escala barato** — e foi assim que chegou ao número da §4
  (razão dados/params compute-ótima de **192×**, muito acima do 20× do Chinchilla).
- **Truque de cooldown que vale ouro aqui:** o MiniCPM injeta os **dados de melhor
  qualidade** só na fase de decay. Com o corpus de `docs/corpus-ptbr.md`, isso significa:
  web filtrada (`gigaverbo-v2`, `edu_int_score>=3`) na fase estável, e
  `gigaverbo-v2-synth` + `Boakpe/historinhas` concentrados no cooldown. Ganho de qualidade
  percebida desproporcional ao custo, e não muda uma linha do modelo.

### 2.4 Batch, grad-accum e data loader — dinheiro parado

- **`GRAD_ACCUM=4` com `BATCH=24` num modelo de 33 M numa T4 de 16 GB não serve pra nada.**
  24×512 = 12 k tokens por micro-step; dá pra rodar `BATCH=48–96` com `GRAD_ACCUM=1`.
  Mesmos FLOPs, GEMMs maiores, 1 step de otimizador em vez de 4 passagens.
- Sinal contrário a considerar: *Small Batch Size Training for Language Models*
  (arXiv 2507.07101) mostra que **batch pequeno com otimizador bem tunado vai muito bem** e
  dá mais updates por token. O ótimo provavelmente é **batch efetivo de 16 k–32 k tokens com
  `GRAD_ACCUM=1`** (mais steps que hoje) e β₂ mais alto. Ou seja: reduzir o batch **efetivo**
  e ao mesmo tempo eliminar a acumulação.
- **O dataset inteiro cabe na GPU.** 27 M tokens × 2 B = **54 MB**. Hoje `get_batch` faz
  list-comprehension em Python + `np.stack` + `.to(dev)` **síncrono**, 4× por step. Subir
  `train.bin` inteiro pra VRAM como tensor `uint16` e indexar lá elimina o data loader do
  caminho crítico. Provavelmente 10–20% do wall-clock. (Quando o corpus for pra 1 B tokens
  = 2 GB, ainda cabe numa L4 de 24 GB; num T4 de 16 GB usar `np.memmap` + pinned + prefetch.)
- **`estimate_loss` usa batches aleatórios.** O val loss reportado carrega ruído de
  amostragem entre avaliações, o que atrapalha comparar duas receitas. Fixar um conjunto
  determinístico de `EVAL_ITERS` batches (offsets sorteados uma vez, seed fixa). Sem isso,
  nenhuma das mudanças desta receita é mensurável com confiança.

### 2.5 `torch.compile` e precisão

- **`torch.compile(model)`**: tipicamente 1,3–1,8× em modelo classe nanoGPT, e nossos shapes
  já são estáticos. Custa ~1 min de compilação num budget de 50 — vale. Fazer **depois** do
  SDPA (§1.1), senão o compile tem que engolir o `nn.MultiheadAttention` inteiro.
- **T4 não tem bf16 nem TF32** (Turing). O código já detecta e cai pra fp16 + GradScaler,
  o que está correto. Mas é argumento forte pra **trocar de GPU se o orçamento for em
  dinheiro e não em horas**: uma L4 faz ~121 TFLOP/s bf16 contra ~65 do T4, com bf16 (sem
  GradScaler, sem risco de overflow) e preço/hora comparável. É o "mesmo orçamento, 2× de
  compute" mais fácil que existe.
- `torch.cuda.amp.GradScaler` está deprecado em favor de `torch.amp.GradScaler("cuda")` —
  cosmético, mas some um warning por run.

---

## 3. Speedruns — o que é portável do `modded-nanogpt`

Repo: https://github.com/KellerJordan/modded-nanogpt — alvo fixo de **val loss ≤ 3,28 no
FineWeb** (a réplica GPT-2 small do `llm.c` de Karpathy), 8×H100, ~85 records de 45 min
(mai/2024) até **~1,27 min** (mai/2026). ~35× de wall-clock no mesmo alvo de loss.

Ganhos medidos por mudança (ablações do repo):

| Mudança | Efeito medido |
|---|---|
| AdamW → **Muon** | 31,4 → 24,9 min (**−20,7%**) — maior salto individual |
| **ReLU² + QK-norm + zero-init** (pacote) | 22,3 → 15,2 min (**−31,8%**) |
| **FlexAttention** ctx 64 k (no lugar de causal densa 1024) | 7,2 → 5,03 min (−30,1%) |
| Tuned LR + **rotary** | 45 → 31,4 min |
| QKVO fundidos + Muon em batch | −24% de overhead |
| NorMuon | 2,476 → 2,345 min (−5,3%) |
| Multi-token prediction | 2,037 → 1,988 min (−2,4%) |

**Portável e vale a pena:**

- **Muon** (§2.1) — a maior peça.
- **ReLU² + QK-norm + zero-init** (§1.2, §1.3, §1.6) — o pacote com melhor relação
  ganho/linha de código de toda esta receita.
- **Rotary** (§1.4).
- **WSD com piso ~0 e cooldown longo** (§2.3) — e conserta um bug real.
- **Momentum warmup do Muon** — 3 linhas, vem junto com o Muon.
- **Norms sem escala aprendida** (§1.5) — simplifica a quantização.
- **QKV fundido num único `nn.Linear(d, 3d)`** — sai de graça ao reescrever a atenção (§1.1).

**Não portável, e por quê:**

- **FlexAttention com janela deslizante longa-curta / ctx 64 k** — o ganho vem de treinar em
  contexto muito longo com esparsidade. Nosso `block_size` é 512 e a NPU exige shape
  estático; janela variável é o oposto do que `npu-deploy.md` pede. **Descartar.**
- **FP8 nas GEMMs** — exige Hopper (H100). T4/L4 não têm. **Descartar.**
- **Skip connections U-net / value embeddings / MUDD gates** — funcionam, mas são ops
  não-padrão que o partitioner do ExecuTorch/LiteRT provavelmente joga pra CPU, e o ganho por
  linha de código é baixo comparado ao pacote acima. **Fora da fase 1.**
- **Logit softcap com `tanh`** — útil no treino, transcendental no export (§1.7).
- **Multi-token prediction** (−2,4%) — real, mas é a última casa decimal. Vantagem: a head
  extra é descartada na inferência, então **não custa nada em NPU**. Candidato razoável pra
  uma segunda rodada, não pra primeira.

**Ressalva de leitura, importante.** Os records são **wall-clock a loss fixa em 8×H100**, num
regime de ~1 época sobre FineWeb (inglês, dados praticamente infinitos). Nosso gargalo é o
oposto: **dados** (§4), numa GPU só, com 5 épocas. Os ganhos de **otimizador e arquitetura**
transferem (são independentes de dados); os de **escala de contexto e paralelismo** não.
Não esperar 35×. Esperar, somando §1 + §2, algo entre **2× e 3,5× de eficiência de compute**
— e a maior parte vem de MFU (§0a) e do bug de schedule (§0b), não de ciência nova.

---

## 4. Leis de escala — a conta com os nossos números

### 4.1 Chinchilla clássico (Hoffmann et al., 2022 — `D ≈ 20·N`)

```
N (params totais)          = 33.870.848
D (tokens únicos)          = 27.000.000
D (tokens processados)     = 2.810 × 49.152 = 138.117.120   (5,12 épocas)

Chinchilla para o nosso N:  D_opt = 20 × 33,87 M = 677 M tokens
Déficit de dados únicos:    677 M / 27 M     = 25,1×   ← o número que manda
Déficit de tokens vistos:   677 M / 138 M    = 4,9×
Razão atual tokens/param:   4,08 (processados) · 0,80 (únicos)
```

**Veredito 1: o modelo está fortemente SUB-TREINADO em dados** — 25× abaixo do mínimo
Chinchilla em tokens únicos.

> **Correção (27/07/2026): o número mudou de ordem de grandeza — para melhor.** O corpus foi
> re-tokenizado e hoje `data_big/train.bin` + `val.bin` = **202.630.887 tokens únicos**
> (800 MB de Wikipedia-pt, 3,948 bytes/token), contra os 27 M desta conta. O déficit
> Chinchilla cai de **25,1× para 3,3×**. Consequências: (a) as épocas deixam de ser um
> problema (§4.4 supõe 5,12 épocas sobre 27 M); (b) **os cálculos de dimensionamento de §4.2,
> §4.4 e §4.6 foram todos feitos com o corpus antigo e precisam ser refeitos**; (c) o que
> continua faltando **não é volume, é diversidade de registro** — o corpus segue sendo
> Wikipedia-pt puro, que é exatamente o diagnóstico da §5. Ver `PLANO.md` §1.4.

### 4.2 O ponto compute-ótimo do orçamento que já gastamos

```
C = 6·N·D = 6 × 33,87e6 × 138,12e6 = 2,807e16 FLOPs
Chinchilla com D = 20N  ⇒  C = 120·N²
N* = sqrt(C/120) = sqrt(2,339e14) = 1,53e7  ≈  15,3 M params
D* = 20·N*                                   ≈  306 M tokens
```

**Veredito 2: para 50 min de T4, o modelo compute-ótimo é de ~15 M params, não 33,9 M.**
Estamos **2,2× grandes demais e 2,2× com poucos tokens** para o compute gasto. Esse é o
desequilíbrio *menos* grave dos três — mas é o mais barato de corrigir (é uma env var).

### 4.3 Quanto vale cada eixo — usando a forma funcional do Chinchilla

`L(N,D) = E + A/N^α + B/D^β`, com `E=1,69  A=406,4  α=0,34  B=410,7  β=0,28`.
**Os valores absolutos não transferem** (constantes ajustadas em inglês, com outro
tokenizer — a fórmula prevê 4,97 e nós medimos 3,63, porque Wikipedia-pt é muito mais
previsível que web geral). Mas as **derivadas** transferem, e é delas que precisamos:

| Termo | Valor no ponto atual |
|---|---|
| `A/N^0.34` (capacidade) | **1,119** |
| `B/D^0.28` (dados) | **2,159** ← quase o dobro |

O termo de dados domina o de capacidade por ~2×. Simulando movimentos:

| Movimento | Δ loss prevista |
|---|---|
| 5× mais tokens (138 M → 677 M) | **−0,77** |
| 2× mais params (33,9 M → 67,7 M), tokens iguais | −0,24 |
| Encolher pra ~17 M **e** usar o compute liberado pra 2,2× mais tokens | **−0,13** |

Ou seja, com o mesmo dinheiro: **mais tokens vale ~3× mais que mais params**, e reequilibrar
o tamanho do modelo vale um extra pequeno mas real. Isso é aritmética, não opinião.

> **Correção (27/07/2026): a direção está certa e agora tem medição — mas só de um lado.**
> Os runs A (33,87 M) e B (219,4 M) deram dois pontos no eixo `N` a tokens quase iguais
> (138,1 M vs 125,8 M): o α **medido neste corpus é 0,046–0,090** conforme `E`, contra o 0,34
> da literatura — **de 3,8× a 7,4× menor**. 6,5× os params compraram 0,1175 nat, **22% do
> previsto** por esta tabela. Ou seja, a linha "2× mais params = −0,24" está **otimista por
> ~6×**; a diferença entre os eixos é **muito maior** que os 3× aqui estimados.
> **E o eixo `D` continua sem nenhum ponto medido:** `β = 0,28` e `B = 410,7` são importados
> do inglês, exatamente como o α que desabou ao ser medido. A próxima rodada (`PLANO.md` §3
> item 4) existe para medir β. Ver `analise-runs.md` §2 e `PLANO.md` §1.2.

### 4.4 Correção do regime "dados escassos" (Muennighoff et al., arXiv 2305.16264)

Até **~4 épocas**, repetir dados custa quase nada em relação a dados novos; a partir daí o
valor de cada repetição decai geometricamente e o retorno de mais compute vai a zero.
Estamos em **5,12 épocas** — bem no limite do "de graça", já começando a pagar.

```
Tokens "efetivos" ≈ 27 M × 4 = 108 M
N_opt sobre dados efetivos    = 108 M / 20 = 5,4 M params
```

**Veredito 3: sob a lente dos dados que temos, 33,9 M params é ~6× grande demais.**
E repetir mais épocas do mesmo corpus **não** resolve — foi o que o run fez.

### 4.5 Correção de deploy: Chinchilla é ótimo de TREINO, não de INFERÊNCIA

O Pocket-LM vai rodar num celular, muitas vezes, com custo de inferência que importa. Para
esse caso a literatura manda ir **muito além** de 20 tok/param:

| Referência | tokens/param |
|---|---|
| Chinchilla (2022) | 20 |
| **MiniCPM** (medido com WSD) | **192** |
| SmolLM2-135M (2 T tokens) | ~14.800 |
| *Beyond Chinchilla-Optimal* (Sardana & Frankle, arXiv 2401.00448) | qualidade ainda melhora até **10.000** |

Aplicado a 33,9 M params: MiniCPM pediria **6,5 B tokens**. O alvo prático de
`docs/corpus-ptbr.md` (0,8–1,5 B tokens para 33 M ⇒ 24–44 tok/param) é o compromisso
razoável entre a teoria e uma rodada de GPU de 1 h.

### 4.6 O que cabe de fato numa rodada — e a configuração que eu rodaria

FLOPs/token = `6N + 12·L·T·d`. Combinando com MFU realista:

| Cenário | MFU | Tokens em 50 min | tok/param |
|---|---|---|---|
| **Hoje** (T4, 33,9 M) | 16% | 138 M | 4,1 |
| T4, 33,9 M, com §1+§2 | ~40% | ~342 M | 10,1 |
| T4, **17 M params** (6L × 384d), com §1+§2 | ~40% | **~670 M** | **39** |
| L4, 33,9 M, com §1+§2 | ~40% | ~637 M | 18,8 |
| L4, **17 M params**, com §1+§2 | ~40% | ~1,25 B | 73 |

Duas leituras:

1. Com **17 M params numa T4 corrigida**, 50 min já dão ~670 M tokens = **39 tok/param** —
   quase o dobro do Chinchilla e dentro da faixa "over-training pra deploy" de §4.5.
   **Isso é atingível esta semana, sem trocar de GPU.**
2. Manter 33,9 M exige ~680 M tokens só pra empatar com Chinchilla → precisa da L4, ou de
   duas rodadas de T4 retomando o checkpoint (que a esteira já suporta).

E o bônus estratégico: **17 M params cai exatamente no buraco `<50 M` do português** que
`estado-da-arte.md` §1.4 identificou como o único terreno não ocupado. Encolher o modelo
não é rebaixar o projeto — é ir pro único lugar onde ele pode ser inédito.

**Configuração sugerida para o próximo run (T4, 50 min):**

```
N_LAYER=6  N_HEAD=6  N_EMBD=384  BLOCK=512  vocab 16k tied   → ~17 M params
BATCH=64   GRAD_ACCUM=1                                       → 32.768 tok/step
LR=1.5e-3 (varrer 1e-3…3e-3)  betas=(0.9, 0.98)  wd=0.1 só em dim>=2
schedule WSD: warmup 3% · stable 77% · decay 20% até ~0, disparado por tempo restante
corpus: Mistura C do corpus-ptbr.md (70% gigaverbo-v2 edu>=3 + 30% historinhas), >= 700 M tokens
```

### 4.7 Resposta direta à pergunta

**Sub ou super-treinado?** As duas coisas, em eixos diferentes:

- **Super-parametrizado** para o compute (2,2×) e para os dados que existem no disco (6×).
- **Sub-treinado** em tokens por um fator de **25×** contra Chinchilla, e de **30–240×**
  contra o alvo de deploy da §4.5.

O gargalo **não é** arquitetura, nem otimizador, nem params. É tokens. §1 e §2 valem
2–3,5× de eficiência de compute; a §4 vale 25× de qualidade de dados. Fazer §1 e §2 primeiro
porque são baratos e porque **multiplicam** o valor de cada token novo — mas não confundir
otimização de esteira com solução do problema.

---

## 5. TinyStories e coerência com poucos parâmetros

Esta é a seção que explica o sintoma "gramatical mas factualmente sem sentido".

### 5.1 O que o TinyStories provou (arXiv 2305.07759)

- Um **GPT-Neo de 28–33 M params** (ctx 512, window 256), treinado em histórias sintéticas
  com vocabulário de criança de 3–4 anos, gera **parágrafos fluentes, gramaticalmente quase
  perfeitos, com consistência de enredo e traços de raciocínio**.
- No domínio das histórias, a completação do modelo de **28 M** é comparável ou melhor que a
  do **GPT-2 XL (1,5 B)** — 50× maior.
- Modelos de **1 M / 3 M / 9 M** e até de **uma única camada** ainda produzem texto
  coerente e recuperam fatos simples.
- **Profundidade importa mais para consistência de conteúdo do que para gramática**: o
  modelo de 28 M com **8 camadas** bate instâncias de 33 M com 2 e 4 camadas. Gramática sai
  de qualquer arquitetura; coerência de discurso exige profundidade.
- Avaliação: **GPT-4 como corretor**, notas separadas em gramática / criatividade /
  consistência / enredo.

### 5.2 A versão não-inglesa — *Regional Tiny Stories* (arXiv 2504.07989)

Replicação em hindi, bengali e marathi, que é o análogo mais próximo do nosso caso:

- **Configuração ótima: 54 M params = 512 hidden, 6 camadas, 8 heads.** Praticamente o nosso
  modelo. Performance **satura acima de 6 camadas**, apesar do crescimento quadrático de
  params — mais uma confirmação de que a §4 (dados) manda, não a §1 (arquitetura).
- Capacidade básica emerge com **4,5–10 M params**; geração de história completa exige
  **41 M+**.
- **Tokenizer específico da língua importa muito**: hindi tirou 8,158 com o tokenizer Sarvam
  contra 7,602 com tiktoken — ~7% de diferença só de tokenizer. Reforça o achado do Cabrita
  citado em `estado-da-arte.md` §4.1.5, e é argumento direto pro nosso BPE pt-BR próprio
  (e pro experimento de vocab 8k vs 16k da §1.8).

### 5.3 O que isso significa para o Pocket-LM — três conclusões acionáveis

**(a) O diagnóstico está certo e não é consertável com arquitetura.** Nossa amostra é
gramatical e vazia porque o corpus (Wikipedia-pt geral) tem entropia de conteúdo muito maior
do que 33,9 M params podem armazenar com 27 M tokens de exposição. Nenhum RoPE, SwiGLU ou
Muon conserta isso. **Qualidade e estreitamento da distribuição substituem escala** — é a
tese central do TinyStories.

**(b) Existe um "TinyStories em português" pronto, e o `corpus-ptbr.md` já o mapeou.**
`Boakpe/historinhas` (Apache-2.0, ~0,4–0,5 B tokens, 1,25 M histórias sintéticas com
vocabulário simples e narrativa fechada). Duas formas de usar, e as duas valem:

- **Como 15–30% da mistura** de um treino geral — que é o que `corpus-ptbr.md` §4 já
  recomenda, pelo motivo certo (ensinar *fechamento de discurso*).
- **Como run dedicado de validação da receita** (proposta nova desta seção): treinar
  **10–15 M params só em Historinhas**. Com 0,4–0,5 B tokens isso dá **30–50 tok/param**,
  ou seja, um modelo genuinamente *bem treinado* pela primeira vez no projeto. Deve produzir
  português **de fato coerente**, não só gramatical. É o experimento de maior valor
  informativo por minuto de GPU do projeto inteiro: se a receita §1+§2 funciona, aparece
  aqui; se não aparece nem aqui, o problema é a esteira, não a escala.

**(c) A métrica precisa mudar junto.** `estado-da-arte.md` §4.2.5 já registra o aviso do
paper do Tucano: os benchmarks pt disponíveis quase não correlacionam com escala de treino.

> **Correção (27/07/2026):** o aviso do Tucano está **desatualizado** — o PoETa v2 (nov/2025)
> encontra correlação positiva clara entre compute e performance. O problema era o benchmark
> antigo, não a ausência de sinal. E a métrica já foi definida: **BPB congelado + CALAME-PT +
> CALAME-PT-NLL** (`plano-avaliacao.md`), com o juiz-LLM em rubrica por marco, não por
> checkpoint. Ver `PLANO.md` D1/D2.
O TinyStories resolveu isso com **LLM-como-juiz em eixos separados** (gramática, consistência,
enredo). É diretamente aplicável ao `eval.py`: 50 prompts fixos, geração determinística,
correção por um modelo maior em 3–4 eixos. Val loss serve pra comparar duas receitas no mesmo
corpus; não serve pra dizer se o modelo ficou bom.

### 5.4 A convergência que fecha o projeto

Três documentos independentes apontam pro mesmo ponto:

- `estado-da-arte.md` §1.4: **não existe base pt-BR abaixo de ~124 M** — o buraco é `<50 M`.
- §4.6 acima: **para 50 min de T4, o compute-ótimo é ~15–17 M params**.
- §5.2: **em língua não-inglesa, 6 camadas × 512 já satura** no regime de domínio estreito.

Ou seja: o modelo que o orçamento paga, o modelo que a lei de escala recomenda e o modelo
que ninguém publicou em português são **o mesmo modelo**. O caminho de maior retorno não é
um GPT pt-BR de uso geral com 33,9 M params meio-treinado — é um modelo de **10–20 M bem
treinado num domínio estreito**, exatamente o reposicionamento da Trilha B de
`estado-da-arte.md` §4.3.

---

## 6. Lista priorizada de mudanças

Ordenada por **ganho ÷ esforço**. "Δ loss" são estimativas com a forma funcional de §4.3 e
as ablações de §3 — direção e ordem de grandeza, não promessa.

### P0 — bugs e compute desperdiçado (fazer antes de qualquer outra coisa)

| # | Mudança | Onde | Δ esperado | Esforço | Risco |
|---|---|---|---|---|---|
| 1 | **Consertar o horizonte do LR / trocar cosseno por WSD** com piso ~0, disparado pelo tempo restante | `train_gpu.py` `lr_at` | **−0,15 a −0,40 nats** | 20 linhas | **nenhum**. É correção de bug (§0b) |

> **Correção (27/07/2026) ao item 1:** a faixa está certa em magnitude mas **mal atribuída**.
> `analise-runs.md` §4.2 decompõe: rodar os mesmos 320 steps **no platô**, sem decay nenhum,
> já dá **−0,153 nat** pelo ajuste da curva do run B — ou seja, **o piso inteiro da faixa pode
> ser "mais treino", com efeito de schedule zero**. O efeito atribuível ao *schedule* é
> **−0,10 a −0,30**, e a queda total esperada no cooldown é **−0,25 a −0,45**. O experimento
> que separa os dois custa 0,78–1,40 unidade e está desenhado em `analise-runs.md` §4.4 — com
> um pré-requisito bloqueante: **nenhum checkpoint dos runs A/B está na bancada**.

> **Correção (27/07/2026) à lista P0 como um todo:** falta um item, e ele é de 3 linhas —
> **gravar `BLOCK`, `BATCH`, `GRAD_ACCUM`, `tok/step` e tokens acumulados no `report.json`**.
> O `train_gpu.py:116` só emite `device/params/vocab/val_ini/train_tokens`, e foi exatamente
> essa omissão que obrigou a deduzir a arquitetura do run B pela contagem de parâmetros.
> Junto: `train_gpu.py:154` gera a amostra com os defaults `temperature=0.8, top_k=40` de
> `model.py:86` — trocar por `temperature=1.0, top_k=0`. Ver `PLANO.md` §3 item 2 e §5.
| 2 | **SDPA + QKV fundido** no lugar de `nn.MultiheadAttention`, máscara de shape fixo, `bias=False` | `model.py` | **1,5–2,5× tokens** no mesmo tempo (≈ −0,2/−0,3 nats) | ~30 linhas, 1 tarde | baixo. **Pré-requisito do export NPU** — trabalho que já era obrigatório |
| 3 | **Eliminar `GRAD_ACCUM`** (BATCH 48–96, accum 1) e **subir `train.bin` inteiro pra VRAM** | `train_gpu.py` | +10–25% de tokens | 15 linhas | nenhum (54 MB) |
| 4 | **`torch.compile(model)`** (depois do #2) | `train_gpu.py` | +30–80% de tokens | 1 linha + 1 min de compile | baixo; testar fallback |
| 5 | **Grupos de weight decay** (`wd` só em `dim>=2`) e **β₂ 0,98** | `train_gpu.py` | alguns % | 10 linhas | nenhum |
| 6 | **`estimate_loss` determinístico** (batches de val fixos) | `train_gpu.py` | 0 (mas **habilita medir tudo o resto**) | 5 linhas | nenhum |

> P0 sozinho deve dar **~2,5× de tokens no mesmo budget + a fase de decay que nunca
> aconteceu**. É o maior retorno por hora de engenharia do documento, e nada aqui é
> arriscado ou controverso.

### P1 — pacote de arquitetura do speedrun (barato, testado)

| # | Mudança | Δ esperado | Esforço | Risco |
|---|---|---|---|---|
| 7 | **Zero-init das projeções de saída** dos blocos | estabilidade; parte do pacote de −31,8% | 4 linhas | nenhum |
| 8 | **QK-norm** em q,k | destrava LR 2–3× maior; **ajuda a quantização int8** | ~6 linhas | baixo (re-varrer LR) |
| 9 | **ReLU²** no lugar de GELU | ~neutro em loss, mais rápido, **melhor pra NPU** | 2 linhas | nenhum |
| 10 | **Subir o LR** pra 1,5e-3–3e-3 (só depois de 7+8) | vários % | env var | baixo — com QK-norm, spike é improvável |
| 11 | **RoPE** no lugar da embedding de posição | poucos % + `block_size` flexível no deploy | ~25 linhas | baixo. NPU seguro (Qwen3/Gemma usam) |
| 12 | **RMSNorm sem escala aprendida** + `elementwise_affine=False` | ~5% de velocidade | 10 linhas | baixo, **trivial de reverter** se o export reclamar |

### P2 — dados e dimensionamento (o maior ganho absoluto)

| # | Mudança | Δ esperado | Esforço | Risco |
|---|---|---|---|---|
| 13 | **Corpus 27 M → ≥ 700 M tokens** com a Mistura C de `corpus-ptbr.md` (70% gigaverbo-v2 edu≥3 + 30% historinhas) | **−0,5 a −0,8 nats** — o maior item do documento | preparo de dados no x86, **zero GPU** | tempo de CPU/rede; nada técnico |

> **Correção (27/07/2026) ao item 13:** o ponto de partida não é mais 27 M — o corpus foi
> re-tokenizado e está em **202,6 M tokens únicos**. Mas **o item não está feito**: o que
> cresceu foi o volume de **Wikipedia-pt**, e a Mistura C existe para trazer **diversidade de
> registro** (web educacional + historinhas), que é o que a §5 identifica como a causa de
> "gramatical e vazio". Alvo revisado: **≥400 M únicos com a mistura**, não ≥700 M de
> qualquer coisa. Risco novo: **6,1 GB de disco livre** — preparar em streaming, escrever o
> `.bin` direto, não materializar o texto todo. Ver `PLANO.md` §3 item 3.
| 14 | **Encolher para ~17 M params** (6L × 6H × 384d) para bater o ponto compute-ótimo de §4.2 | −0,13 nats **e** entra no nicho `<50 M` do português | env vars | nenhum. Reversível |
| 15 | **Cooldown com dados de alta qualidade** (synth + historinhas só nos últimos 20%) | ganho de coerência percebida > ganho de loss | baixo, se #1 já for WSD | nenhum |
| 16 | **Run "pt-BR TinyStories"**: 10–15 M params só em `Boakpe/historinhas` (§5.3b) | **o teste decisivo da receita**; primeiro modelo do projeto de fato bem treinado | 1 rodada de 50 min | nenhum. Alto valor informativo |

### P3 — ciência de verdade (maior payoff, maior risco)

| # | Mudança | Δ esperado | Esforço | Risco |
|---|---|---|---|---|
| 17 | **Muon** nas matrizes 2D (AdamW no resto) | **1,4× de eficiência de compute** em 130 M params (arXiv 2509.02046); −20,7% no speedrun | ~60 linhas + tuning de LR próprio | **médio** — separação de grupos errada degrada em silêncio |
| 18 | **Trocar T4 por L4** se o orçamento for em R$ e não em horas | ~2× de compute, e bf16 sem GradScaler | mudar o job | nenhum, fora custo |
| 19 | **Experimento vocab 8 k vs 16 k** | desconhecido, plausivelmente positivo (§1.8, §5.2) | 1 rodada de 25 min | nenhum |

> **Correção (27/07/2026) ao item 19:** deixou de ser "desconhecido" e subiu de prioridade.
> `vocab-e-destilacao.md` §A.2 rodou os três preditores do paper de vocabulário com as
> constantes do repo oficial e mediu `f(V)` no **nosso** corpus: o ótimo do Pocket-LM está
> entre **2.500 e 6.500** — o alvo é **4.096**, com 8.192 como opção conservadora, e 16.384 só
> se justificaria a partir de ~50 B tokens de treino. Dois detalhes que economizam trabalho:
> **não é preciso re-treinar o BPE** (truncar `merges` do `tokenizer.json` dá o tokenizer
> exato), e os FLOPs liberados têm de virar **mais tokens**, nunca mais camadas — com o total
> de params travado o ganho previsto cai de −0,26 para −0,005. Pré-condição: **BPB congelado
> antes**, senão a comparação é impossível.
| 20 | **`eval.py` com LLM-como-juiz** em 4 eixos (§5.3c) | 0 em loss; **habilita saber se ficou bom** | meia tarde | nenhum |

### P4 — só depois de tudo acima

SwiGLU (§1.6) · multi-token prediction (§3, −2,4%, e grátis no export) · peri-norm (§1.7) ·
logit softcap (§1.7) · Soap/Kron no lugar do Muon **se** a razão dados/params passar de 8×
(§2.1) · desamarrar as embeddings só no último terço do treino (§1.8).

### Não fazer

| Item | Motivo |
|---|---|
| **GQA** | −5% de ppl em modelo pequeno e economiza um KV-cache de 4,2 MB que não incomoda ninguém (§1.9) |
| **Untied embeddings** (o tempo todo) | +25% de params e +4,2 MB no artefato int4, contra a meta de tamanho (§1.8) |
| **FlexAttention / janela deslizante / ctx 64 k** | ganho vem de contexto longo; nosso block é 512 e a NPU exige shape estático (§3) |
| **FP8** | exige Hopper (§3) |
| **U-net skips / value embeddings / MUDD** | ops não-padrão → fallback pra CPU no export, ganho/linha baixo (§3) |
| **Post-norm** | instável nesse tamanho sem truque de init (§1.7) |
| **Mais épocas do mesmo corpus** | já estamos em 5,12, acima do limite de ~4 onde repetir sai de graça (§4.4) |

### Sequência sugerida (3 rodadas)

1. **Rodada A** — P0 completo, mesmo modelo, mesmo corpus. Serve de controle: mede
   isoladamente quanto valem o bug de schedule e a MFU. Esperado: val ~3,2–3,4.
2. **Rodada B** — P1 completo + LR novo, ainda no mesmo corpus. Mede o pacote do speedrun
   sem confundir com mudança de dados.
3. **Rodada C** — P2: corpus novo (≥700 M tokens), modelo de ~17 M, WSD com cooldown de
   qualidade. É a primeira que deve gerar português com conteúdo, não só com gramática.
   Em paralelo, o run #16 (pt-BR TinyStories) como validação barata.

Muon (#17) entra na rodada D, sozinho, para ser medido contra a C.

---

## Fontes

**Speedrun e arquitetura**
- modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt ·
  ablações: https://deepwiki.com/KellerJordan/modded-nanogpt/10-experimental-results ·
  discussão: https://github.com/KellerJordan/modded-nanogpt/discussions/23 ·
  retrospectiva: https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months
- The Automated LLM Speedrunning Benchmark: https://arxiv.org/pdf/2506.22419
- The Big LLM Architecture Comparison (Raschka): https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison
- Modernizing GPT-2 (RoPE + RMSNorm + SwiGLU): https://recsysml.substack.com/p/modernizing-gpt-2-a-journey-from
- Small Language Models Can Still Pack a Punch (survey): https://arxiv.org/pdf/2501.05465
- GQA: https://ar5iv.labs.arxiv.org/html/2305.13245 · https://sebastianraschka.com/llm-architecture-gallery/gqa/
- Qwen3 Technical Report (QK-norm, RoPE, tying): https://arxiv.org/html/2505.09388v1

**Otimizador e schedule**
- Fantastic Pretraining Optimizers and Where to Find Them: https://arxiv.org/html/2509.02046v2
- Practical Efficiency of Muon for Pretraining: https://arxiv.org/html/2505.02222v1
- Muon, muP e o trade-off compute/tempo (Essential AI): https://www.essential.ai/research/optimizer
- AdaMuon: https://arxiv.org/pdf/2507.11005
- Predictable Scale / Step Law (η e B ótimos): https://arxiv.org/html/2503.04715v1
- Small Batch Size Training for Language Models: https://arxiv.org/pdf/2507.07101
- WSD — river valley loss landscape: https://arxiv.org/pdf/2410.05192
- Training dynamics do cooldown em WSD: https://arxiv.org/pdf/2508.01483
- Optimal LR schedules under functional scaling laws: https://arxiv.org/abs/2602.06797
- Power Scheduler (agnóstico a batch e tokens): https://arxiv.org/pdf/2408.13359

**Leis de escala**
- Chinchilla (Hoffmann et al.): https://arxiv.org/abs/2203.15556
- Beyond Chinchilla-Optimal (inferência): https://arxiv.org/abs/2401.00448
- Scaling Data-Constrained Language Models (épocas repetidas): https://arxiv.org/abs/2305.16264 ·
  versão JMLR: https://jmlr.org/papers/volume26/24-1000/24-1000.pdf
- Resolving Discrepancies in Compute-Optimal Scaling: https://arxiv.org/pdf/2406.19146
- MiniCPM (WSD e razão 192×): https://arxiv.org/abs/2404.06395
- SmolLM2 (data-centric, 135 M em 2 T tokens): https://arxiv.org/html/2502.02737v1

**Modelos minúsculos coerentes**
- TinyStories: https://arxiv.org/pdf/2305.07759 · https://openreview.net/forum?id=yiPtWSrBrN
- Regional Tiny Stories (não-inglês, 54 M / 6 camadas / tokenizer): https://arxiv.org/html/2504.07989v1
- TinyStories-33M (pesos): https://huggingface.co/roneneldan/TinyStories-33M
- Historinhas (o TinyStories pt-BR): https://huggingface.co/datasets/Boakpe/historinhas
