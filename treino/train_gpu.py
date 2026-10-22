"""Treino na GPU (Lightning L4/A10G/etc). Modelo maior que o char-level da CPU, tokens BPE,
AMP bfloat16, LR cosseno com warmup e acumulacao de gradiente. Retomavel a partir de ckpt.pt.
Escreve tudo em OUT_DIR (default ./out) pra virar artifact do job e ser puxado pro Hetzner."""
from __future__ import annotations
import contextlib
import json
import math
import os
import time
import numpy as np
import torch

from model import GPT, GPTConfig

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(DIR, "data"))
OUT = os.environ.get("OUT_DIR", os.path.join(DIR, "out"))
CKPT = os.path.join(OUT, "ckpt.pt")
REPORT = os.path.join(OUT, "report.json")

# hiperparametros (env-configuraveis) — defaults ~50M params num L4 de 23GB
BLOCK = int(os.environ.get("BLOCK", "512"))
N_LAYER = int(os.environ.get("N_LAYER", "8"))
N_HEAD = int(os.environ.get("N_HEAD", "8"))
N_EMBD = int(os.environ.get("N_EMBD", "512"))
BATCH = int(os.environ.get("BATCH", "24"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
LR = float(os.environ.get("LR", "6e-4"))
WARMUP = int(os.environ.get("WARMUP", "200"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "100000"))
DECAY_FRAC = float(os.environ.get("DECAY_FRAC", "0.2"))   # fracao final do orcamento em decaimento
LR_FLOOR = float(os.environ.get("LR_FLOOR", "0.0"))       # LR no fim do decaimento
# fase de estilo (fine-tuning): retoma um checkpoint mas com orcamento NOVO. Sem isso o
# gasto_s herdado (2400s de um treino anterior) faz frac_tempo passar de 1 no primeiro
# step e o LR nasce no piso — o treino nao sai do lugar.
RESET_SCHEDULE = os.environ.get("RESET_SCHEDULE", "") == "1"
DECAY_FORMA = os.environ.get("DECAY_FORMA", "1-sqrt")     # 1-sqrt | linear
OTIM = os.environ.get("OTIM", "adamw")                     # adamw | muon
# multiplicadores do Muon por grupo, JSON. Os defaults vieram de outro projeto e no A/B
# de 28/07 perderam para o AdamW (3.985 contra 3.644): o modelo dispara na frente ate o
# passo 600 e depois empaca, assinatura de passo grande demais. Existe para varrer.
MUON_MULT = json.loads(os.environ.get("MUON_MULT", "{}") or "{}")
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0"))        # 0 desliga a media de pesos
EMA_A_CADA = int(os.environ.get("EMA_A_CADA", "10"))
# horizonte da media em PASSOS. 0 = usa EMA_DECAY direto. Regra pratica: ~10% dos passos
# do run — horizonte maior que o proprio run faz a media nunca alcancar o modelo.
EMA_HORIZONTE = int(os.environ.get("EMA_HORIZONTE", "0"))
# Replay: fatia de cada batch vinda do corpus ANTIGO, contra esquecimento ao continuar
# treino em dado novo. Medido em 405M e 10B (arxiv 2403.08763): 5% recupera ~80% do
# esquecimento para mudanca leve de dominio; 25% para mudanca forte (outra lingua).
REPLAY_BIN = os.environ.get("REPLAY_BIN", "")
REPLAY_FRAC = float(os.environ.get("REPLAY_FRAC", "0.05")) if REPLAY_BIN else 0.0
# fracao dos micro-batches que vem do corpus PREMIUM durante a fase de decay do WSD.
# 0.5 espelha a mistura do mid-training do OLMo 2 (~50% dado curado).
PREMIUM_FRAC = float(os.environ.get("PREMIUM_FRAC", "0.5"))
BUDGET_S = int(os.environ.get("TRAIN_BUDGET_S", str(3 * 3600)))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "250"))
EVAL_ITERS = int(os.environ.get("EVAL_ITERS", "50"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "500"))
# a cada N steps o ckpt.pt sobe para o R2 (0 desliga). Upload de 1,1 GB custa ~13 s a
# cada ~27 min de treino (<1% de overhead) e limita a perda por queda de maquina a
# meia hora — sem isto, so o push_out do FIM persiste e uma queda na hora 7 perde tudo.
CKPT_R2_EVERY = int(os.environ.get("CKPT_R2_EVERY", "0"))
# 1 = cronograma por PASSO (usar em ablacao, ver comentario no laco principal);
# 0 = por tempo (default, usado em treino normal)
PASSO_FIXO = os.environ.get("SCHEDULE_POR_PASSO", "") == "1"

# --- treino distribuido (DDP) -------------------------------------------------------
# Ativado quando lancado por torchrun, que define RANK/WORLD_SIZE. Sem torchrun, tudo
# abaixo vira no-op e o script roda como sempre — nao ha caminho novo em execucao unica.
#
# Corretude verificada em `teste_ddp.py` ANTES de gastar GPU: 2 processos com meio lote
# cada reproduzem a perda de 1 processo com o lote inteiro, com diferenca de 8,59e-08 (erro
# de ponto flutuante). O modo de falha que isso descarta e o pior: gradiente nao
# sincronizado faz o treino RODAR e a perda CAIR, e so no fim se descobre que N GPUs
# aprenderam como uma.
DDP_ON = int(os.environ.get("WORLD_SIZE", "1")) > 1
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
MUNDO = int(os.environ.get("WORLD_SIZE", "1"))
E_MESTRE = RANK == 0

if DDP_ON:
    import torch.distributed as dist
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)
    dev = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"
else:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = dev.startswith("cuda")
# bf16 so em Ampere+ (A100/L40S/H100); T4/Turing so tem fp16 (que exige GradScaler)
AMP_DTYPE = (torch.bfloat16 if (dev.startswith("cuda") and torch.cuda.is_bf16_supported())
             else torch.float16)
torch.manual_seed(1337)


def load_bin(name):
    return np.fromfile(os.path.join(DATA, name), dtype=np.uint16)


def carrega_corpus(prefixo: str):
    """Carrega um corpus que pode ser um arquivo unico OU uma lista de shards.

    O corpus do ClassiCC sai em pedacos `main-sNNNNN-pNN.bin` (1 GB cada) porque a
    maquina que o gera nao tem disco para materializar 24 GB. Aqui os pedacos sao
    concatenados em memoria via memmap->concatenate. A ORDEM e ordenada por nome e
    depois os BLOCOS sao permutados pelo get_batch — nao ha currículo acidental porque
    a varredura permutada ja embaralha os blocos de 512 tokens, independente de onde
    cada shard comecou.

    `prefixo` pode ser: "train.bin" (arquivo unico, comportamento antigo) ou um glob
    tipo "main-*.bin" / "premium-*.bin".
    """
    import glob as _glob
    caminho = os.path.join(DATA, prefixo)
    if os.path.exists(caminho):
        return np.fromfile(caminho, dtype=np.uint16)
    arquivos = sorted(_glob.glob(caminho))
    if not arquivos:
        raise SystemExit(f"nenhum arquivo casa com {caminho}")
    if len(arquivos) == 1:
        return np.memmap(arquivos[0], dtype=np.uint16, mode="r")
    # varios shards: concatena UMA vez em disco e le por memmap. O np.concatenate
    # anterior materializava tudo em RAM — 24,6 GB para o corpus limpo, com pico de
    # ~49 GB durante a copia. O memmap deixa o cache de pagina do SO decidir o que
    # fica residente; o get_batch le blocos aleatorios e nao precisa do resto.
    unico = caminho.replace("*", "TUDO").replace("?", "_") + ".cat"
    esperado = sum(os.path.getsize(f) for f in arquivos)
    if not (os.path.exists(unico) and os.path.getsize(unico) == esperado):
        # Sob DDP, SO o rank 0 constroi; os demais esperam o arquivo completo aparecer.
        # Na primeira rodada os 4 ranks construiram o MESMO .tmp ao mesmo tempo: o
        # primeiro os.replace levou o arquivo e os outros tres morreram com
        # FileNotFoundError (04/08, derrubou o lancamento inteiro). O teste da bancada
        # nao pegou porque rodou em processo unico.
        if RANK == 0:
            tmp = unico + f".tmp{os.getpid()}"
            with open(tmp, "wb") as fo:
                for f in arquivos:
                    with open(f, "rb") as fi:
                        while (blk := fi.read(64 << 20)):
                            fo.write(blk)
            os.replace(tmp, unico)   # atomico: nunca fica .cat pela metade com nome final
        else:
            t_espera = time.time()
            while not (os.path.exists(unico) and os.path.getsize(unico) == esperado):
                if time.time() - t_espera > 3600:
                    raise SystemExit(f"rank {RANK}: esperei 1h pelo {unico} e nada")
                time.sleep(5)
    arr = np.memmap(unico, dtype=np.uint16, mode="r")
    print(json.dumps({"corpus": prefixo, "shards": len(arquivos),
                      "tokens": int(arr.size), "cat": os.path.basename(unico)}),
          flush=True)
    return arr


# Pico DENSO em bf16, por GPU. O datasheet da NVIDIA anuncia 1979 TFLOPs para a H100 SXM
# com o rodape "*With sparsity" — o denso e METADE. Dividir pelo numero de marketing
# reporta metade do MFU real e manda otimizar o alvo errado: foi o que aconteceu aqui,
# com um "16%" anotado que quase certamente era 32%.
# Referencia de alvo realista: modelo pequeno rende MENOS MFU em H100 que em A100, porque
# o pico de calculo subiu 3,2x e a banda de memoria so 1,7x. Medicao publicada da config
# mais parecida com a nossa (760M, seq 512, H100) da 40,4%. O teto aqui e ~40%, nao 56%.
PICO_BF16 = {"H100": 989e12, "H100_NVL": 835e12, "H100_PCIE": 756e12,
             "H200": 989e12, "A100": 312e12, "L40S": 362e12, "L4": 121e12}
GPU_PICO = float(os.environ.get("GPU_PICO_FLOPS", "0")) or \
    PICO_BF16.get(os.environ.get("GPU_TIPO", "H100").upper(), 989e12)


def flops_por_token(model, cfg) -> int:
    """FLOPs de treino por token — formula do PaLM (arxiv 2204.02311, Apendice B):

        6N + 12 * n_layer * n_head * head_dim * block_size

    Duas sutilezas que mudam o numero e que a maioria das implementacoes erra:

    - **`pos.weight` sai da conta** (embedding de posicao e lookup, nao faz matmul), mas
      **`tok.weight` FICA**: com embeddings amarrados esse mesmo tensor e a matriz do
      classificador final e participa da GEMM de saida. `parameters()` ja o conta uma vez.
    - **O termo de atencao usa 12, nao 6.** O llm.c usa 6 porque cita a Tabela 1 do
      Kaplan, que conta so um dos dois matmuls; com 6 o MFU sai subestimado.

    No nosso 269M com block 512 a atencao e ~7% do total. Alongar o contexto aumenta
    justamente ESSA fatia (13,5% em 1024, 23,8% em 2048), que e a parte cara — mais um
    motivo para nao mexer no block_size por razao de desempenho.
    """
    n = sum(p.numel() for p in model.parameters())
    pos = getattr(model, "pos", None)
    if pos is not None and hasattr(pos, "weight"):
        n -= pos.weight.numel()
    q = cfg.n_embd // cfg.n_head
    return 6 * n + 12 * cfg.n_layer * cfg.n_head * q * cfg.block_size


_sweep = {"epoca": -1, "ordem": None, "pos": 0, "fase": 0}
# DOIS geradores independentes, um por decisao. Com UM so compartilhado, o numero de
# sorteios consumidos dependia do RESULTADO do primeiro (so sorteia replay se nao caiu em
# premium) — entao mudar PREMIUM_FRAC deslocava a sequencia inteira e, com ela, a ordem de
# varredura do corpus. Numa ablacao de PREMIUM_FRAC isso faria os bracos diferirem em DUAS
# variaveis (fracao E ordem dos dados), que e o erro classico que invalida o experimento.
_rng_premium = np.random.default_rng(20260728)
_rng_replay = np.random.default_rng(20260729)


def ordem_da_epoca(n_tokens, epoca, rank=0, mundo=1):
    """Blocos que ESTE processo le na epoca, e a fase da epoca.

    Todos os processos geram a mesma permutacao (mesma semente) e cada um fica com a sua
    fatia `[rank::mundo]`: juntos cobrem a epoca sem sobreposicao. Sem a fatia, os N
    processos liam o mesmo micro-lote — o treino roda e a perda cai, mas so 1/N das fichas
    processadas sao distintas.

    A permutacao e cortada num multiplo de `mundo` antes de fatiar, para todas as fatias
    terem o mesmo tamanho. Com fatias desiguais um processo viraria a epoca antes dos
    outros e passaria a ler a permutacao seguinte enquanto os demais ainda leem a atual,
    e as fatias voltariam a se sobrepor. O corte deixa de fora menos de `mundo` blocos
    por epoca, sorteados de novo a cada epoca. Com mundo=1 nada muda.
    """
    rng = np.random.default_rng(1337 + epoca)
    fase = int(rng.integers(0, BLOCK))
    # o numero de blocos ja desconta a fase: assim blocos*BLOCK + fase nunca
    # passa do fim e nao precisa de modulo (que fazia blocos do fim colidirem
    # com blocos do inicio, desfazendo a cobertura exata)
    n_blocos = max(1, (n_tokens - BLOCK - 1 - fase) // BLOCK + 1)
    ordem = rng.permutation(n_blocos)
    if mundo > 1:
        ordem = ordem[:n_blocos - n_blocos % mundo][rank::mundo]
        if len(ordem) == 0:
            raise SystemExit(f"corpus pequeno demais para {mundo} processos: "
                             f"{n_blocos} blocos de {BLOCK}")
    return ordem, fase


def tempo_comum(t0):
    """Segundos desde t0 segundo o rank 0, o mesmo valor em todos os processos.

    Sob DDP, o fim do orcamento e a entrada no decay (que liga o corpus premium) dependem
    do relogio. Com o relogio de cada processo, um rank podia entrar no decay ou sair do
    laco um passo antes dos outros: no primeiro caso as epocas da varredura dessincronizam
    e as fatias voltam a se sobrepor; no segundo, os demais travam esperando o all-reduce.
    """
    decorrido = time.time() - t0
    if not DDP_ON:
        return decorrido
    t = torch.tensor([decorrido], dtype=torch.float64, device=dev)
    dist.broadcast(t, 0)
    return float(t.item())


def get_batch(data, permutado=True):
    """Varredura permutada com fase deslocada por epoca.

    O `np.random.randint` original amostra offsets com reposicao: numa passada do
    tamanho de uma epoca ~0.6% dos tokens nunca sao vistos e outros aparecem 10-12x.
    Isso vira variancia entre seeds, que e justamente o que atrapalha comparar duas
    receitas em rodadas curtas de 25-50 min.

    Aqui cada epoca e uma permutacao dos blocos (cobertura exata, cada token uma vez)
    e a fase desloca o inicio, entao as fronteiras de bloco nao caem sempre nos mesmos
    tokens — senao o modelo nunca veria certos pares atravessando a fronteira.
    """
    if permutado:
        if _sweep["ordem"] is None or _sweep["pos"] + BATCH > len(_sweep["ordem"]):
            _sweep["epoca"] += 1
            ordem, fase = ordem_da_epoca(len(data), _sweep["epoca"], RANK, MUNDO)
            _sweep.update({"ordem": ordem, "pos": 0, "fase": fase})
        blocos = _sweep["ordem"][_sweep["pos"]:_sweep["pos"] + BATCH]
        _sweep["pos"] += BATCH
        ix = blocos * BLOCK + _sweep["fase"]
    else:
        ix = np.random.randint(0, len(data) - BLOCK - 1, size=BATCH)
    x = np.stack([data[i:i + BLOCK] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + BLOCK] for i in ix]).astype(np.int64)
    xt = torch.from_numpy(x).to(dev, non_blocking=True)
    yt = torch.from_numpy(y).to(dev, non_blocking=True)
    return xt, yt


def lr_at(step, frac_tempo=None):
    """Warmup-Stable-Decay disparado pelo TEMPO, nao pelo numero de steps.

    O cosseno anterior era calculado sobre MAX_STEPS=100000, mas o treino para por
    TRAIN_BUDGET_S. Medido no run pocket-h200-v2: no ultimo step o LR ainda estava em
    99.8% do pico — o run inteiro aconteceu no plato e nunca viu a fase de decaimento,
    que e onde a maior parte da queda de loss acontece.

    Aqui `frac_tempo` = tempo decorrido / orcamento. O decaimento comeca quando resta
    DECAY_FRAC do orcamento e vai ate LR_FLOOR, entao a fase acontece SEMPRE, qualquer
    que seja a maquina ou quantos steps couberem na janela.
    """
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    if frac_tempo is None:                      # sem relogio: cosseno antigo
        prog = min(1.0, (step - WARMUP) / max(1, MAX_STEPS - WARMUP))
        return 0.1 * LR + 0.5 * (LR - 0.1 * LR) * (1 + math.cos(math.pi * prog))
    ini_decay = 1.0 - DECAY_FRAC
    if frac_tempo < ini_decay:
        return LR                               # fase estavel, no pico
    p = min(1.0, (frac_tempo - ini_decay) / max(1e-9, DECAY_FRAC))
    # forma 1-sqrt: cai rapido no comeco do decay e passa mais tempo em LR baixo.
    # Hagele et al. (arXiv 2405.18392) mede 1-sqrt >= linear >= nada; o Tucano 2 usa
    # a mesma forma. DECAY_FORMA=linear mantem o comportamento dos runs antigos.
    if DECAY_FORMA == "1-sqrt":
        fator = 1.0 - math.sqrt(p)
    else:
        fator = 1.0 - p
    return LR * fator + LR_FLOOR * (1.0 - fator)


@torch.no_grad()
def estimate_loss(model, data):
    model.eval()
    tot = 0.0
    for _ in range(EVAL_ITERS):
        x, y = get_batch(data, permutado=False)   # nao tocar no _sweep do treino
        with torch.autocast(device_type=dev.split(":")[0], dtype=AMP_DTYPE, enabled=use_amp):
            _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / EVAL_ITERS


_pub = {"ultimo": 0.0, "erros": 0}


def publica_progresso(payload: dict, min_intervalo: float = 60.0) -> None:
    """Sobe o progresso pro Drive pra dar pra acompanhar a curva DE FORA enquanto roda.

    O SDK do Lightning nao entrega log de job em execucao, entao um run de 40 min passa
    as cegas — foi assim que o pocket-h200-v2 gastou 3.14 unidades antes de alguem ver
    que o LR estava travado no pico desde o step 200.

    Nunca derruba o treino: falha de rede aqui e irrelevante perto de perder o run. Para
    de tentar depois de 5 erros seguidos.
    """
    if _pub["erros"] >= 5:
        return
    agora = time.time()
    if agora - _pub["ultimo"] < min_intervalo:
        return
    _pub["ultimo"] = agora
    try:
        os.makedirs(OUT, exist_ok=True)
        local = os.path.join(OUT, "progress.json")
        tmp = local + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, local)
        # So R2: a leitura do Drive da Lightning esta quebrada desde 27/07 (o run
        # pocket-269m-v4 ficou preso la) e em 29/07 a ESCRITA travou a funcao inteira
        # por 90+ min sem lancar excecao (achado com ckpt.pt avancando normal no R2
        # enquanto progress.json ficava parado - ver docs/historia-e-licoes.md). O
        # Drive nunca foi necessario: o R2 e o unico caminho que provou funcionar.
        # connect/read timeout explicito: sem isso uma trava de rede aqui fica pendurada
        # pra sempre em vez de cair no except e ser contada/logada.
        import boto3
        from botocore.config import Config
        _s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
            config=Config(signature_version="s3v4", retries={"max_attempts": 2},
                          connect_timeout=10, read_timeout=20))
        _s3.upload_file(local, os.environ.get("POCKET_LM_BUCKET", "pocket-lm"),
                        f"live/{os.environ.get('RUN_NAME', 'sem-nome')}/progress.json")
        _pub["erros"] = 0
    except Exception as e:  # noqa: BLE001
        _pub["erros"] += 1
        print(json.dumps({"aviso": "publica_progresso falhou",
                          "erro": type(e).__name__, "seguidos": _pub["erros"]}), flush=True)


def escolhe_fonte(train, replay, premium=None, em_decay=False):
    """Sorteia de qual corpus vem este micro-batch.

    O sorteio e por micro-batch, nao por step: com GRAD_ACCUM micro-batches, uma fracao
    de 5% vira ~5% dos micro-batches, e o gradiente de cada step ja mistura os dois
    corpora. Sortear por step daria steps inteiramente antigos de vez em quando, que e
    mais ruidoso para a mesma fracao de dados.

    A varredura permutada so vale para o corpus novo: o replay e amostragem com
    reposicao de propria natureza — ele nao existe para cobrir o corpus antigo, so para
    lembrar o modelo de que ele existe.
    """
    # SEMPRE sorteia os dois, mesmo que o primeiro ja decida: assim o consumo de
    # aleatoriedade e constante e independente das fracoes, e dois runs com fracoes
    # diferentes veem a MESMA ordem de dados.
    sorte_premium = _rng_premium.random()
    sorte_replay = _rng_replay.random()
    if em_decay and premium is not None and sorte_premium < PREMIUM_FRAC:
        # fase final: dado curado entra com LR ja caindo — e onde o OLMo 2 1B mediu
        # +11,8 pts de media. Amostragem com reposicao: o pool premium e maior que o
        # que o decay consome, entao repeticao e rara.
        return get_batch(premium, permutado=False)
    if replay is None or sorte_replay >= REPLAY_FRAC:
        return get_batch(train)
    return get_batch(replay, permutado=False)


def save_ckpt(model, opt, scaler, step, cfg, hist, gasto_s=0.0, ema=None):
    # so o rank 0 grava — N processos escrevendo o mesmo caminho corrompem o arquivo
    if not E_MESTRE:
        return
    """gasto_s: segundos de treino ja consumidos, somando runs anteriores.

    Sem isso a retomada reinicia o relogio do WSD e o LR volta ao pico depois de ja
    ter decaido — o decay feito no run anterior seria desfeito. Pelo mesmo motivo a
    epoca da varredura permutada tambem viaja no checkpoint: senao a retomada repete
    exatamente a mesma ordem de dados do inicio.
    """
    os.makedirs(OUT, exist_ok=True)
    tmp = CKPT + ".tmp"
    opts = opt if isinstance(opt, (list, tuple)) else [opt]
    d = {"model": model.state_dict(), "opt": opts[0].state_dict(),
         "opts": [o.state_dict() for o in opts], "otim": OTIM,
         "scaler": scaler.state_dict(), "step": step,
         "cfg": vars(cfg), "hist": hist[-500:],
         "gasto_s": float(gasto_s), "sweep_epoca": _sweep["epoca"]}
    if ema is not None:
        # a media entra como state_dict COMPLETO e alternativo: quem for exportar
        # escolhe qual dos dois usar sem precisar saber que houve EMA no treino
        d["ema"] = ema.estado()
        d["model_ema"] = {k: v.clone() for k, v in ema.sombra.items()}
    torch.save(d, tmp)
    os.replace(tmp, CKPT)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))
    vocab_size = meta["vocab_size"]
    train = carrega_corpus(os.environ.get("TRAIN_BIN", "train.bin"))
    val = carrega_corpus(os.environ.get("VAL_BIN", "val.bin"))
    # fase final com dado premium: quando PREMIUM_BIN existe e o treino entra na fracao
    # de decay do WSD, o get_batch passa a sortear do corpus premium com PREMIUM_FRAC de
    # probabilidade. E a alavanca de maior ganho medido no nosso porte (OLMo 2 1B:
    # +11,8 pts de media gastando 1,25% do orcamento em dado curado no fim).
    premium = None
    if os.environ.get("PREMIUM_BIN"):
        premium = carrega_corpus(os.environ["PREMIUM_BIN"])
    if len(train) <= BLOCK + 1 or len(val) <= BLOCK + 1:
        raise SystemExit(f"dados pequenos demais pro block {BLOCK}: "
                         f"train={len(train)} val={len(val)} tokens")

    cfg = GPTConfig(vocab_size=vocab_size, block_size=BLOCK,
                    n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD)
    model = GPT(cfg).to(dev)
    from otim import MediaPesos, monta_otimizador
    # o otimizador, a EMA e o checkpoint enxergam o modelo CRU. So o forward/backward passa
    # pelo wrapper — DDP so existe para sincronizar gradiente.
    modelo_cru = model
    opts, desc_otim = monta_otimizador(modelo_cru, LR, OTIM, mult=MUON_MULT or None)
    if DDP_ON:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(modelo_cru,
                    device_ids=[LOCAL_RANK] if torch.cuda.is_available() else None)
    opt = opts[0]
    ema = (MediaPesos(modelo_cru, EMA_DECAY, EMA_A_CADA, EMA_HORIZONTE)
           if EMA_DECAY > 0 or EMA_HORIZONTE > 0 else None)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and AMP_DTYPE == torch.float16))

    replay = None
    if REPLAY_BIN:
        replay = np.fromfile(REPLAY_BIN, dtype=np.uint16)
        if len(replay) <= BLOCK + 1:
            raise SystemExit(f"REPLAY_BIN pequeno demais: {len(replay)} tokens")

    step0, hist, gasto_anterior = 0, [], 0.0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=dev)
        if ck.get("cfg", {}).get("vocab_size") == vocab_size:
            modelo_cru.load_state_dict(ck["model"])
            # o estado do otimizador so volta se for o MESMO otimizador: carregar
            # estado de AdamW num Muon (ou vice-versa) explodiria em chave faltando, e
            # trocar de otimizador no meio do treino e justamente um caso previsto
            if ck.get("otim", "adamw") == OTIM and ck.get("opts"):
                for o, est in zip(opts, ck["opts"]):
                    o.load_state_dict(est)
            elif ck.get("otim", "adamw") == OTIM and OTIM == "adamw":
                opt.load_state_dict(ck["opt"])
            else:
                print(json.dumps({"aviso": "otimizador mudou, estado descartado",
                                  "antes": ck.get("otim", "adamw"), "agora": OTIM}),
                      flush=True)
            if ema is not None and ck.get("ema"):
                ema.carrega(ck["ema"])
            if ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
            step0 = 0 if RESET_SCHEDULE else ck.get("step", 0)
            hist = ck.get("hist", [])
            gasto_anterior = 0.0 if RESET_SCHEDULE else float(ck.get("gasto_s", 0.0))
            _sweep["epoca"] = int(ck.get("sweep_epoca", -1))

    model.train()
    step = step0
    last = None
    fpt = flops_por_token(model, cfg)
    tokens_step = BATCH * GRAD_ACCUM * BLOCK
    loss_acum = torch.zeros((), device=dev)
    n_acum = 0
    t_janela = time.time()
    steps_janela = 0
    val_ini = estimate_loss(model, val)
    print(json.dumps({"device": dev, "params": modelo_cru.num_params(),
                      "vocab": vocab_size, "val_ini": round(val_ini, 4),
                      "train_tokens": int(len(train)),
                      "n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD,
                      "block": BLOCK, "batch": BATCH, "grad_accum": GRAD_ACCUM,
                      "tokens_por_step": BATCH * GRAD_ACCUM * BLOCK,
                      # tokens_por_step e por processo; somando os processos:
                      "mundo": MUNDO,
                      "tokens_por_step_todos": BATCH * GRAD_ACCUM * BLOCK * MUNDO,
                      "lr_pico": LR, "warmup": WARMUP, "decay_frac": DECAY_FRAC,
                      "budget_s": BUDGET_S, "otim": desc_otim,
                      "flops_por_token": fpt, "gpu_pico_tflops": GPU_PICO / 1e12,
                      "ema": {"decaimento": round(ema.decaimento, 6),
                              "a_cada": ema.a_cada,
                              "horizonte_passos": ema.horizonte} if ema else None,
                      "replay": {"bin": os.path.basename(REPLAY_BIN),
                                 "frac": REPLAY_FRAC,
                                 "tokens": int(len(replay))} if replay is not None
                                else None}), flush=True)

    pre_decay_salvo = os.path.exists(os.path.join(OUT, "ckpt_pre_decay.pt"))
    while step < MAX_STEPS:
        decorrido = tempo_comum(t0)
        if decorrido >= BUDGET_S:
            break
        # o orcamento e do TREINO inteiro, nao deste processo: sem somar o gasto
        # anterior, uma retomada devolveria o LR ao pico
        # Por padrao o cronograma e por TEMPO (ver docstring de lr_at): garante que a
        # fase de decaimento aconteca qualquer que seja a maquina. Mas para ABLACAO isso
        # e defeito: bracos com throughput diferente veriam quantidades diferentes de
        # token, e a comparacao mistura duas variaveis. Com SCHEDULE_POR_PASSO=1 o
        # cronograma passa a ser por PASSO, entao todos os bracos veem exatamente os
        # mesmos tokens e diferem so no que se quer medir.
        if PASSO_FIXO:
            frac_tempo = min(1.0, step / max(1, MAX_STEPS))
        else:
            frac_tempo = min(1.0, (gasto_anterior + decorrido) / max(1.0, BUDGET_S))
        # checkpoint da fase estavel, SEPARADO do ckpt.pt rotativo: e o unico ponto
        # de retomada canonico do WSD (MiniCPM/Ibrahim) — estender do checkpoint ja
        # decaido exigiu re-warmup e custou o ext1 inteiro (val 2,55->2,81). Salvo
        # UMA vez, na entrada do decay, e nunca sobrescrito pelos saves periodicos.
        if not pre_decay_salvo and frac_tempo >= 1.0 - DECAY_FRAC and step > 0:
            ckpt_original = CKPT
            try:
                globals()["CKPT"] = os.path.join(OUT, "ckpt_pre_decay.pt")
                save_ckpt(modelo_cru, opts, scaler, step, cfg, hist,
                          gasto_anterior + time.time() - t0, ema)
                print(json.dumps({"ckpt_pre_decay": step}), flush=True)
            finally:
                globals()["CKPT"] = ckpt_original
            pre_decay_salvo = True
        lr_agora = lr_at(step, frac_tempo)
        for o in opts:
            for g in o.param_groups:
                g["lr"] = lr_agora * g.get("lr_escala", 1.0)
            o.zero_grad(set_to_none=True)
        for i_micro in range(GRAD_ACCUM):
            x, y = escolhe_fonte(train, replay, premium,
                                 em_decay=frac_tempo >= 1.0 - DECAY_FRAC)
            # so o ULTIMO micro-batch sincroniza: com DDP ingenuo seriam GRAD_ACCUM
            # all-reduces por passo, mesma conta pagando varias vezes a rede
            ultimo = i_micro == GRAD_ACCUM - 1
            ctx = model.no_sync() if (DDP_ON and not ultimo) else contextlib.nullcontext()
            with ctx:
                with torch.autocast(device_type=dev.split(":")[0], dtype=AMP_DTYPE,
                                    enabled=use_amp):
                    _, loss = model(x, y)
                    loss = loss / GRAD_ACCUM
                scaler.scale(loss).backward()
            # NAO chamar .item() aqui: e um sync host-device por micro-batch, no meio do
            # backward. Acumula na GPU e le so na avaliacao, que ja sincroniza de qualquer
            # forma. Com GRAD_ACCUM micro-batches isso eram GRAD_ACCUM paradas por step.
            loss_acum += loss.detach()
        for o in opts:
            scaler.unscale_(o)
        torch.nn.utils.clip_grad_norm_(modelo_cru.parameters(), 1.0)
        for o in opts:
            scaler.step(o)
        scaler.update()
        step += 1
        steps_janela += 1
        n_acum += GRAD_ACCUM
        if ema is not None:
            ema.atualiza(modelo_cru, step)
        if step % EVAL_EVERY == 0:
            # throughput medido na JANELA entre avaliacoes: nao adiciona sync nenhum, e
            # diluir sobre EVAL_EVERY steps tira o ruido de step a step
            dt = max(1e-9, time.time() - t_janela)
            tok_s = steps_janela * tokens_step / dt
            mfu = 100.0 * fpt * tok_s / GPU_PICO
            last = float(loss_acum.item() / max(1, n_acum)) * GRAD_ACCUM
            loss_acum.zero_(); n_acum = 0
            t_janela = time.time(); steps_janela = 0
            vl = estimate_loss(model, val)
            # a media so vale se for medida: sem este numero lado a lado nao ha como
            # decidir qual dos dois pesos exportar no fim
            vl_ema = None
            if ema is not None and ema.n > 0:
                guardado = ema.aplica_em(modelo_cru)
                vl_ema = estimate_loss(model, val)
                ema.restaura(modelo_cru, guardado)
            hist.append({"step": step, "val": round(vl, 4),
                         **({"val_ema": round(vl_ema, 4)} if vl_ema else {})})
            print(json.dumps({"step": step, "val": round(vl, 4),
                              **({"val_ema": round(vl_ema, 4)} if vl_ema else {}),
                              "train": round(last, 4), "lr": round(lr_agora, 6),
                              "tok_s": int(tok_s), "mfu": round(mfu, 2),
                              "frac_tempo": round(frac_tempo, 3),
                              "min": round((time.time() - t0) / 60, 1)}), flush=True)
            publica_progresso({
                "run": os.environ.get("RUN_NAME", "sem-nome"),
                "estado": "rodando", "step": step, "val": round(vl, 4),
                "train": round(last, 4), "lr": round(lr_agora, 6),
                "frac_tempo": round(frac_tempo, 3),
                "min": round((time.time() - t0) / 60, 1),
                "params": modelo_cru.num_params(), "val_ini": round(val_ini, 4),
                "tokens_por_step": BATCH * GRAD_ACCUM * BLOCK,
                "tokens_processados": step * BATCH * GRAD_ACCUM * BLOCK,
                "mundo": MUNDO,
                # supoe o mesmo numero de processos desde o passo 0; numa retomada com
                # outro mundo, o trecho anterior fica contado com o mundo atual
                "tokens_processados_todos": step * BATCH * GRAD_ACCUM * BLOCK * MUNDO,
                "config": {"n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD,
                           "block": BLOCK, "batch": BATCH, "grad_accum": GRAD_ACCUM,
                           "lr_pico": LR, "decay_frac": DECAY_FRAC, "budget_s": BUDGET_S},
                "hist": hist[-60:]})
        if step % CKPT_EVERY == 0:
            save_ckpt(modelo_cru, opts, scaler, step, cfg, hist,
                      gasto_anterior + time.time() - t0, ema)
            if CKPT_R2_EVERY and step % CKPT_R2_EVERY == 0:
                try:
                    import boto3
                    from botocore.config import Config
                    _s3 = boto3.client(
                        "s3",
                        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                        region_name="auto",
                        config=Config(signature_version="s3v4", retries={"max_attempts": 2},
                                      connect_timeout=10, read_timeout=60))
                    _s3.upload_file(CKPT, os.environ.get("POCKET_LM_BUCKET", "pocket-lm"),
                                    f"runs/{os.environ.get('RUN_NAME', 'sem-nome')}/ckpt.pt")
                    print(json.dumps({"ckpt_r2": step}), flush=True)
                except Exception as e:
                    print(json.dumps({"ckpt_r2_falhou": type(e).__name__}), flush=True)

    val_fim = estimate_loss(model, val)
    val_fim_ema = None
    if ema is not None and ema.n > 0:
        guardado = ema.aplica_em(modelo_cru)
        val_fim_ema = estimate_loss(model, val)
        ema.restaura(modelo_cru, guardado)
    save_ckpt(modelo_cru, opts, scaler, step, cfg, hist,
              gasto_anterior + time.time() - t0, ema)
    # "treino_terminado", nao "terminado": depois disto ainda vem a amostra e o
    # push_out do checkpoint. Marcar como terminado aqui faz quem le de fora tentar
    # baixar um artefato que ainda nao subiu (e o download devolve arquivo VAZIO, nao erro)
    publica_progresso({"run": os.environ.get("RUN_NAME", "sem-nome"),
                       "estado": "treino_terminado",
                       "step": step, "val_ini": round(val_ini, 4), "val_fim": round(val_fim, 4),
                       "min": round((time.time() - t0) / 60, 1),
                       "params": modelo_cru.num_params(), "hist": hist[-60:]}, min_intervalo=0.0)

    amostra = None
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))
        ids = tok.encode("O ").ids or [0]
        idx = torch.tensor([ids], dtype=torch.long, device=dev)
        out = modelo_cru.generate(idx, max_new_tokens=200)[0].tolist()
        amostra = tok.decode(out)
    except Exception as e:
        amostra = f"(falha ao gerar: {e})"

    rep = {"device": dev, "params": modelo_cru.num_params(), "steps_run": step - step0,
           "steps_total": step, "val_ini": round(val_ini, 4), "val_fim": round(val_fim, 4),
           "val_fim_ema": round(val_fim_ema, 4) if val_fim_ema else None,
           "otim": desc_otim,
           # replay e premium sao amostrados com reposicao: so a varredura do corpus
           # principal garante fichas distintas entre processos
           "mundo": MUNDO, "epoca_final": _sweep["epoca"],
           "tokens_processados_todos_nesta_execucao":
               (step - step0) * BATCH * GRAD_ACCUM * BLOCK * MUNDO,
           "train_ult": round(last, 4) if last else None, "amostra": amostra,
           "duracao_min": round((time.time() - t0) / 60, 1)}
    if E_MESTRE:   # como no checkpoint: N processos no mesmo caminho corrompem o arquivo
        json.dump(rep, open(REPORT, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "amostra"}), flush=True)
    print("AMOSTRA:", amostra, flush=True)
    return 0


if __name__ == "__main__":
    main()
