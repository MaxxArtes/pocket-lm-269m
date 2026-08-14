"""SFT de conversa sobre o base pt-BR (269M). Feito para rodar na T4 gratis do Kaggle:
fp16 + GradScaler (Turing nao tem bf16), teto de tempo com desligamento limpo, telemetria
e checkpoint pro R2 (a CLI do Kaggle nao entrega log de sessao em andamento).

  RUN_ORIGEM=pocket-269m-ext3 DIALOGOS=dialogos.jsonl RUN_NAME=sft-v1 python train_sft.py

Decisoes que NAO sao obvias e ja custaram caro em outro lugar:

- Perda: os tokens da resposta pesam 1,0 e os do prompt pesam PLW=0,15, em vez da mascara
  dura de antes. O <|im_end|> que fecha a resposta pesa 1,0 junto com ela — mascarar o
  im_end e o erro classico, o modelo nunca aprende a parar e fala sozinho ate o limite.
  O porque do 0,15 e a ressalva de escala estao em `codifica`.
- Vocabulario cresce de 16.384 para 16.512, nao para 16.386: os 2 tokens de ChatML mais
  126 slots reservados mantem o alinhamento de 128 (kernel de matmul) e dao espaco pra
  token novo depois sem re-treinar. Os slots viram <|unused_N|> no tokenizer para que o
  tamanho do vocabulario do tokenizer e o da matriz batam — se nao baterem, a conversao
  pra GGUF quebra no fim, depois do treino todo.
- Linhas novas da embedding nascem na MEDIA das existentes + ruido, nao em N(0, 0.02):
  a media ja esta na regiao onde a camada seguinte espera receber ativacao; um vetor
  aleatorio no meio de pesos treinados produz logit fora de escala nos primeiros passos.
- TOKENS_WARMUP: nos primeiros passos SO a embedding recebe gradiente E o gradiente das
  16.384 linhas ANTIGAS e zerado. Sem esse zeramento o warmup faz o oposto do que
  promete: como a head compartilha a matriz (weight tying), "treinar so a embedding"
  atualiza TODAS as linhas, inclusive as ja bem ajustadas pelo pre-treino. Achado na
  revisao do agy, 31/07.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn

from model import GPT, GPTConfig

DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT_DIR", os.path.join(DIR, "out"))
CKPT = os.path.join(OUT, "ckpt_sft.pt")
BUCKET = os.environ.get("POCKET_LM_BUCKET", "pocket-lm")
RUN_NAME = os.environ.get("RUN_NAME", "sft-sem-nome")

DIALOGOS = os.environ.get("DIALOGOS", os.path.join(DIR, "conversa_out", "piloto2k.jsonl"))
TOKENIZER_BASE = os.environ.get("TOKENIZER", os.path.join(DIR, "data_big", "tokenizer.json"))
BLOCK = int(os.environ.get("BLOCK", "512"))
BATCH = int(os.environ.get("BATCH", "8"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
EPOCAS = float(os.environ.get("EPOCAS", "2"))
# 2e-5. A historia deste numero, porque ele ja custou duas rodadas:
#
# Comecou em 1e-3, copiado da receita do SmolLM2-135M/360M (alignment-handbook,
# config_smol.yaml). O erro nao foi a fonte, foi copiar o LR sem o DENOMINADOR: a
# configuracao deles usa contexto 8192 e lote efetivo de 131.072 tokens por passo; a nossa
# usa 16.384 — **lote 8x menor**. LR escala com tamanho de lote. Resultado medido em
# 02/08: a perda de validacao SUBIU de 2,63 para 3,81 em 300 passos, divergindo.
#
# 1e-4 (escalonamento linear por lote: 1e-3/8 = 1,25e-4) funcionou — val de 2,241 para
# 0,2152 — mas custou 12,3 pontos de CALAME-PT, confirmados por log-verossimilhanca
# (+0,58 nat) e portanto esquecimento REAL, nao artefato de decodificacao.
#
# 2e-5 vem de medicao em OLMo-1 1B, OLMo-2 1B, Gemma-3 1B, Hubble 1B e SmolLM-3 3B
# (arXiv:2604.13627): com a MESMA perda de treino, LR maior da desempenho fora de
# distribuicao consistentemente pior, e a recomendacao literal e usar o menor LR que ainda
# atinja desempenho aceitavel. O limite do FINCH (arXiv:2605.20005) — esquecimento por
# passo <= LR x raiz(perda) — explica por que o dano se concentra no inicio, quando a
# perda ainda e alta.
# 1e-4, e NAO os 2e-5 que eu tinha colocado. Dois motivos, o segundo e um erro meu:
#
# (a) 2e-5 e receita de 7B (Zephyr-7B, Tulu 3). A configuracao publica mais proxima em
#     escala — SmolLM2-360M em SFT — usa 1,0e-3, e a mais proxima em TAREFA (BERT em SQuAD,
#     110M-340M, QA extrativa) usa 5e-5. Nao ha receita medida em 100M-1B usando 2e-5.
# (b) o cosseno com piso 0,1 partindo de 1e-4 TERMINAVA em 1e-5. Uma constante de 2e-5
#     termina ACIMA disso. A minha mudanca nao era "menos taxa" — era menos no comeco e
#     MAIS no fim, que e a pior das duas metades.
LR = float(os.environ.get("LR", "1e-4"))
LR_FLOOR_FRAC = float(os.environ.get("LR_FLOOR_FRAC", "0.1"))
# "cosseno" (padrao antigo) ou "constante" apos o aquecimento. O cosseno mantem LR alto
# justamente no inicio, que e onde o limite do FINCH diz que o esquecimento acontece.
# "wsd" (padrao), "cosseno" ou "constante". WSD = aquecimento, estavel, e resfriamento
# nos ultimos COOLDOWN_FRAC. Reproduz o comeco e o FIM da rodada 2 que funcionou (que
# terminava em 1e-5), sem manter taxa alta no meio. O ganho do resfriamento satura em ~20%
# dos passos (arXiv:2405.18392, 33M-360M).
AGENDA = os.environ.get("AGENDA_LR", "wsd")
COOLDOWN_FRAC = float(os.environ.get("COOLDOWN_FRAC", "0.2"))
WARMUP_FRAC = float(os.environ.get("WARMUP_FRAC", "0.1"))   # 10%, mesma receita
WARMUP = int(os.environ.get("WARMUP", "0"))                 # 0 = deriva de WARMUP_FRAC
TOKENS_WARMUP = int(os.environ.get("TOKENS_WARMUP", "40"))
# peso da perda nos tokens do PROMPT. 0 = mascara dura (o que se fazia antes), 1 = treina
# tudo igual. 0,15 vem do GAM de Huerta-Enochian (arXiv:2401.13586) para Rg <= 1; a
# regressao deles da 0,242. Ver `codifica` para o porque e para a ressalva de escala.
PLW = float(os.environ.get("PLW", "0.15"))
# 0, e nao 0,01: a receita medida do SmolLM2-135M/360M (config_smol.yaml) nao declara
# weight_decay, e o default do HF Trainer e 0,0. O 0,01 aqui era suposicao minha.
WD = float(os.environ.get("WD", "0.0"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.02"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "100"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "500"))
# teto de tempo: sessao do Kaggle morre em 12h SEM aviso e leva o disco junto. Parar em
# 11h com checkpoint no R2 e a diferenca entre "treino terminou" e "perdeu tudo".
BUDGET_S = int(os.environ.get("TRAIN_BUDGET_S", str(11 * 3600)))
# replay de pre-treino contra perda de capacidade base (CALAME/LAMBADA). 0 desliga.
REPLAY_BIN = os.environ.get("REPLAY_BIN", "")
# 5%, e nao os 2% que estavam aqui por conta de Scialom et al. (arXiv:2205.12393, T0-3B).
# Ibrahim et al. (arXiv:2403.08763, Tabela 2) mediram replay em 405M — a
# NOSSA escala. Em deslocamento forte, 1% recupera 0,73 nat de esquecimento custando 0,01
# nat no dominio novo; 5% recupera 0,99 pelo mesmo custo; so em 50% o custo aparece (0,11).
# Eles adotaram 5% para deslocamento fraco, que e o nosso caso (SFT em pt-BR sobre
# pre-treino em pt-BR). Ressalva: a medicao e de pre-treino continuado, nao de SFT.
# 10%, e nao 25%. Ibrahim et al. (arXiv:2403.08763, 405M — a escala publicada mais proxima
# da nossa) medem em deslocamento forte: 10% recupera 79% do esquecimento por +0,02 nat de
# custo de adaptacao; 25% recupera 88% por +0,05. O retorno marginal cai por fator ~200 do
# primeiro ao ultimo ponto da curva. 25% esta no joelho, pagando o dobro por 9 pontos.
REPLAY_FRAC = float(os.environ.get("REPLAY_FRAC", "0.10")) if REPLAY_BIN else 0.0
# ADITIVO: o lote de replay NAO substitui um lote de tarefa. Substituir cortaria os passos
# de tarefa em 10% ao mesmo tempo que a agenda ja e conservadora — dois freios de uma vez.
REPLAY_ADITIVO = os.environ.get("REPLAY_ADITIVO", "1") == "1"
SEMENTE = int(os.environ.get("SEED", "1337"))
# conjunto de desenvolvimento para EM/F1 durante o treino. Vazio desliga a selecao por
# qualidade e o treino volta a guardar o ultimo passo.
QA_DEV = os.environ.get("QA_DEV", "")
QA_DEV_N = int(os.environ.get("QA_DEV_N", "160"))

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
VOCAB_BASE = 16384
VOCAB_NOVO = int(os.environ.get("VOCAB_NOVO", "16512"))   # multiplo de 128

dev = "cuda" if torch.cuda.is_available() else "cpu"
def _dtype_amp():
    """bf16 SO com Tensor Core de verdade (Ampere, capacidade >= 8.0).

    `torch.cuda.is_bf16_supported()` devolve True na T4 (Turing, sm_75) por EMULACAO — nao
    ha GEMM bf16 passando por Tensor Core abaixo de 8.0. A rodada 2 rodou assim: 13 s/passo
    e ~4% de MFU, contra os ~2,4 s/passo que 16% de MFU dariam. Foram ~7 horas onde caberiam
    ~2. Nao e questao de precisao, e de nao existir o caminho rapido no silicio."""
    if dev != "cuda":
        return torch.float16
    maj, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if maj >= 8 else torch.float16


AMP_DTYPE = _dtype_amp()
torch.manual_seed(SEMENTE)
_rng = random.Random(SEMENTE)
_pub = {"ultimo": 0.0, "erros": 0}


def s3():
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 2},
                      connect_timeout=10, read_timeout=60))


def publica(payload: dict, min_intervalo: float = 60.0, forcar: bool = False) -> None:
    """Telemetria pro R2. Nunca derruba o treino; desiste depois de 5 erros seguidos."""
    if _pub["erros"] >= 5:
        return
    agora = time.time()
    if not forcar and agora - _pub["ultimo"] < min_intervalo:
        return
    _pub["ultimo"] = agora
    try:
        os.makedirs(OUT, exist_ok=True)
        local = os.path.join(OUT, "progress_sft.json")
        with open(local + ".tmp", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(local + ".tmp", local)
        s3().upload_file(local, BUCKET, f"live/{RUN_NAME}/progress.json")
        _pub["erros"] = 0
    except Exception as e:  # noqa: BLE001
        _pub["erros"] += 1
        print(json.dumps({"aviso": "publica falhou", "erro": type(e).__name__}), flush=True)


def monta_tokenizer_chat() -> "object":
    """Base 16.384 + im_start/im_end + reservados ate VOCAB_NOVO. Salvo em OUT para o
    export de GGUF usar EXATAMENTE este arquivo — tokenizer diferente do treino gera
    modelo que responde lixo sem dar erro em lugar nenhum."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER_BASE)
    base = tok.get_vocab_size(with_added_tokens=True)
    if base != VOCAB_BASE:
        print(json.dumps({"aviso": "vocab base inesperado", "vi": base}), flush=True)
    novos = [IM_START, IM_END] + [f"<|unused_{i}|>" for i in range(VOCAB_NOVO - base - 2)]
    from tokenizers import AddedToken
    tok.add_special_tokens([AddedToken(t, normalized=False, special=True) for t in novos])
    destino = os.path.join(OUT, "tokenizer_chat.json")
    os.makedirs(OUT, exist_ok=True)
    tok.save(destino)
    return tok


def cresce_vocab(model: GPT, novo: int) -> None:
    """Embedding e head compartilham peso (weight tying) — uma matriz so pra crescer."""
    antigo = model.tok.weight.shape[0]
    if antigo >= novo:
        return
    with torch.no_grad():
        w = model.tok.weight.data
        media = w.mean(dim=0, keepdim=True)
        desvio = w.std().item()
        extra = media.repeat(novo - antigo, 1) + torch.randn(
            novo - antigo, w.shape[1], device=w.device, dtype=w.dtype) * (desvio * 0.1)
        nova = torch.cat([w, extra], dim=0)
    emb = nn.Embedding(novo, model.c.n_embd).to(nova.device, nova.dtype)
    emb.weight.data.copy_(nova)
    head = nn.Linear(model.c.n_embd, novo, bias=False).to(nova.device, nova.dtype)
    head.weight = emb.weight
    model.tok, model.head, model.c.vocab_size = emb, head, novo


def parte_dialogo(texto: str) -> list[tuple[str, str]]:
    """'U: ...\\nA: ...' -> [(papel, conteudo)]. Conteudo pode ter varias linhas."""
    turnos: list[tuple[str, str]] = []
    papel, buf = None, []
    for linha in texto.splitlines():
        m = re.match(r"^([UA]):\s?(.*)$", linha)
        if m:
            if papel:
                turnos.append((papel, "\n".join(buf).strip()))
            papel, buf = ("user" if m.group(1) == "U" else "assistant"), [m.group(2)]
        elif papel:
            buf.append(linha)
    if papel:
        turnos.append((papel, "\n".join(buf).strip()))
    return [t for t in turnos if t[1]]


def codifica(tok, turnos: list[tuple[str, str]]) -> tuple[list[int], list[int], list[float]] | None:
    """ChatML com peso PLW no prompt e 1.0 na resposta (o <|im_end|> dela INCLUSO).

    Mudou de mascara dura (peso 0) para peso PLW por medicao. Huerta-Enochian
    (arXiv:2401.13586) define Rg = comprimento da resposta / comprimento do prompt e roda
    60 experimentos variando o peso do prompt. O caso `AlpacaDataShort`, com Rg = 0,08,
    deu textualmente: *"models fine-tuned with prompt loss masking (wp = 0) almost always
    produced the worst scores"*. O peso otimo pela regressao foi 0,242; pelo GAM, 0,155.

    O nosso Rg e ~0,03 — resposta de ~8 tokens contra prompt de ~230 — ou seja, MAIS
    extremo que o caso medido, e do lado onde a mascara total perde. Com peso 0, so ~4%
    dos tokens dao gradiente, e as camadas de baixo quase nao recebem sinal para
    relacionar pergunta e passagem.

    RESSALVA HONESTA: as duas medicoes (esta e Shi et al., arXiv:2405.14394) sao a 7B.
    Nao existe medicao publicada abaixo de 1B. O raciocinio de que regularizacao importa
    mais em modelo pequeno com 45k exemplos favorece PLW>0, mas e raciocinio, nao medida —
    por isso PLW e variavel de ambiente e entra no A/B contra 0.
    """
    ids: list[int] = []
    rot: list[int] = []
    peso: list[float] = []

    def mete(txt: str, resposta: bool):
        e = tok.encode(txt, add_special_tokens=False).ids
        ids.extend(e)
        rot.extend(e)
        peso.extend([1.0 if resposta else PLW] * len(e))

    for papel, conteudo in turnos:
        mete(f"{IM_START}{papel}\n", False)
        mete(conteudo, papel == "assistant")
        mete(f"{IM_END}\n", papel == "assistant")
    if len(ids) < 8 or len(ids) > BLOCK:
        return None
    return ids, rot, peso


def carrega_dialogos(tok) -> tuple[list, list, dict]:
    brutos = [json.loads(l) for l in open(DIALOGOS, encoding="utf-8") if l.strip()]
    amostras, longos, vazios = [], 0, 0
    for d in brutos:
        # `turnos` pronto tem PRECEDENCIA sobre o texto "U:/A:". O QA extrativo cola uma
        # passagem inteira dentro do turno do usuario, e o `parte_dialogo` remonta turno
        # por regex de inicio de linha: uma materia com linha comecando em "A:" ou "U:"
        # (fala citada, sigla, item de lista) seria cortada em turno falso, em silencio.
        # Com os turnos ja estruturados, nada dentro da passagem vira estrutura.
        if isinstance(d.get("turnos"), list):
            turnos = [(p, c) for p, c in d["turnos"] if c and c.strip()]
        else:
            turnos = parte_dialogo(d.get("dialogo", ""))
        if not turnos or turnos[-1][0] != "assistant":
            vazios += 1
            continue
        cod = codifica(tok, turnos)
        if cod is None:
            longos += 1
            continue
        amostras.append(cod)
    _rng.shuffle(amostras)
    n_val = max(8, int(len(amostras) * VAL_FRAC))
    stats = {"lidos": len(brutos), "usados": len(amostras),
             "descartados_tamanho": longos, "descartados_forma": vazios,
             "tokens_resposta": sum(sum(1 for p in a[2] if p >= 1.0)
                                    for a in amostras[n_val:]),
             "tokens_prompt": sum(sum(1 for p in a[2] if 0 < p < 1.0)
                                  for a in amostras[n_val:]),
             "plw": PLW}
    return amostras[n_val:], amostras[:n_val], stats


def lote(amostras: list, idxs: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.zeros(len(idxs), BLOCK, dtype=torch.long)
    y = torch.full((len(idxs), BLOCK), -100, dtype=torch.long)
    # peso 0 no ENCHIMENTO. O -100 do y ja zera a perda ali, mas o peso tem que zerar
    # tambem: ele e o denominador da media, e contar posicao vazia no denominador
    # encolheria a perda proporcionalmente ao quanto a amostra e curta.
    w = torch.zeros(len(idxs), BLOCK, dtype=torch.float32)
    for j, i in enumerate(idxs):
        ids, rot, peso = amostras[i]
        # alvo e o token SEGUINTE: entra ids[:-1], aprende rot[1:]
        n = len(ids) - 1
        x[j, :n] = torch.tensor(ids[:-1], dtype=torch.long)
        y[j, :n] = torch.tensor(rot[1:], dtype=torch.long)
        w[j, :n] = torch.tensor(peso[1:], dtype=torch.float32)
    return (x.to(dev, non_blocking=True), y.to(dev, non_blocking=True),
            w.to(dev, non_blocking=True))


def perda_ponderada(model, x: torch.Tensor, y: torch.Tensor,
                    w: torch.Tensor) -> torch.Tensor:
    """Perda com peso por token. Substitui o `model(x, y)`, que faz media uniforme."""
    logits, _ = model(x)
    ce = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), y.view(-1), reduction="none")
    wf = w.view(-1)
    return (ce * wf).sum() / wf.sum().clamp(min=1e-6)


def lote_replay(dados: np.ndarray, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    ix = np.random.randint(0, len(dados) - BLOCK - 1, size=n)
    x = torch.from_numpy(np.stack([dados[i:i + BLOCK].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([dados[i + 1:i + 1 + BLOCK].astype(np.int64) for i in ix]))
    return x.to(dev), y.to(dev)


@torch.no_grad()
def avalia(model, amostras) -> float:
    model.eval()
    perdas = []
    for i in range(0, len(amostras), BATCH):
        x, y, w = lote(amostras, list(range(i, min(i + BATCH, len(amostras)))))
        with torch.autocast(device_type=dev, dtype=AMP_DTYPE, enabled=dev == "cuda"):
            loss = perda_ponderada(model, x, y, w)
        perdas.append(loss.item())
    model.train()
    return float(np.mean(perdas)) if perdas else float("nan")


@torch.no_grad()
def avalia_qa(model, tok, itens: list) -> dict:
    """Qualidade REAL: EM/F1 gerando de verdade, com os dois lados separados.

    Existe porque a perda de validacao nao serve para escolher checkpoint. A LIMA
    (arXiv:2305.11206, apendice B) mediu perplexidade SUBINDO enquanto a qualidade de
    geracao MELHORAVA, e por isso abandonaram perplexidade e escolheram o checkpoint a mao
    num conjunto pequeno. Guardar o ultimo passo, ou o de menor perda, e apostar contra
    essa medicao.

    Os dois lados vao separados porque os modos de falha sao opostos e a media aritmetica
    os esconde: "recusa tudo" da NoAns 100 / HasAns 0, "nunca recusa" da o inverso. A
    media HARMONICA so sobe se os dois subirem — e o ChatQA (arXiv:2401.10225, tabela 8)
    mostra o trade-off acontecendo de verdade entre modelos reais.
    """
    import avalia_qa as aq
    model.eval()
    has, noans, extrativas, geradas = [], [], 0, 0
    for it in itens:
        usuario, ouro = it["turnos"][0][1], it["turnos"][1][1]
        pred = aq.responde(model, model.c, tok, usuario, dev=dev)
        if it.get("respondivel"):
            has.append((aq.em(pred, ouro), aq.f1(pred, ouro)))
            geradas += 1
            if not aq.e_recusa(pred) and aq.normaliza(pred) in aq.normaliza(usuario):
                extrativas += 1
        else:
            noans.append(float(aq.e_recusa(pred)))
    model.train()
    hf = 100 * sum(x[1] for x in has) / max(1, len(has))
    ne = 100 * sum(noans) / max(1, len(noans))
    harm = 0.0 if hf <= 0 or ne <= 0 else 2 * hf * ne / (hf + ne)
    return {"HasAns_EM": round(100 * sum(x[0] for x in has) / max(1, len(has)), 2),
            "HasAns_F1": round(hf, 2), "NoAns_EM": round(ne, 2),
            "media_harmonica": round(harm, 2),
            "extratividade": round(100 * extrativas / max(1, geradas), 2),
            "n_has": len(has), "n_noans": len(noans)}


def baixa_base() -> dict:
    """Checkpoint do base: R2 (RUN_ORIGEM) ou arquivo local em CKPT_BASE."""
    local = os.environ.get("CKPT_BASE", "")
    if not local:
        origem = os.environ["RUN_ORIGEM"]
        local = os.path.join(OUT, "base.pt")
        os.makedirs(OUT, exist_ok=True)
        if not os.path.exists(local):
            s3().download_file(BUCKET, f"runs/{origem}/ckpt.pt", local)
    return torch.load(local, map_location="cpu", weights_only=False)


def salva(model, opt, step, hist, stats, subir=True, sufixo: str = "") -> None:
    """`sufixo` separa o checkpoint do MELHOR do checkpoint periodico. Sem ele os dois
    escreveriam no mesmo arquivo e o salvamento seguinte apagaria o melhor — que e
    exatamente o problema que a selecao por qualidade existe para resolver."""
    os.makedirs(OUT, exist_ok=True)
    alvo = CKPT.replace(".pt", f"{sufixo}.pt") if sufixo else CKPT
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "cfg": vars(model.c), "hist": hist[-500:], "dados": stats,
                "chatml": {"im_start": IM_START, "im_end": IM_END}}, alvo + ".tmp")
    os.replace(alvo + ".tmp", alvo)
    if subir:
        try:
            s3().upload_file(alvo, BUCKET,
                             f"runs/{RUN_NAME}/ckpt_sft{sufixo}.pt")
            s3().upload_file(os.path.join(OUT, "tokenizer_chat.json"), BUCKET,
                             f"runs/{RUN_NAME}/tokenizer_chat.json")
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"aviso": "upload ckpt falhou", "erro": type(e).__name__}),
                  flush=True)


def main() -> int:
    t0 = time.time()
    tok = monta_tokenizer_chat()
    treino, val, stats = carrega_dialogos(tok)
    if not treino:
        raise SystemExit("nenhum dialogo utilizavel")

    d = baixa_base()
    cfg = GPTConfig(**d["cfg"])
    model = GPT(cfg)
    model.load_state_dict(d["model"] if "model" in d else d, strict=True)
    cresce_vocab(model, VOCAB_NOVO)
    model.to(dev)

    replay = None
    if REPLAY_BIN:
        replay = np.memmap(REPLAY_BIN, dtype=np.uint16, mode="r")

    passos_epoca = max(1, len(treino) // (BATCH * GRAD_ACCUM))
    total = int(passos_epoca * EPOCAS)
    global WARMUP
    if WARMUP <= 0:
        WARMUP = max(10, int(total * WARMUP_FRAC))
    sem_wd = [p for n, p in model.named_parameters() if p.dim() < 2 or "tok" in n]
    com_wd = [p for n, p in model.named_parameters() if p.dim() >= 2 and "tok" not in n]
    opt = torch.optim.AdamW([{"params": com_wd, "weight_decay": WD},
                             {"params": sem_wd, "weight_decay": 0.0}],
                            lr=LR, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(enabled=AMP_DTYPE == torch.float16 and dev == "cuda")

    def lr_no(step: int) -> float:
        if step < WARMUP:
            return LR * (step + 1) / WARMUP
        if AGENDA == "wsd":
            inicio_cool = total - int(total * COOLDOWN_FRAC)
            if step < inicio_cool:
                return LR
            g = (step - inicio_cool) / max(1, total - inicio_cool)
            return LR * (LR_FLOOR_FRAC + (1 - LR_FLOOR_FRAC) * (1 - g))
        if AGENDA == "constante":
            # constante apos o aquecimento. O cosseno mantem LR alto no INICIO, e o limite
            # do FINCH (esquecimento por passo <= LR x raiz(perda)) diz que e exatamente
            # ali, com a perda ainda alta, que o dano acontece. Com LR ja baixo (2e-5) nao
            # ha o que decair: decair so subtreina a segunda metade.
            return LR
        f = (step - WARMUP) / max(1, total - WARMUP)
        return LR * (LR_FLOOR_FRAC + (1 - LR_FLOOR_FRAC) * 0.5 * (1 + math.cos(math.pi * min(1.0, f))))

    qa_dev = []
    if QA_DEV and os.path.exists(QA_DEV):
        qa_dev = [json.loads(l) for l in open(QA_DEV, encoding="utf-8") if l.strip()]
        _rng.shuffle(qa_dev)
        # equilibra os lados: com 20% de recusa, uma amostra de 160 traria ~32 negativos e
        # a estimativa do lado que mais importa sairia com intervalo largo demais
        resp = [d for d in qa_dev if d.get("respondivel")]
        recu = [d for d in qa_dev if not d.get("respondivel")]
        meio = QA_DEV_N // 2
        qa_dev = resp[:meio] + recu[:meio]
        _rng.shuffle(qa_dev)
        print(json.dumps({"qa_dev": len(qa_dev), "respondiveis": len(resp[:meio]),
                          "recusas": len(recu[:meio])}), flush=True)
    hist: list[dict] = []
    qa_hist: list[dict] = []
    melhor = {"media_harmonica": -1.0}
    print(json.dumps({"inicio": RUN_NAME, "dados": stats, "passos": total,
                      "epocas": EPOCAS, "lr": LR, "replay": REPLAY_FRAC,
                      "vocab": VOCAB_NOVO, "params": sum(p.numel() for p in model.parameters()),
                      "val0": None}), flush=True)

    ordem: list[int] = []
    model.train()
    for step in range(total):
        if time.time() - t0 > BUDGET_S:
            print(json.dumps({"parada": "teto de tempo", "step": step}), flush=True)
            break
        # so a embedding treina enquanto os tokens novos ainda sao ruido
        if step == 0 and TOKENS_WARMUP > 0:
            for n, p in model.named_parameters():
                p.requires_grad_("tok" in n or "head" in n)
        if step == TOKENS_WARMUP:
            for p in model.parameters():
                p.requires_grad_(True)

        for g in opt.param_groups:
            g["lr"] = lr_no(step)
        opt.zero_grad(set_to_none=True)
        soma = 0.0
        for _ in range(GRAD_ACCUM):
            # BUG CORRIGIDO em 02/08: o ramo de replay definia x e y mas NAO o `w`, e a
            # perda ponderada usava a mascara do lote ANTERIOR, de tarefa. Onde aquele lote
            # tinha enchimento o peso era zero — e os tokens de replay naquelas posicoes
            # eram simplesmente IGNORADOS. O replay de 5% da rodada 2 valia uma fracao
            # desconhecida e menor disso, o que ajuda a explicar por que nao segurou o
            # CALAME. Replay e texto corrido: todo token conta, peso 1,0 uniforme.
            usa_replay = replay is not None and _rng.random() < REPLAY_FRAC
            if usa_replay and not REPLAY_ADITIVO:
                x, y = lote_replay(replay, BATCH)
                w = torch.ones_like(y, dtype=torch.float32)
            else:
                if len(ordem) < BATCH:
                    ordem = list(range(len(treino)))
                    _rng.shuffle(ordem)
                idxs, ordem = ordem[:BATCH], ordem[BATCH:]
                x, y, w = lote(treino, idxs)
            with torch.autocast(device_type=dev, dtype=AMP_DTYPE, enabled=dev == "cuda"):
                loss = perda_ponderada(model, x, y, w)
            if torch.isnan(loss):
                continue
            scaler.scale(loss / GRAD_ACCUM).backward()
            soma += loss.item() / GRAD_ACCUM
            # ADITIVO: o replay entra como micro-lote EXTRA, sem consumir um passo de
            # tarefa. Substituir cortaria os passos de tarefa na mesma proporcao, e com a
            # agenda ja conservadora seriam dois freios de uma vez.
            if usa_replay and REPLAY_ADITIVO:
                xr, yr = lote_replay(replay, BATCH)
                wr = torch.ones_like(yr, dtype=torch.float32)
                with torch.autocast(device_type=dev, dtype=AMP_DTYPE, enabled=dev == "cuda"):
                    lr_loss = perda_ponderada(model, xr, yr, wr)
                if not torch.isnan(lr_loss):
                    scaler.scale(lr_loss / GRAD_ACCUM).backward()
        if step < TOKENS_WARMUP and model.tok.weight.grad is not None:
            model.tok.weight.grad[:VOCAB_BASE].zero_()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        # fp16 na T4: pico de loss dos tokens novos pode fazer o GradScaler encolher a
        # escala ate o treino parar de andar sem lancar erro nenhum. Avisa alto.
        if scaler.is_enabled() and scaler.get_scale() < 1.0:
            print(json.dumps({"aviso": "escala do fp16 colapsando",
                              "escala": scaler.get_scale(), "step": step}), flush=True)
        if (step + 1) % EVAL_EVERY == 0 or step == total - 1:
            v = avalia(model, val)
            hist.append({"step": step + 1, "train": round(soma, 4), "val": round(v, 4),
                         "lr": round(lr_no(step), 8),
                         "min": round((time.time() - t0) / 60, 1)})
            print(json.dumps(hist[-1]), flush=True)
            # `qa_hist` VAI junto: a avaliacao por EM/F1 e o numero que decide o
            # checkpoint, e ate agora ela so era impressa no log — que o Kaggle entrega
            # apenas DEPOIS que o kernel termina. Resultado: o unico sinal ao vivo era a
            # perda, justamente a metrica que a LIMA mostrou ser anticorrelacionada com
            # qualidade. Sem isto, decidir no meio do treino e impossivel.
            publica({"run": RUN_NAME, "hist": hist[-200:], "dados": stats,
                     "total": total, "qa_hist": qa_hist[-20:], "qa_melhor": melhor})
        if CKPT_EVERY and (step + 1) % CKPT_EVERY == 0:
            # QUALIDADE decide, nao perda. Guarda o MELHOR por media harmonica em vez de
            # deixar o ultimo salvamento sobrescrever tudo — ver `avalia_qa` para o porque.
            # Amostra pequena de proposito: geracao gulosa item a item e cara, e a decisao
            # aqui e "este checkpoint e melhor que o guardado", nao a medida final.
            if qa_dev:
                q = avalia_qa(model, tok, qa_dev[:QA_DEV_N])
                q["step"] = step + 1
                qa_hist.append(q)
                print(json.dumps({"qa": q}), flush=True)
                if q["media_harmonica"] > melhor["media_harmonica"]:
                    melhor = dict(q)
                    salva(model, opt, step + 1, hist, stats, sufixo="_melhor")
            salva(model, opt, step + 1, hist, stats)

    v = avalia(model, val)
    if qa_dev:
        q = avalia_qa(model, tok, qa_dev[:QA_DEV_N])
        q["step"] = total
        qa_hist.append(q)
        print(json.dumps({"qa_final": q}), flush=True)
        if q["media_harmonica"] > melhor["media_harmonica"]:
            melhor = dict(q)
            salva(model, opt, total, hist, stats, sufixo="_melhor")
    salva(model, opt, total, hist, stats)
    rel = {"run": RUN_NAME, "val_final": round(v, 4), "hist": hist, "dados": stats,
           # CONFIG COMPLETA no relatorio, e nao so no comando que lancou. Os relatorios
           # das rodadas de pre-treino (classicc-v1, ext1, ext2, ext3) gravaram passos e
           # perda mas NAO gravaram lote, acumulacao nem tokens — resultado: a contagem de
           # tokens do modelo nao e verificavel pelo registro, so por arqueologia no script
           # de lancamento. A checklist da ACL (item C1/C2) pede exatamente isso.
           "epocas": EPOCAS, "lr": LR, "replay": REPLAY_FRAC, "plw": PLW,
           "batch": BATCH, "grad_accum": GRAD_ACCUM, "block": BLOCK,
           "lote_efetivo": BATCH * GRAD_ACCUM,
           "tokens_por_passo": BATCH * GRAD_ACCUM * BLOCK,
           "tokens_processados": total * BATCH * GRAD_ACCUM * BLOCK,
           "wd": WD, "warmup_frac": WARMUP_FRAC, "semente": SEMENTE,
           "rodada_unica": True,   # ACL C3: declarar e obrigatorio; N seeds nao e
           "amp_dtype": str(AMP_DTYPE), "dispositivo": dev,
           "qa_hist": qa_hist, "qa_melhor": melhor,
           "minutos": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(OUT, "report_sft.json"), "w", encoding="utf-8") as f:
        json.dump(rel, f, ensure_ascii=False, indent=1)
    try:
        s3().upload_file(os.path.join(OUT, "report_sft.json"), BUCKET,
                         f"runs/{RUN_NAME}/report.json")
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"aviso": "upload report falhou", "erro": type(e).__name__}), flush=True)
    print(json.dumps(rel, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
