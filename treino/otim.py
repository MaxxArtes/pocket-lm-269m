"""Otimizador Muon e media de pesos — as duas melhorias que NAO mudam formato de tensor.

Por que estas duas primeiro: o checkpoint atual e GPT-2 (posicao aprendida, LayerNorm,
MLP GELU, MHA completa). Trocar para RoPE/GQA/SwiGLU mudaria o formato dos pesos e o
checkpoint viraria lixo silencioso. Estas duas mexem so no ESTADO DO OTIMIZADOR e numa
copia dos pesos, entao entram num treino ja em andamento.

**Muon** (Keller Jordan). Ortogonaliza a atualizacao das matrizes 2D via iteracao de
Newton-Schulz antes de aplicar. Medido com tuning justo por Stanford (arxiv 2509.02046)
em 130M-1,2B: **1,3-1,4x sobre AdamW na faixa 0,1-0,5B**, caindo para ~1,1x em 1,2B — o
ganho ENCOLHE com a escala, e 269M esta na faixa boa. Regra que nao pode ser violada:
Muon so nas matrizes 2D ocultas. Embedding, head e todo parametro 1-D ficam no AdamW —
a ortogonalizacao nao faz sentido em vetor, e em embedding degrada.

**Media de pesos (EMA)**. Tres resultados independentes de 2025/2026 (arxiv 2602.03702,
2511.18903) mostram que media dos ultimos pesos **substitui o decaimento de LR**: LR
constante + media atinge a mesma loss de um cosseno bem tunado. Resolve exatamente o
nosso problema de retomar um treino cujo cronograma de LR ja foi consumido — e custa uma
copia dos pesos, sem tocar em nenhum tensor do modelo.

AVISO sobre hiperparametro: os LRs por grupo abaixo sao ponto de partida, nao valores
tunados para o nosso modelo. O proprio repo do speedrun anota que **weight decay e o
hiperparametro mais sensivel, seguido do LR**, e que "val loss no step 1000 nao
correlaciona com a loss final" — nao julgue a troca por curva parcial nem por rodada
curta.
"""
from __future__ import annotations

import torch


def _ortogonaliza(G: torch.Tensor, passos: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz quintico: aproxima a matriz ortogonal mais proxima de G.

    Roda em bfloat16 de proposito — a iteracao e estavel em baixa precisao e e o que
    torna o custo desprezivel (~0,7% de FLOP a mais no tamanho tipico de camada). Os
    coeficientes sao os do Muon original, escolhidos para convergir rapido no intervalo
    de valores singulares que aparece na pratica, nao para precisao numerica.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposto = G.size(0) > G.size(1)
    if transposto:
        X = X.T
    for _ in range(passos):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposto:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Momento + ortogonalizacao da atualizacao. So para parametros 2D."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 passos_ns=5, weight_decay=0.0):
        super().__init__(list(params), dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                            passos_ns=passos_ns,
                                            weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        perda = closure() if closure is not None else None
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon recebeu parametro {tuple(p.shape)} de {p.ndim}D — "
                        "embedding, head e 1-D tem que ir para o AdamW")
                est = self.state[p]
                if "buf" not in est:
                    est["buf"] = torch.zeros_like(p.grad)
                buf = est["buf"]
                buf.mul_(g["momentum"]).add_(p.grad)
                d = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                d = _ortogonaliza(d, g["passos_ns"])
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                # a escala compensa matriz retangular: sem isso camadas de formatos
                # diferentes recebem passos de magnitude incompativel
                p.add_(d, alpha=-g["lr"] * max(1.0, p.size(0) / p.size(1)) ** 0.5)
        return perda


def separa_parametros(model) -> dict:
    """Divide o modelo nos quatro grupos que recebem tratamento diferente.

    `matriz` = matrizes 2D ocultas (atencao e MLP) -> Muon
    `embed`  = tabelas de embedding (token e posicao) -> AdamW, LR bem maior
    `head`   = projecao de saida, quando NAO amarrada ao embedding -> AdamW, LR menor
    `escalar`= todo parametro 1-D (normas e vieses) -> AdamW

    A separacao por nome, e nao so por dimensao, importa: o embedding tambem e 2-D e
    seria capturado por um filtro ingenuo `p.ndim == 2`, indo parar no Muon.
    """
    grupos = {"matriz": [], "embed": [], "head": [], "escalar": []}
    vistos = set()
    for nome, p in model.named_parameters():
        if not p.requires_grad or id(p) in vistos:
            continue          # embeddings amarrados aparecem duas vezes; conta uma
        vistos.add(id(p))
        if p.ndim < 2:
            grupos["escalar"].append(p)
        elif nome.startswith(("tok.", "pos.")) or "emb" in nome:
            grupos["embed"].append(p)
        elif nome.startswith("head."):
            grupos["head"].append(p)
        else:
            grupos["matriz"].append(p)
    # guarda contra renomear o modelo: se `tok.`/`pos.` virarem `wte`/`wpe` (o nome do
    # GPT-2 original), a classificacao por nome falha silenciosamente e as tabelas de
    # embedding — que sao 2D — caem no grupo `matriz` e vao para o Muon. O treino roda,
    # nao da erro, e o modelo degrada sem sinal. Todo LM tem embedding: grupo vazio aqui
    # significa que a regra de nome nao casou com esta arquitetura.
    if not grupos["embed"]:
        raise ValueError(
            "nenhum parametro classificado como embedding — a regra por nome nao casou "
            f"com este modelo. Parametros 2D vistos: "
            f"{[n for n, p in model.named_parameters() if p.ndim == 2][:8]}")
    return grupos


def monta_otimizador(model, lr_base: float, tipo: str = "adamw",
                     weight_decay: float = 0.1, mult=None):
    """Devolve (lista_de_otimizadores, descricao).

    Com `tipo='adamw'` reproduz exatamente o comportamento antigo (um AdamW sobre tudo),
    para que a comparacao A/B seja honesta: so a variavel testada muda.
    """
    def marca_escala(opts_):
        """Grava em cada grupo quanto o LR dele vale em relacao ao LR base.

        Sem isto, o laco do treino que faz `g["lr"] = lr_agora` a cada passo apagaria
        os multiplicadores por grupo — o embedding e o head voltariam a ter o mesmo LR
        das matrizes e a separacao perderia o sentido. Com a escala guardada, o
        cronograma WSD multiplica em vez de sobrescrever.
        """
        for o in opts_:
            for g_ in o.param_groups:
                g_["lr_escala"] = g_["lr"] / lr_base if lr_base else 1.0
        return opts_

    if tipo == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr_base,
                                betas=(0.9, 0.95), weight_decay=weight_decay)
        return marca_escala([opt]), {"tipo": "adamw", "grupos": 1}

    if tipo != "muon":
        raise ValueError(f"tipo de otimizador desconhecido: {tipo}")

    # multiplicadores sobre o LR base. Os relativos vem da configuracao vencedora da
    # track de otimizacao do modded-nanogpt (embedding com LR ~2 ordens de grandeza
    # acima do head); os valores absolutos AINDA NAO foram tunados para este modelo.
    m = {"matriz": 40.0, "embed": 1000.0, "head": 6.0, "escalar": 25.0}
    m.update(mult or {})
    g = separa_parametros(model)
    opts = []
    if g["matriz"]:
        opts.append(Muon(g["matriz"], lr=lr_base * m["matriz"], momentum=0.95,
                         weight_decay=weight_decay))
    resto = [(k, g[k]) for k in ("embed", "head", "escalar") if g[k]]
    if resto:
        opts.append(torch.optim.AdamW(
            [{"params": ps, "lr": lr_base * m[k],
              "weight_decay": weight_decay if k != "escalar" else 0.0}
             for k, ps in resto],
            lr=lr_base, betas=(0.9, 0.95)))
    desc = {"tipo": "muon", "mult": m,
            "n": {k: sum(p.numel() for p in v) for k, v in g.items()}}
    return marca_escala(opts), desc


class MediaPesos:
    """Media exponencial dos pesos, mantida em fp32 na CPU.

    Na CPU de proposito: a copia nao disputa memoria da GPU com o treino, e o custo de
    copiar 269M floats a cada N passos e desprezivel perto de um passo de treino.

    `a_cada` existe porque atualizar a cada passo nao muda o resultado de forma
    perceptivel e multiplica a transferencia GPU->CPU. O horizonte efetivo da media e
    aproximadamente `a_cada / (1 - decaimento)` passos de treino.
    """

    def __init__(self, model, decaimento: float = 0.999, a_cada: int = 10,
                 horizonte: int = 0):
        self.a_cada = max(1, a_cada)
        if horizonte > 0:
            # horizonte em PASSOS DE TREINO, que e a unidade em que se pensa. Sem isto
            # a escolha do decaimento e as cegas: medido aqui em 28/07, decaimento 0.999
            # com a_cada=5 da horizonte de 5000 passos, e num run de 1200 a media ficou
            # em 7.37 contra 3.64 dos pesos vivos — ela nunca alcancou o modelo. Quem
            # olhasse so o numero concluiria que media de pesos nao funciona.
            self.decaimento = max(0.0, 1.0 - self.a_cada / float(horizonte))
        else:
            self.decaimento = decaimento
        self.horizonte = int(self.a_cada / max(1e-9, 1.0 - self.decaimento))
        self.n = 0
        self.sombra = {k: v.detach().float().cpu().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def atualiza(self, model, step: int) -> None:
        if step % self.a_cada:
            return
        d = self.decaimento
        # os primeiros passos usam media aritmetica: com a sombra ainda "jovem", o
        # decaimento fixo a deixaria presa nos pesos iniciais por milhares de passos
        if self.n < 10:
            d = min(d, self.n / (self.n + 1.0))
        for k, v in model.state_dict().items():
            if k in self.sombra:
                self.sombra[k].mul_(d).add_(v.detach().float().cpu(), alpha=1 - d)
        self.n += 1

    def estado(self) -> dict:
        return {"decaimento": self.decaimento, "a_cada": self.a_cada,
                "n": self.n, "sombra": self.sombra}

    def carrega(self, est: dict) -> None:
        self.decaimento = est.get("decaimento", self.decaimento)
        self.a_cada = est.get("a_cada", self.a_cada)
        self.n = est.get("n", 0)
        # o checkpoint e carregado com map_location=dev: sem voltar para CPU/fp32, a
        # sombra ficaria na GPU e o proximo `atualiza` somaria tensores de dois dispositivos
        sombra = est.get("sombra")
        if sombra is not None:
            # checkpoints antigos sob DDP gravaram a media com o prefixo `module.`
            self.sombra = {k.removeprefix("module."): v.detach().float().cpu()
                           for k, v in sombra.items()}

    @torch.no_grad()
    def aplica_em(self, model) -> dict:
        """Troca os pesos do modelo pela media e devolve os originais, para restaurar."""
        sd = model.state_dict()
        guardado = {k: sd[k].detach().clone() for k in self.sombra}
        for k, v in self.sombra.items():
            sd[k].copy_(v.to(sd[k].device, sd[k].dtype))
        return guardado

    @torch.no_grad()
    def restaura(self, model, guardado: dict) -> None:
        sd = model.state_dict()
        for k, v in guardado.items():
            sd[k].copy_(v)
