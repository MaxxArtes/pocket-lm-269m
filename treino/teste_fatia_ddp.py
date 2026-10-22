"""Cada processo do DDP le a sua fatia da epoca? Roda na CPU, sem GPU e sem dado real.

    python teste_fatia_ddp.py

Parte 1 (um processo): com mundo=1 a ordem e identica a do codigo anterior ao conserto, e
com 2, 3 e 4 processos as fatias sao disjuntas, do mesmo tamanho e cobrem a epoca menos
de `mundo` blocos.

Parte 2 (torchrun, 2 processos, gloo): chama o `get_batch` REAL, com RANK e WORLD_SIZE
vindos do torchrun, por varias epocas. Confere que a intersecao entre os processos e vazia
em cada epoca, que a uniao cobre a epoca e que 2 processos fecham a epoca na metade dos
passos de 1. Hashes diferentes nos primeiros lotes, sozinhos, nao provariam isso. Confere tambem que
`tempo_comum` devolve o mesmo valor nos dois processos com relogios locais diferentes, e
que a media de pesos retoma na CPU em fp32.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
AMBIENTE = {"BLOCK": "8", "BATCH": "4"}
N_TOKENS = 8 * 103 + 5          # 103 blocos menos a fase: nao divide por 2, 3 nem 4
EPOCAS = 3


def ordem_antiga(n_tokens, epoca, block):
    rng = np.random.default_rng(1337 + epoca)
    fase = int(rng.integers(0, block))
    n_blocos = max(1, (n_tokens - block - 1 - fase) // block + 1)
    return rng.permutation(n_blocos), fase


def parte_unidade():
    import train_gpu as tg
    for n in (N_TOKENS, 8 * 64 + 9, 8 * 7 + 3):
        for ep in range(4):
            antiga, fase_a = ordem_antiga(n, ep, tg.BLOCK)
            nova, fase_n = tg.ordem_da_epoca(n, ep)
            assert fase_a == fase_n and np.array_equal(antiga, nova), "mundo=1 mudou a ordem"
            for mundo in (2, 3, 4):
                fatias = [tg.ordem_da_epoca(n, ep, r, mundo) for r in range(mundo)]
                assert len({f for _, f in fatias}) == 1, "fase difere entre processos"
                tamanhos = {len(o) for o, _ in fatias}
                assert len(tamanhos) == 1, f"fatias desiguais {tamanhos}"
                uniao = np.concatenate([o for o, _ in fatias])
                assert len(set(uniao.tolist())) == len(uniao), "fatias se sobrepoem"
                assert set(uniao.tolist()) <= set(antiga.tolist())
                assert len(antiga) - len(uniao) < mundo, "fatia deixou blocos demais de fora"
    import torch
    from otim import MediaPesos
    m = torch.nn.Linear(3, 2)
    ema = MediaPesos(m, a_cada=1)
    est = ema.estado()
    # como viria de outro dispositivo/precisao, e de um run DDP antigo (prefixo module.)
    est["sombra"] = {"module." + k: v.half() for k, v in est["sombra"].items()}
    ema.carrega(est)
    assert all(v.dtype == torch.float32 and v.device.type == "cpu" for v in ema.sombra.values())
    assert set(ema.sombra) == set(m.state_dict())
    ema.atualiza(m, 0)
    ema.restaura(m, ema.aplica_em(m))
    print("parte 1 ok: mundo=1 identico ao anterior; fatias disjuntas e iguais; EMA retoma em CPU/fp32")


def coleta(saida):
    """Roda dentro de cada processo: le EPOCAS epocas pelo get_batch real."""
    import train_gpu as tg
    data = np.arange(N_TOKENS, dtype=np.int64)
    lidos = {}
    passos = {}
    while True:
        x, _ = tg.get_batch(data)
        ep = tg._sweep["epoca"]
        if ep >= EPOCAS:
            break
        inicios = x[:, 0].tolist()
        lidos.setdefault(ep, []).extend(inicios)
        passos[ep] = passos.get(ep, 0) + 1
    import time
    t0 = time.time()
    time.sleep(1.5 * tg.RANK)          # relogios locais divergem de proposito
    tempo = tg.tempo_comum(t0)
    with open(os.path.join(saida, f"rank{tg.RANK}.json"), "w") as f:
        json.dump({"mundo": tg.MUNDO, "lidos": lidos, "passos": passos, "tempo": tempo}, f)
    if tg.DDP_ON:
        tg.dist.barrier()
        tg.dist.destroy_process_group()


def roda_coleta(mundo, saida):
    env = dict(os.environ, **AMBIENTE)
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(k, None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    alvo = [os.path.abspath(__file__), "--coleta", saida]
    if mundo > 1:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(mundo),
               "--master_port", "29541"] + alvo
    else:
        cmd = [sys.executable] + alvo
    p = subprocess.run(cmd, cwd=DIR, env=env, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise SystemExit(f"coleta mundo={mundo} falhou rc={p.returncode}\n{p.stderr[-2000:]}")
    return [json.load(open(os.path.join(saida, f"rank{r}.json"))) for r in range(mundo)]


def parte_integracao():
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        um = roda_coleta(1, d1)[0]
        dois = roda_coleta(2, d2)
    assert all(r["mundo"] == 2 for r in dois), "torchrun nao chegou ao train_gpu"
    for ep in map(str, range(EPOCAS)):
        a, b = (set(r["lidos"][ep]) for r in dois)
        assert not a & b, f"epoca {ep}: {len(a & b)} blocos lidos pelos dois processos"
        assert len(a) == len(dois[0]["lidos"][ep]), f"epoca {ep}: bloco repetido no processo"
        ordem, fase = ordem_antiga(N_TOKENS, int(ep), int(AMBIENTE["BLOCK"]))
        epoca = set((ordem * int(AMBIENTE["BLOCK"]) + fase).tolist())
        # cada processo deixa de fora a cauda que nao fecha um lote, e o corte < mundo
        assert (a | b) <= epoca and len(epoca) - len(a | b) < 2 * int(AMBIENTE["BATCH"]) + 2, \
            f"epoca {ep}: uniao nao cobre a epoca"
        p1, p2 = um["passos"][ep], dois[0]["passos"][ep]
        assert p2 == dois[1]["passos"][ep], "processos viraram a epoca em passos diferentes"
        assert abs(2 * p2 - p1) <= 1, f"epoca {ep}: 1 processo {p1} passos, 2 processos {p2}"
    assert dois[0]["tempo"] == dois[1]["tempo"], f"tempo difere entre processos: {[r['tempo'] for r in dois]}"
    print(f"parte 2 ok: tempo comum entre processos; {EPOCAS} epocas, 2 processos disjuntos, metade dos passos")


if __name__ == "__main__":
    os.environ.update({k: v for k, v in AMBIENTE.items() if k not in os.environ})
    sys.path.insert(0, DIR)
    if len(sys.argv) > 2 and sys.argv[1] == "--coleta":
        coleta(sys.argv[2])
    else:
        parte_unidade()
        parte_integracao()
