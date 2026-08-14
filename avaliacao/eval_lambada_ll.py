"""LAMBADA-PT por LOG-VEROSSIMILHANCA: separa comportamento de conhecimento.

Existe porque o v3g caiu 36,85 -> 31,71 no LAMBADA por GERACAO, mas o CALAME por NLL
quase nao mudou (+0,095 nat). Se o NLL da palavra de ouro do LAMBADA tambem ficar
proximo entre base e ajustado, a queda por geracao e comportamento aprendido no SFT
(terminador \n, formato de acao) e nao perda de lingua. Revisado pelo agy 05/08:
a media sobre TODOS os sub-tokens da palavra (nll_da_palavra) e o calculo certo.

  python3 eval_lambada_ll.py base.pt:tok.json ajustado.pt:tok.json [limite]
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_calame_ll import carrega, nll_da_palavra  # noqa: E402


def main() -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    base = "datasets/TucanoBR/lambada-pt@refs%2Fconvert%2Fparquet"
    linhas = []
    for cd in fs.ls(base, detail=False):
        nome = cd.rstrip("/").split("/")[-1]
        if nome == ".gitattributes":
            continue
        for sp in fs.ls(cd, detail=False):
            for a in fs.ls(sp, detail=False):
                if not a.endswith(".parquet"):
                    continue
                with fs.open(a, "rb") as fp:
                    linhas.extend(pq.ParquetFile(fp).read().to_pylist())
    lim = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 0
    if lim:
        linhas = linhas[:lim]
    print(json.dumps({"itens": len(linhas)}), flush=True)

    res = {}
    for arg in [a for a in sys.argv[1:] if ":" in a]:
        c, t = arg.rsplit(":", 1)
        m, cfg, tok = carrega(c, t)
        vals = []
        for k, r in enumerate(linhas):
            v = nll_da_palavra(m, cfg, tok, r["sentence"], str(r["last_word"]))
            if v is not None:
                vals.append(v)
            if (k + 1) % 1000 == 0:
                print(json.dumps({"m": c.split("/")[-2], "feitos": k + 1}), flush=True)
        res[c] = {"nll_lambada": round(sum(vals) / len(vals), 4), "n": len(vals)}
        print(json.dumps({c: res[c]}), flush=True)
        del m
    os.makedirs("avaliacao", exist_ok=True)
    json.dump(res, open("avaliacao/lambada_ll.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
