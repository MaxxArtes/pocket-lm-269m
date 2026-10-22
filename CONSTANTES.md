# Constantes medidas do projeto (a "física" para projetar sem chute)
Atualizado 14/08/2026. Toda linha tem origem em medição própria — citar a fonte ao usar.
Versão pública: cobre apenas o material do 269M.

## Velocidade e custo de treino
| o quê | valor | fonte |
|---|---|---|
| MFU 269M em H100 | 19% (108k tok/s) | corpus-limpo-v1 |
| escala DDP 4x H100 | ~4,0x em fichas **processadas**; em fichas **distintas**, 1x (os 4 processos liam o mesmo lote; o `train_gpu.py` publicado ainda não fatia) | corpus-limpo-v1; correção 21/08 |
| SFT 50k pares (fração 4090) | ~8,5 min, ~US$ 0,05 | adapters 14/08 |
| download R2->host: sequencial | ~1 MB/s (INACEITÁVEL) | Taiwan 14/08 |
| download R2->host: 24 fluxos | ~15x mais rápido | idem |

## Leis de receita (a régua do que repetir)
| o quê | valor | fonte |
|---|---|---|
| batch crítico | ≈ 22,9·D^0,47 → 1M tokens p/ D=8bi | arxiv 2412.01505, bate c/ verificação cruzada |
| LR por largura | LR ∝ 1/n_embd (referência: 6e-4 @ n_embd 1024) | regra muP + 3 runs estáveis |
| épocas de premium: aprendizado puro | ≤ ~2,7 | curva de dose |
| primeiros sinais de memorização | ~5 (5-shot cai) | curva de dose |
| repetição elevada | com 81-148 épocas a língua ficou pior, mas esses runs tiveram 7,0-7,6 bi de fichas contra 16,2 bi da base de comparação (efeito confundido) | conversa/executor-v1 |
| dose de conversa no SFT | 7% compra 50pp coerência por 0,4pp intenção; satura depois | curva de dose |
| replay 25% no SFT | CALAME 18→43, 5-shot 19→41, execução intacta | adapters 14/08 |
| SFT sem replay | destrói língua (CALAME 50→17) | cobertura 13/08 |
| SFT sequencial | destrói tarefa anterior (86→44) | cobertura 13/08 |

## Instrumentos (quanto confiar em cada número)
| o quê | valor | fonte |
|---|---|---|
| ruído do juiz cego (mesma entrada) | ±9 pp (excursão até 25) | 5 rodadas 13/08 |
| avaliar base sem SFT | SÓ com corte no próximo turno (3 conclusões erradas sem isso) | 13/08 |
| dedup por amostra | subestima 24x — sempre conjunto inteiro | corpus-limpo |
| distinct-2 | comparável só com mesmo n | 13/08 |

## Geração de dado (professores)
| o quê | valor | fonte |
|---|---|---|
| melhor ancoragem | DeepSeek-V4-Flash (2,00/2, $1/1M saída, MIT) | juiz 790 ex |
| melhor diversidade | Qwen2.5-7B local (distinct-2 0,718; 66% passa filtro) | idem |
| prompt 4 eixos + contínuos, t=1,15 | +12% distinct-2 E 100% ancoragem | varredura 13/08 |
| duas-etapas e mistura no mesmo formato | NÃO pagam (medido) | testa_desenhos |
| custo 10k exemplos (DeepSeek) | ~US$ 2-3 | premium 13/08 |
| GCP L4 spot p/ geração local | 2.161 tok/s, US$ 0,63/11,4M | bench 13/08 |

## Operação Vast
- fração de GPU (gpu_frac<1) NUNCA para treino; busca automática não filtra num_gpus → lançar por --oferta-id
- ~20% das instâncias travam no pull do Docker; >15 min em loading com status parado = destruir e relançar (inet_down anunciado NÃO prevê)
- budget de tempo (WSD!) deriva da velocidade MEDIDA — budget errado muda as épocas do premium
- matar treino por fora dispara o bloco "FIM" do job → apagar report/done do R2 antes do tick do vigia
- SSH: credenciais R2 vivem em /proc/1/environ; relançar com setsid+</dev/null
