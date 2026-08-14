# Sumário dos dados de treino — Pocket-LM 269M

Documento público de origem dos dados. Serve a três coisas ao mesmo tempo: seção do model
card, resposta ao campo "origem dos dados" de edital de fomento, e antecipação do dever de
publicar sumário dos conteúdos protegidos usados (art. 62 do PL 2338/2023, que prevê
vigência imediata quando virar lei — ver `licencas-e-risco-juridico.md`).

Atualizado em 31/07/2026, com a extensão `ext3` ainda em execução.

## Modelo

- **Pocket-LM 269M** — 269.228.032 parâmetros. Transformer decoder-only (20 camadas,
  n_embd 1024, 16 cabeças, contexto 512), treinado **do zero**, sem partir de pesos de
  terceiros em nenhum momento.
- **Tokenizador próprio**: BPE de 16.384 tokens treinado no nosso próprio corpus pt-BR.
  Não é tokenizador de terceiros.

## Fontes de pré-treino

| Fonte | Papel | Licença | Observação |
|---|---|---|---|
| **ClassiCC-PT** | ~60% do mix | **nenhuma declarada** no dataset card | corpus pt-BR filtrado por classificador, derivado de snapshots do Common Crawl (CC-2021-31/39, CC-2022-40), extração via Trafilatura. Os classificadores do projeto original foram treinados com anotação do GPT-4o |
| **Polygl0t/gigaverbo-v2-synth** | ~30% do mix da extensão | **Apache 2.0** | dado **sintético**, gerado pelos modelos Qwen2.5-32B/14B/7B-Instruct (todos Apache 2.0). Reponderado por nós: matemática e jurídico removidos, código reduzido a ~2% |
| **TucanoBR/wikipedia-PT** | ~10% do mix da extensão | CC BY-SA 3.0 | gramática normativa e texto longo coerente |

**Não usados, e por quê:** Corpus Carolina (contradição de licença não resolvida entre USP e
Hugging Face, com componente não-comercial); LegalPT e corpora de matemática (decisão
técnica — a capacidade de um 269M não os absorve).

## Volume

| Etapa | Tokens | Situação |
|---|---|---|
| Pré-treino base (`pocket-269m-classicc-v1`) | ~4,62 bilhões | concluído |
| Extensão de validação (`ext2`) | +0,75 bilhão | concluída |
| Extensão principal (`ext3`) | ~8,8 bilhões previstos | **em execução** |

Para comparação pública: o Tucano-160m usou 169 bilhões de tokens e o TeenyTinyLlama, 6,2
bilhões. O nosso diferencial declarado é eficiência de dados, não volume.

## Dados de conversa (fase de SFT, ainda não executada)

Diálogos **sintéticos**, gerados por modelos professores via API (OpenRouter, Together,
xAI, SiliconFlow, Groq) e filtrados por um juiz LLM calibrado contra julgamento humano do
autor (kappa 0,954, com itens sabotados como controle). Sementes de assunto vindas do
ClassiCC-PT. **Nenhuma conversa real de usuário é usada em treino** — decisão explícita do
projeto, tomada quando se constatou que conversas do próprio autor continham credenciais.

Corpus de sementes arquivado à parte: **Wikinotícias em português, 23.896 notícias**,
CC BY 2.5, com arquivo de atribuição.

## Compromissos

1. **O corpus não é redistribuído.** Publicamos a receita, os scripts e os ponteiros para
   as fontes originais — não cópias do texto.
2. **Pesos publicados sob Apache 2.0**, seguindo o precedente do campo (Tucano,
   TeenyTinyLlama, OLMo, Pythia).
3. **Canal de contato e procedimento de remoção** publicados junto com o modelo, para
   pedido de titular de dados pessoais.
4. Atribuição a Wikipedia e Wikinotícias no card — não exigida pela licença nessa
   modalidade de uso, feita como boa prática.
5. Pendência assumida: obter declaração escrita dos autores do ClassiCC-PT sobre a licença
   antes de qualquer uso comercial.
