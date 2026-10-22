# Licenças do corpus e risco jurídico de publicar os pesos — 31/07/2026

Atualizado em 22/10/2026 com o corpus de licença declarada, que entrou depois desta análise.

## Quadro por fonte (verificado no card e na origem, não presumido)

| Fonte | Licença real | Treinar? | Atribuição? | Share-alike nos pesos? |
|---|---|---|---|---|
| **ClassiCC-PT** | **NENHUMA declarada** — sem tag, sem campo `license`, sem LICENSE no repo. Origem: Common Crawl (CC-2021-31/39, CC-2022-40), extração Trafilatura | de fato sim (nada foi licenciado); risco herdado das páginas | não exigível | não |
| **Polygl0t/gigaverbo-v2-synth** | **Apache 2.0** (corpo do card). Gerado por **Qwen2.5-32B/14B/7B-Instruct**, todos Apache 2.0 | **sim** | não | não |
| **TucanoBR/wikipedia-PT** | CC BY-SA 3.0 | sim | só se redistribuir o TEXTO | **não** |
| **Wikinews pt** | CC BY 2.5 | sim | idem | não existe SA em BY 2.5 |
| **Corpus Carolina** | contradição real sem errata (USP: BY-NC-SA 4.0; HF: BY 4.0) e **ambas licenciam só o cabeçalho**, não os textos | só com triagem documento a documento | sim | irrelevante — trava é o **NC** |

**A dúvida do dado sintético está resolvida e é boa notícia:** o gigaverbo-synth foi gerado
por modelos **Qwen2.5, todos Apache 2.0** — não há cláusula de "no competing models" no
caminho. Era o risco mais provável de inviabilizar a rodada de extensão, e não existe.

**Carolina: NÃO usar** (já era a decisão; agora com motivo confirmado).

## Corpus de licença declarada (desde 04/08/2026)

Montado em 04/08/2026 para substituir o ClassiCC-PT. É a fase estável da `conversa-v1` e da
`executor-v1` e a fase final da `generica-v1`. A `generica-v1` continua pesos que já tinham
visto o ClassiCC-PT, e a troca de corpus não apaga essas fases anteriores.

Licenças conferidas em 22/10/2026 na API do Hugging Face e no card de cada repositório.
Documentos e duplicatas vêm do relatório da deduplicação; tokens de treino, dos shards do
corpus (16 bits por token).

| Fonte (`por_Latn`) | Licença | Documentos antes → depois da dedup | Tokens depois da dedup | Tokens de treino |
|---|---|---|---|---|
| finepdfs | ODC-By 1.0 | 625.540 → 476.278 | 3,68 bi | 3,61 bi |
| FineWeb2-HQ | ODC-By 1.0 | 6.192.836 → 6.116.293 | 4,56 bi | 4,50 bi |
| HPLT 2.0 cleaned | CC0 1.0 | 4.832.928 → 4.691.310 | 4,25 bi | 4,20 bi |
| **Total** | | **11.651.304 → 11.283.881** | **12,49 bi** | **12,31 bi** |

A diferença entre as duas últimas colunas é a validação: um shard de cada fonte (0,18 bi
tokens no total) foi movido para fora do prefixo de treino.

**Origem e condições de cada fonte, segundo o card:**

- **`HuggingFaceFW/finepdfs`:** PDFs de 105 snapshots do Common Crawl (2013 a 2025), parte
  buscada de novo na origem; o card cita sobretudo domínios jurídicos e educacionais. Uso
  sujeito também aos Termos de Uso do Common Crawl. E-mails e endereços IP anonimizados;
  formulário de remoção.
- **`epfml/FineWeb2-HQ`:** páginas HTML do Common Crawl, subconjunto filtrado por qualidade do
  FineWeb2. Uso sujeito também aos Termos de Uso do Common Crawl. E-mails e IPs anonimizados,
  com aviso de que ainda pode haver dado pessoal; formulário de remoção.
- **`HPLT/HPLT2.0_cleaned`:** sobretudo crawls do Internet Archive, com parte do Common Crawl.
  O CC0 cobre só o empacotamento ("we do not own any of the text"); cumprir a lei aplicável,
  inclusive a GDPR, fica por conta de quem usa; política de aviso e remoção.

**Deduplicação.** Exata, com uma tabela única para as três fontes: o hash dos primeiros 64
tokens de cada documento. Quando o documento aparece em mais de uma fonte, fica na primeira
da ordem acima. Saíram 23,86% dos documentos do finepdfs por repetição interna; no FineWeb2-HQ,
1,24% internos e 32 repetidos do finepdfs; no HPLT, 1,44% internos e 1,49% repetidos das duas
anteriores. Não é deduplicação aproximada: textos quase iguais que diferem no começo
continuam, e textos diferentes que começam igual (um cabeçalho institucional padrão, por
exemplo) saem como duplicata. A taxa desses descartes indevidos não foi medida.

**O que a licença declarada cobre e o que não cobre:**

- **Cobre a base, não cada texto.** O ODC-By 1.0 licencia a base de dados e diz, na §2.4, que
  não cobre nenhum direito sobre os conteúdos individuais. O card do HPLT diz o mesmo do CC0.
  Os textos continuam com os direitos dos seus autores. A análise de direito brasileiro abaixo
  vale também para este corpus: a LDA não tem exceção para mineração de texto e dados.
- **Atribuição.** A §4.3 do ODC-By exige, de quem usa publicamente um trabalho produzido com a
  base, um aviso de que o conteúdo veio dela. Se pesos de modelo contam como trabalho
  produzido é questão aberta; o aviso custa pouco, e os cards no Hugging Face trazem a seção
  "Atribuições de dados". Texto que atende a §4.3: "Contém informação de finepdfs e
  FineWeb2-HQ, disponibilizados sob a ODC Attribution License."
- **Termos do Common Crawl.** O finepdfs e o FineWeb2-HQ declaram que o uso também se sujeita a
  eles.
- **Dado pessoal.** Vale a mitigação da seção LGPD abaixo: não redistribuir o corpus e manter
  canal de remoção. O canal tira o dado das próximas rodadas de treino; pesos já publicados
  não esquecem o que viram.
- **Diversidade.** As três fontes vêm da web. O que muda é o formato e o pipeline: PDF
  extraído, HTML filtrado e crawl do Internet Archive. Versões anteriores das receitas
  chamavam o finepdfs de "fonte não-crawl", e isso estava errado.

## CC BY-SA não contamina os pesos

- **Texto da licença**: "Adapted Material" só existe quando a modificação exige permissão
  sob direito autoral; a cláusula de share-alike só dispara *"if You Share Adapted Material"*.
- **Posição oficial da Creative Commons** (FAQ de IA): quem faz mineração de texto e dados
  "não precisa aderir às exigências de marcação e atribuição para essa atividade", e o
  ShareAlike "só se aplica se a obra for modificada E compartilhada publicamente".
- **A OSI declara explicitamente** que "não toma posição sobre a natureza jurídica dos
  Parâmetros" — está em aberto, não decidido contra nós.
- **Precedente prático, unânime:** Tucano (Apache 2.0, com Wikipedia BY-SA e ainda material
  NC/NC-ND e Llama 2 no corpus), TeenyTinyLlama (Apache 2.0), OLMo/OLMo 2 (Apache 2.0),
  Pythia (Apache 2.0, The Pile). **Nenhum projeto aplicou CC BY-SA aos pesos.**

## Direito brasileiro

**Lei escrita:** a LDA (9.610/98) tem rol de exceções **taxativo** no art. 46, **sem**
exceção de mineração de texto e dados nem de cópia técnica. Não há hoje autorização legal
expressa para treinar com texto da web.

**O PL 2338/2023 NÃO virou lei.** Aprovado no Senado em 10/12/2024, está na Câmara desde
17/03/2025 e em 17/06/2026 seguia **"Aguardando Parecer"** — sem parecer, sem substitutivo,
sem votação. O texto do Senado traz: art. 62 (dever de publicar **sumário** dos conteúdos
protegidos usados, vigência imediata); art. 63 (exceção de TDM **só** para instituições
científicas/educacionais, acesso lícito e **sem fins comerciais**); art. 64 (opt-out do
titular); art. 65 (**dever de remunerar** titulares).

**Jurisprudência: não há caso concreto brasileiro** sobre treinar IA com conteúdo protegido.
Os dois casos próximos (voz de locutor no TJSP, música gerada por IA no PA) são sobre
**output**, não sobre treino. Não citar como precedente de treino.

## LGPD

Base legal usual é legítimo interesse (art. 7º, IX) com LIA documentada, ou o art. 4º, II,
"b" para fins exclusivamente acadêmicos por órgão de pesquisa. **Dado sensível (art. 11)
não admite legítimo interesse** — é o ponto fraco de corpus web irrestrito. Precedente
citado (ANPD suspendendo cautelarmente o uso de dados de brasileiros pela Meta em 2024)
**não pôde ser reaberto na varredura** — o site retornou conteúdo restrito. **Confirmar
antes de citar em edital.**

**Risco real de publicar os pesos: baixo.** O que a ANPD atacou foi coleta por controlador
com relação direta com titulares, não publicação de pesos; e um modelo de 269M memoriza
pouco. Mitigação suficiente: não redistribuir o corpus, publicar canal de contato e
procedimento de remoção, responder pedido de titular.

## Decisão: pesos em Apache 2.0

Motivos: é o padrão do campo e o precedente direto no nosso nicho; tem concessão e
retaliação de patente, que a MIT não tem; CC BY-SA nos pesos seria assumir voluntariamente
uma obrigação que a **própria Creative Commons diz não existir**, e mataria adoção; OpenRAIL
cria restrição de uso que reduz adoção sem reduzir o risco jurídico que de fato temos.

## Condições para publicar (checklist)

1. **Não redistribuir o corpus.** Publicar receita e ponteiros. É na redistribuição que o
   risco (NC, termos do Common Crawl, LGPD) se materializa.
2. **Publicar o sumário dos dados de treino** — antecipa o art. 62 do PL e é exatamente o
   que edital de fomento pergunta. Fazer agora, não no lançamento.
3. Descartar o Corpus Carolina.
4. **ClassiCC-PT sem licença declarada:** pedir declaração por escrito aos autores (Thales
   Sales Almeida, Rodrigo Nogueira, Helio Pedrini) antes de qualquer rodada comercial.
5. Registrar no card: os classificadores do ClassiCC-PT foram treinados com anotação do
   **GPT-4o**, e a cláusula de "no competing models" da OpenAI existe. O vínculo é indireto
   (filtro, não texto), mas registrar é diligência.
6. Atribuir Wikipedia/Wikinews no card: **não é exigido** por licença, é boa prática.
7. Manter nos cards o aviso de atribuição do ODC-By (§4.3) para finepdfs e FineWeb2-HQ.
