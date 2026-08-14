# Desenho do SFT de Conversa — para revisão antes de treinar (06/08/2026)

**APROVADO pelo dono em 06/08/2026** (persona e recusas-leves como estão). Libera
geração de dados + treino em 07/08.

## Arquitetura: adapter LoRA sobre a base CONGELADA
r=16, alpha=16, all-linear (esteira do item 3, validada até o toggle no llama.cpp),
LR 3e-4, 2 épocas, SEM replay (esquecimento zero por construção — CALAME 53,13 exato
com adapter desligado é PORTÃO DE SANIDADE, não esperança). Treino na mesada da Modal.
O app carrega o adapter certo por modo: comando → adapter de ações; papo → este.

## Formato: texto puro, sem tocar no vocabulário
`usuário: <fala>\nassistente: <resposta>\n` — turnos concatenados como contexto,
resposta termina no terminador `\n` (mesma mecânica que consertou as ações).
Inferência: histórico + "assistente:" com stop=["\nusuário:", "\n\n"].

## Persona (decisão de produto — sua palavra final)
Brasileiro, direto, caloroso sem melação. 1-3 frases. **Proibido**: corporativês
("como posso ajudar?", "fico feliz em", "não hesite"), emoji, listas não pedidas.
**Obrigatório**: admitir "não sei" quando não sabe (10% dos dados exemplificam isso);
recusa curta e sem sermão para pedidos perigosos (medicina/direito/dano: "isso é
assunto pra profissional, não pra mim").

## Dados (REVISADO pós-agy: ~3.000 diálogos cirúrgicos, maioria 1-2 turnos)
Mistura: 45% cotidiano/ajuda prática · 20% papo social curto · 15% factual-com-
honestidade (com "não sei" correto em ~6% do TOTAL — dose reduzida: 269M acha o atalho da preguiça) · 10% sobre o próprio aparelho/assistente ·
10% recusas-leves. Persona embutida no prompt do professor COM exemplos negativos
("NUNCA responda assim: 'Claro! Fico feliz...'").

## Auditorias ANTES de treinar (reprovou → dado volta, não treina)
1. **Tom** (a lição do alerta externo): frequência de boilerplate por blocklist <0,5%
   dos turnos; mediana do turno do assistente ≤18 palavras; PMI do lado-assistente vs
   diálogo natural de referência sem assinatura corporativa no topo.
2. **Atalho** padrão: comprimento por classe, PMI, cópia de exemplo de prompt.
3. **Diversidade**: ≥3 professores; nenhum >50% do dado.

## Portões pré-registrados do MODELO (medidos em 100 prompts virgens)
- Juiz cego (modelo DIFERENTE dos professores): "coerente + no personagem" ≥70%;
- Formato: 100% dos turnos terminam no terminador; zero corridas >60 tokens;
- "Não sei" honesto: nos 20 prompts-armadilha (fatos obscuros), inventar fato em >30%
  reprova;
- Sanidade da base: CALAME acc == 53,13 com adapter off (igualdade exata);
- Ações intactas: suíte de ações com o adapter de AÇÕES carregado, inalterada.

## Camada do app (alerta externo 06/08, resolvido por arquitetura)
Troca de modo NÃO recarrega adapter: os dois ficam residentes desde a inicialização
e a troca é só escala 0.0/1.0 (mecânica do toggle validada no item 3 via
/lora-adapters). Custo: +42 MB de RAM (2×21 MB) e matmuls extras do LoRA não-fundido
(~5-10% de tok/s com adapter ativo — medir no Device Farm na rodada do app).
Fundir e trocar GGUF inteiro está descartado (2×177 MB + centenas de ms por troca).

## Custos: professor ~US$ 8-12 · treino Modal US$ 0 · avaliações ~US$ 1

## Revisão adversarial (agy 06/08) — correções ACEITAS antes de treinar
1. **Dose cortada: 10-12k → ~3.000 diálogos** (r16 em 269M: mais que isso decora bordão
   e degenera léxico). Custo de professor cai pra ~US$ 3-4.
2. **"Não sei" reduzido a ~6%** + portão de falso-positivo: 30 perguntas ultra-básicas,
   **0% de "não sei" tolerado** — colapso de preguiça reprova o treino inteiro.
3. **Blindagem do \n**: entrada do usuário passa por replace('\n',' ') no app ANTES do
   template (regra de cliente, documentada); persona proíbe resposta multi-parágrafo,
   então o terminador \n continua válido no lado do assistente; stop de geração =
   "\nusuário:" além do \n. Portão novo: 10 prompts com \n injetado — qualquer
   alucinação de tag reprova.
4. **Juiz cego desmembrado e endurecido**: coerência+formato ≥95%; personagem ≥85%
   (o composto de 70% mascarava colapso de utilidade).
5. **Distinct-2 ≥ 0,70** nas 100 respostas (anti-bordão).
6. **PPL de linguagem**: adapter LIGADO em 100 frases neutras ≤ +20% vs base pura.
