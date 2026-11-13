# Algoritmo 1 — sumarização das atas

Implementação do Algoritmo 1 e do experimento de determinismo que o antecede,
especificados em `ic_exploracao/latex/relatorio_ic_parcial.tex`, que é a fonte
normativa: o algoritmo "Sumarização das atas" e o procedimento
`TesteDeterminismo`, com as funções que eles chamam. Este arquivo registra como
executar e quais decisões de implementação foram tomadas, com a evidência de
cada uma. Os resultados da execução estão no mesmo relatório, na seção de
resultados do Algoritmo 1.

O trabalho tem duas partes:

| Parte | O que faz | No relatório | Estado |
|---|---|---|---|
| Experimento de determinismo, anterior ao algoritmo | 10 atas × 10 submissões × 2 idiomas, com o estado do servidor limpo antes de cada chamada. Fixa `N_sub`. | procedimento `TesteDeterminismo` | concluído em 19/10/2026: variância nula, **`N_sub = 1`** |
| Algoritmo 1: geração do corpus | Gera o corpus com o `N_sub` fixado pelo experimento e relata a lista `Acima` | algoritmo "Sumarização das atas" | concluído em 20/10/2026: 404 resumos, sem falhas |

A análise de variância sobre o corpus saiu do algoritmo: com `N_sub = 1` não
há pares de submissões a comparar. `analisar_variancia.py --fonte corpus`
ainda a faria se `N_sub > 1`.

### Nomes antigos que ainda aparecem no código

Os comentários do código usam a nomenclatura do relatório. Os nomes de funções
e constantes, as mensagens impressas, os textos de ajuda (`--help`) e os logs
ainda usam a da especificação anterior, que tinha as etapas A, B e C:

| Nome antigo | Nome no relatório |
|---|---|
| etapa A | experimento de determinismo, procedimento `TesteDeterminismo` |
| etapa B | Algoritmo 1, geração do corpus |
| etapa C | removida: a análise de variância perdeu o objeto com `N_sub = 1` |
| S0 | decisão de `N_sub` (`determinismo_frio_relatorio.json`) |
| S1 | o corpus, tabela `Resumo` |
| S2, `Over[lang]` | lista `Acima[lang]` |
| sub-rotina A | laço sobre os pares `(a, b)` de submissões |
| sub-rotinas B, C, D | `SimilaridadeLexical`, `SimilaridadeRougeL`, `SimilaridadeEmbedding` |
| sub-rotina E (`CLEAR_STATE`) | `LimparEstado` (`descarregar_modelo` no código) |
| sub-rotina F | `AmostraPorComprimento` |
| sub-rotina G | `RelatarSimilaridades` |
| linha 23 | passo 12 de `TesteDeterminismo` (decisão de `N_sub`) |
| E4 do Algoritmo 1 | entrada `LLM` do Algoritmo 1 |
| E5 do Algoritmo 2 | função `Ingerir` do Algoritmo 2 |

---

## Como executar

```bash
cd ic_exploracao/algoritmo1

# Experimento de determinismo -- ja executado; so precisa ser refeito se modelo,
# Ollama ou prompt mudarem
python determinismo.py --selecionar      # fixa a amostra (AmostraPorComprimento)
./rodar_determinismo_frio.sh             # as 200 chamadas, ~8 h
python determinismo.py --verificar-frio  # veredito e decisao de N_sub

# Algoritmo 1
./rodar_geracao.sh                       # gera o corpus com o N_sub fixado pelo experimento
python status.py                         # quanto ja foi feito e quanto falta
python analisar_variancia.py --fonte corpus   # lista Acima do corpus
```

A geração do corpus não escolhe `N_sub`. `gerar_resumos.py` lê o relatório da
segunda execução do experimento de determinismo
(`output/algoritmo1/determinismo_frio_relatorio.json`) e **aborta** se ele
estiver ausente, incompleto, se não for da condição com estado limpo, ou se a
decisão registrada nele divergir de `config_alg1.N_SUB`. Passar `--n-sub` com
valor diferente da decisão também aborta.

Volume da geração: 404 resumos (202 atas × 2 idiomas), concluídos em
20/10/2026, em 19,7 h de processamento. Resultado na seção "Resultado da
geração do corpus", abaixo.

### Execução desacoplada

`rodar_geracao.sh` e `rodar_determinismo_frio.sh` desacoplam o processo do
terminal (`PPID = 1`, sobrevive ao fechamento do editor) e o embrulham em
`caffeinate -is`, porque a máquina está configurada com `pmset sleep 1`. A tela
pode apagar; **a tampa não pode ser fechada**, porque isso suspende o sistema
mesmo com `caffeinate`. Os dois scripts se recusam a rodar ao mesmo tempo:
limpar o estado do servidor numa execução corromperia a outra.

### Retomada

Os dois scripts são idempotentes. Rodar de novo retoma de onde parou: o script
lê o seu JSONL, monta o conjunto das chaves `(data, j)` já gravadas e gera
apenas o que falta. Cada resumo é gravado com `flush` e `fsync` imediatos,
então o prejuízo máximo de qualquer interrupção é **um** resumo, o que estava em
andamento.

Se a interrupção ocorrer no meio de uma escrita, a última linha do JSONL fica
truncada. Geração e análise detectam isso, descartam a linha e seguem; o par
`(data, j)` correspondente é regerado.

Na partida, a geração do corpus descarrega o modelo do servidor (passo 1 do
Algoritmo 1). Sem isso, uma ata interrompida no meio do processamento deixaria
parte do próprio prompt no cache, e a retomada daquela ata seguiria outro
caminho de execução (ver "Estado de execução na geração do corpus", abaixo).

### Interromper de propósito

```bash
kill $(cat ../output/algoritmo1/logs/geracao.pid)              # geração do corpus
kill $(cat ../output/algoritmo1/logs/determinismo_frio.pid)    # experimento de determinismo
```

---

## Arquivos

| Arquivo | Função |
|---|---|
| `config_alg1.py` | constantes do Algoritmo 1 e do experimento de determinismo, cada uma anotada com a entrada do relatório que a define; caminhos dos arquivos |
| `determinismo.py` | experimento de determinismo: amostra (`AmostraPorComprimento`), as chamadas com estado limpo, veredito e decisão de `N_sub`. Também guarda os modos da primeira execução do teste (`--verificar`, `--diagnosticar`) e a verificação de versão (`--verificar-versao`) |
| `gerar_resumos.py` | Algoritmo 1: geração, medição em tokens BERT, lista `Acima`; `LimparEstado` (`descarregar_modelo`); leitura do `N_sub` fixado pelo experimento |
| `analisar_variancia.py` | `SimilaridadeLexical`, `SimilaridadeRougeL`, `SimilaridadeEmbedding` e `RelatarSimilaridades` sobre uma fonte: `frio` (segunda execução do experimento), `padrao` (primeira execução) ou `corpus` (lista `Acima` do corpus e, se `N_sub > 1`, as similaridades entre submissões) |
| `verificar_num_ctx.py` | verificação que autorizou o `num_ctx` adaptativo |
| `status.py` | progresso e tempo restante da geração do corpus, lido do checkpoint |
| `rodar_geracao.sh` | disparo desacoplado da geração do corpus |
| `rodar_determinismo_frio.sh` | disparo desacoplado do experimento de determinismo |

`config.py` e `utils.py`, da fase exploratória, não fazem parte do repositório
público. **Nada aqui depende deles.**

---

## Saídas

Cada condição experimental tem arquivo próprio. O corpus do estudo não contém
nenhum resumo do teste de determinismo, e vice-versa.

| Saída (nome no relatório) | Caminho | Conteúdo |
|---|---|---|
| decisão de `N_sub` | `output/algoritmo1/determinismo_frio_relatorio.json` | veredito por ata, `todas_identicas`, `n_sub_decidido` |
| `Resumo`, o corpus | `resumos/resumos_<lang>.jsonl` | um JSON por resumo, chave `(lang, data, j)`, só `j = 1` |
| `Acima[lang]` | `output/algoritmo1/over_l_max_corpus_<lang>.csv` | resumos do corpus acima de `L_max`, com o excesso |
| `Sim` | `output/algoritmo1/similaridades_frio_<lang>.parquet` | segunda execução do experimento: um registro por `(métrica, ata, a, b)` |
| estatísticas de `Sim` | `output/algoritmo1/estatisticas_frio_{global,por_ata,por_ano,por_periodo}.csv` | segunda execução do experimento: agregações (i) a (iv) |
| — | `resumos/determinismo_frio_<lang>.jsonl` | os 200 resumos da segunda execução do experimento |
| — | `resumos/determinismo_padrao_<lang>.jsonl` | os 200 resumos da primeira execução do teste, sem limpeza de estado |
| — | `output/algoritmo1/*_padrao_*` | similaridades e agregações da primeira execução |
| — | `output/algoritmo1/determinismo_diagnostico.json` | a replicação controlada que isolou a causa da variação |
| — | `output/algoritmo1/registro_execucao_*.jsonl` | ambiente, parâmetros e tempos de cada execução |

Até 19/10/2026 o corpus e a primeira execução do teste compartilhavam
`resumos_<lang>.jsonl`. Eles foram separados: o arquivo do corpus ficou só com
`j = 1`, e as 200 submissões da primeira execução foram para
`determinismo_padrao_<lang>.jsonl`.

### Contrato do corpus (tabela `Resumo`) — é a entrada da função `Ingerir` do Algoritmo 2

Um objeto JSON por linha, com estes campos:

| Campo | Tipo | Observação |
|---|---|---|
| `lang` | `str` | `pt` ou `en` |
| `data` | `str` | `YYYYMMDD`, identifica a ata; junta com `dataset_<lang>.csv` |
| `j` | `int` | submissão, 1 a `N_sub`; com `N_sub = 1`, sempre 1 |
| `seed` | `int` | `S0 + P*j`; sem efeito na saída com `temperature = 0` |
| `resumo` | `str` | **texto literal do modelo**, sem pós-processamento |
| `n_tokens_bert` | `int` | medido com o tokenizador do idioma (BERTimbau, `bert-base-portuguese-cased`, em pt; BERT, `bert-base-cased`, em en), incluindo `[CLS]` e `[SEP]` |
| `excede_l_max` | `bool` | `n_tokens_bert > 512` |
| `excesso` | `int` | `max(0, n_tokens_bert - 512)` |
| `num_ctx` | `int` | patamar usado nesta ata |
| `prompt_eval_count` | `int` | tokens de prompt processados pelo servidor |
| `eval_count` | `int` | tokens gerados |
| `done_reason` | `str` | `stop` indica que o modelo encerrou sozinho, não por `num_predict` |
| `truncamento_suspeito` | `bool` | sempre `false` em execução sadia |
| `tempo_s`, `n_chars_ata`, `n_chars_resumo` | | medições auxiliares |
| `llm_model`, `llm_digest`, `ollama_version`, `prompt_sha256` | `str` | procedência do resumo |
| `gerado_em` | `str` | ISO 8601, UTC |

O campo `resumo` é o texto como saiu do modelo. Na ingestão, a função `Ingerir`
do Algoritmo 2 remove o preâmbulo dos resumos em inglês, tokeniza e corta o
final até 512 tokens. A conformidade com os 512 tokens **não** é imposta aqui,
de propósito (passo 11 do Algoritmo 1).

---

## Experimento de determinismo

Determinado pelo orientador, com critério de decisão definido antes da execução:
dez atas, dez submissões cada, `temperature = 0`; se todos os resumos de uma
mesma ata saírem iguais, a repetição não é informativa e passa-se a uma
submissão por ata. O critério é o passo 12 do procedimento `TesteDeterminismo`.
O relato completo está no relatório, com os textos divergentes lado a lado em
apêndice.

Amostra fixada por regra, antes de observar qualquer resumo
(`AmostraPorComprimento`): as 202 atas ordenadas por comprimento do texto,
tomadas em dez posições igualmente espaçadas, extremos incluídos. As mesmas dez
reuniões nos dois idiomas. Cobre atas de 5.807 a 67.477 caracteres e os três
patamares de `num_ctx` em português, dois em inglês.

### Primeira execução, sem limpeza de estado

As dez submissões de cada ata em sequência, com o modelo residente.

| | combinações idênticas | combinações que variaram |
|---|---|---|
| PT | 9 de 10 | 1 (20230920) |
| EN | 8 de 10 | 2 (20070606, 20110720) |

Nas três que variaram, o padrão foi sempre `j = 1` sozinho contra `j = 2..10`
byte-idênticos entre si — apesar de nove sementes distintas, o que já excluía a
semente. `python determinismo.py --diagnosticar` refez `j = 1` com a **mesma**
semente, variando só o estado do cache:

| | chamada fria | chamada quente |
|---|---|---|
| pt 20230920 | reproduziu `j = 1` | reproduziu `j = 2..10` |
| en 20070606 | reproduziu `j = 1` | reproduziu `j = 2..10` |
| en 20110720 | reproduziu `j = 1` | reproduziu `j = 2..10` |

A primeira submissão de cada ata processava o texto do zero; as nove seguintes
reaproveitavam o prompt já no cache. Somas de ponto flutuante não são
associativas, e a ordem delas muda entre os dois caminhos; quando dois
candidatos a próximo token estão quase empatados, isso inverte o `argmax` e o
texto diverge dali em diante.

Medida assim, a variância seria da assimetria do servidor, não do sumarizador.
Por isso a limpeza de estado (`LimparEstado`, passo 6 de `TesteDeterminismo`)
entrou no teste.

### Segunda execução, como especificada, com estado limpo antes de cada chamada

| | combinações idênticas | combinações que variaram |
|---|---|---|
| PT | 10 de 10 | nenhuma |
| EN | 10 de 10 | nenhuma |

```
metrica    lang   n    minimo        mediana  maximo  IQR      desvio
embedding  en    450  0.9999999579    1.0     1.0    5.2e-08  3.4e-08
embedding  pt    450  0.9999999305    1.0     1.0    7.4e-08  4.5e-08
lexical    en    450  1.0000000000    1.0     1.0    0.0      0.0
lexical    pt    450  1.0000000000    1.0     1.0    0.0      0.0
rougeL     en    450  1.0000000000    1.0     1.0    0.0      0.0
rougeL     pt    450  1.0000000000    1.0     1.0    0.0      0.0
```

Variância nula. O cosseno difere de 1 só na sétima casa, por arredondamento de
`float32`. Decisão (passo 12 de `TesteDeterminismo`): **`N_sub = 1`**.

O resumo em estado limpo é idêntico ao `j = 1` da primeira execução nas 20
combinações. É isso que dispensa limpar o estado na geração do corpus (ver
abaixo).

Custo: 7,9 h (5,1 h PT, 2,7 h EN), contra 2,5 h da mesma amostra sem limpeza —
cerca de 3,2 vezes, porque cada chamada recarrega o modelo e processa o prompt
inteiro.

A alternativa de medir variabilidade real com `temperature > 0` foi descartada
pelo orientador: aumentaria o risco de o modelo acrescentar informação ausente
da ata.

### Estado de execução na geração do corpus

Com `N_sub = 1`, cada ata é submetida uma única vez, e essa chamada encontra no
cache o prompt de outra ata, nunca o da própria — o caminho dos `j = 1`, que o
experimento mostrou coincidir com o estado limpo. Limpar o estado antes de cada
uma das 404 chamadas produziria os mesmos resumos a 3,2 vezes o custo.

Para que isso valha também depois de falhas, `gerar_resumos.py` descarrega o
modelo na partida e antes de cada retentativa. Se o experimento tivesse fixado
`N_sub > 1`, o Algoritmo 1 limparia o estado antes de toda chamada (passo 6).

Limitação: a equivalência foi verificada nas 20 combinações da amostra, não nas
202 atas de cada idioma, e depende do servidor, da versão do Ollama e do modelo.
Ela foi demonstrada na 0.35.1 e confirmada na 0.40.5 — ver "Versões do
servidor", abaixo.

### Efeito sobre a lista `Acima`

A variante que reaproveitava o cache em en 20110720 tinha 584 tokens BERT; a do
estado limpo tem 359. Por isso a primeira execução contava 3 das 10 atas
inglesas acima de 512 tokens, e a segunda conta 2. O número que vale para o
estudo é o do corpus, na seção seguinte.

---

## Resultado da geração do corpus

Concluída em 20/10/2026. Integridade conferida sobre o corpus completo: 202 atas
por idioma, nenhuma faltando ou duplicada, só `j = 1`, prompt e modelo iguais em
todas, nenhuma truncada na entrada, nenhuma falha.

### Lista `Acima` — resumos acima de 512 tokens

| | acima de 512 | proporção | excesso mediano | excesso máximo |
|---|---|---|---|---|
| PT | 14 de 202 | 6,9% | 62 tokens | 296 |
| EN | 37 de 202 | 18,3% | 66 tokens | 400 |

O inglês tem cerca de 2,6 vezes mais resumos cortados na ingestão pelo
Algoritmo 2. Arquivos: `output/algoritmo1/over_l_max_corpus_{pt,en,resumo}.csv`.

**O que o corte do final remove** (função `Ingerir` do Algoritmo 2): na mediana,
11% do resumo (máximo de 37% em PT e 44% em EN). O trecho removido contém termos
da decisão de política monetária (Selic, decisão, manter, elevar, reduzir) em 4
dos 14 resumos PT e 20 dos 37 EN, mas, em todos os 51 casos, esses termos também
aparecem na parte que fica. A alternativa head+tail do baseline removeria o meio,
que contém esses termos em 6 de 14 (PT) e 31 de 37 (EN). A medição é por palavras-
chave: indica onde a decisão é mencionada, não garante que o trecho removido seja
redundante.

### Duas saídas degeneradas

Em pt 20030423 (808 tokens) e en 20110608 (912 tokens), o modelo passou a
reproduzir trechos da ata em vez de resumi-la e parou só no limite de
`num_predict = 900` (`done_reason = "length"`). No caso PT a cópia é visível:
o fim do resumo está sem acentos, como o texto de entrada. Ficam como saíram,
pelo Algoritmo 1, e são cortados na ingestão.

### Preâmbulo nos resumos em inglês

55 dos 202 resumos EN começam por uma frase fixa, em uma de duas formas:
"Here is a summary of the document in 300 words or less:" (curta, 37 vezes, 13
tokens BERT) ou "Here is a summary of the document in 300 words or less,
preserving the topics discussed and the assessments presented:" (longa, 18
vezes, 22 tokens BERT). As duas coincidem até "less": a curta termina ali, com
dois-pontos, e a longa continua com vírgula. Nenhum resumo PT tem preâmbulo, e
nenhum dos 37 resumos EN acima do limite começa com ele. O corpus mantém o texto
literal; o preâmbulo é removido na ingestão pelo Algoritmo 2 (passo 1 da função
`Ingerir`), por decisão do orientador.

---

## Decisões de implementação

Cada decisão abaixo tem evidência medida. Nenhuma altera prompt, sementes ou
modelo.

### 1. `num_ctx` corrigido de 16384 para 32768

Correção de erro factual no valor da especificação original, aprovada pelo
orientador e registrada no relatório (seção sobre a correção do `num_ctx`). O
Ollama não recusa prompt maior que `num_ctx`: processa `num_ctx / 2` e descarta o
resto, sem aviso. Com 16384, 49 das 202 atas portuguesas entrariam pela metade e
nenhuma das inglesas — o corpus PT chegaria mutilado e o EN não, confundindo a
comparação entre idiomas.

### 2. `num_ctx` adaptativo por ata — adotado após verificação

Em vez do valor único da entrada `LLM` do Algoritmo 1, cada ata usa o menor
patamar de `{8192, 16384, 32768}` que a acomode sem truncar. Distribuição: PT
70/65/67 atas, EN 102/100/0.

A otimização só é legítima se não alterar o resumo, senão o comprimento da ata
ficaria confundido com o patamar usado. Isso foi medido, não suposto, por
`verificar_num_ctx.py`, na mesma ata com a mesma semente:

| `num_ctx` | PT — SHA-256 do resumo | PT — tempo | EN — tempo |
|---|---|---|---|
| 8.192 | `8853f7654aee4813` | 27,1 s | 44,7 s |
| 16.384 | `8853f7654aee4813` | 30,4 s | 45,4 s |
| 32.768 | `8853f7654aee4813` | 44,0 s | 51,9 s |

Resumo byte-idêntico nos três patamares, nos dois idiomas, com tempo crescente
em `num_ctx`. Medições em `output/algoritmo1/verificacao_num_ctx_{pt,en}.json`.
O experimento de determinismo reforça a verificação: sua amostra cobre os três
patamares e não houve variação em nenhum.

Para reverter ao valor único: `./rodar_geracao.sh --num-ctx-modo fixo`.

Duas salvaguardas acompanham a otimização. A estimativa de tokens do prompt é
pessimista de propósito — o Ollama não expõe endpoint de tokenização
(`/api/tokenize` responde 404) e o tokenizador do Llama 3.1 não é redistribuível,
então o divisor usado é menor que a razão caracteres/token medida. E a ausência
de truncamento não depende dessa estimativa: é conferida depois, em
`prompt_eval_count`, uma vez por ata; se houver truncamento o patamar sobe
sozinho e o resumo truncado é descartado.

As atas são processadas agrupadas por patamar, porque trocar `num_ctx` obriga o
Ollama a recarregar o modelo. A ordem não afeta resultado algum: cada ata é
independente e a saída é chaveada por `(lang, data, j)`.

### 3. Dobramento para ASCII em `SimilaridadeLexical` e `SimilaridadeRougeL`

As atas de entrada são ASCII puro, mas **os resumos têm acento** — o modelo
escreve "política", "deprecição". O tokenizador padrão do pacote `rouge-score`
descarta tudo fora de `[a-z0-9]`, o que fragmentaria "política" em "pol" + "tica"
de forma arbitrária. Por isso `SimilaridadeLexical` (Jaccard) e
`SimilaridadeRougeL` (ROUGE-L) recebem o texto normalizado por NFKD e dobrado
para ASCII.

A transformação é aplicada identicamente aos dois membros de todo par, logo não
pode enviesar uma comparação par a par, e não altera o corpus: o resumo é
gravado e entregue literal. `SimilaridadeEmbedding` recebe o resumo literal,
porque o codificador de sentenças é multilíngue e trata acentuação nativamente.

Consequência: a implementação de ROUGE-L é local, para poder receber o texto já
dobrado. `analisar_variancia.py` confere a implementação local contra o pacote
`rouge-score` antes de rodar e **aborta** se divergirem — a conferência atual
acusa desvio máximo de 0,0 em quatro casos de teste.

### 4. `N_sub` fixado pelo experimento de determinismo

Na primeira versão do algoritmo, `N_sub = 10` era fixo. Agora é entrada do
Algoritmo 1, fixada pelo experimento de determinismo, e a geração a lê do
relatório do experimento. Em `config_alg1.py`: `N_SUB = 1` (o valor fixado,
conferido contra o relatório), `N_SUB_MAX = 10` (`N_sub_max`, o valor adotado se
houver variação), `N_SUB_DETERMINISMO = 10` (`N_test`) e
`N_ATAS_DETERMINISMO = 10` (`K`).

A nota do Algoritmo 1 sobre o espaçamento das sementes registra a correção:
fixar a semente não torna a submissão reproduzível, porque com
`temperature = 0` ela não é lida.

---

## Custo medido

| | chamadas | tempo |
|---|---|---|
| Experimento de determinismo, primeira execução (sem limpeza de estado) | 200 | 2,5 h |
| Experimento de determinismo, segunda execução (limpeza antes de cada chamada) | 200 | 7,9 h |
| Algoritmo 1 (corpus), `N_sub = 1` | 404 | 19,7 h (PT 10,3 h, EN 9,4 h) |

Modelo de tempo da geração do corpus, ajustado sobre 173 resumos reais (cada um
a primeira e única chamada da sua ata): `t ≈ 9,5 + 0,01462 · n_tokens` de
prompt. Mediana por resumo: PT 144 s, EN 157 s. A estimativa inicial
(`25,1 + 0,00285 · n_tokens`, ~5,7 h no total) vinha de submissões repetidas da
mesma ata, que reaproveitam o cache, e subestimou o tempo em cerca de 3,5 vezes.

---

## Ambiente

| Componente | Versão |
|---|---|
| Ollama | **0.40.5** desde 16/10/2026 (antes 0.34.2 e 0.35.1; ver abaixo) |
| LLM | `llama3.1:8b` — digest `46e0c10c039e` — Q4_K_M |
| Python | 3.10.0, em `ic_exploracao/.venv` |
| transformers / torch | 4.44.2 / 2.4.1 |
| sentence-transformers | 3.0.1 |
| embeddings | `paraphrase-multilingual-MiniLM-L12-v2`, salvo em `model/pre-trained/` |

### Versões do servidor

O app do Ollama se atualiza sozinho, e isso aconteceu durante o estudo. Versões
efetivamente usadas, verificadas pelo campo `ollama_version` de cada resumo e
pelo horário de partida do processo do servidor:

| Fase | Data | Versão |
|---|---|---|
| Piloto e `verificar_num_ctx.py` | 05/10 | 0.34.2 |
| Primeira execução do teste e diagnóstico | 12/10 | 0.35.1 |
| Segunda execução do experimento de determinismo | 19/10 | 0.40.5 |
| Algoritmo 1 (corpus) | 05/10 a 20/10 | mistura, ver abaixo |

| Corpus | 0.34.2 | 0.35.1 | 0.40.5 |
|---|---|---|---|
| PT | 14 | 8 | demais |
| EN | — | 10 | demais |

O modelo é o mesmo em todos os resumos. A segunda execução do experimento não
gravava a versão do servidor nos registros; a 0.40.5 foi estabelecida porque o
processo do servidor em 0.40.5 partiu em 16/10 às 12h27 e seguiu no ar, sem
reinício, durante toda a execução.

Evidência de que a troca não alterou resultados: os 20 resumos da amostra
gerados em 0.34.2 e 0.35.1 são idênticos aos da segunda execução do experimento,
em 0.40.5. Os outros 12 resumos antigos do corpus (todos PT) não têm essa
contraprova.

**Verificado na 0.40.5 (20/10/2026)** com `python determinismo.py
--verificar-versao`: nas três atas sensíveis, chamadas logo depois da ata que as
precedeu na geração, a primeira chamada reproduziu o estado limpo da segunda
execução do experimento, e uma segunda chamada reproduziu a variante quente já
vista na 0.35.1. O corpus seguiu o caminho equivalente ao estado limpo, e o
efeito do cache é o mesmo nas duas versões. Resultado em
`output/algoritmo1/verificacao_versao_0.40.5.json`.

Salvaguardas desde 20/10: `gerar_resumos.py` confere a versão do servidor antes
de cada chamada e para a geração se ela mudar; `determinismo.py` grava versão e
digest em cada registro e aborta se a versão mudar no meio do experimento.
**Desative a atualização automática do app do Ollama antes de qualquer nova
geração.**

`gerar_resumos.py` confere o SHA-256 dos dois prompts na partida e **aborta** se
algum tiver sido alterado, porque o prompt é constante do estudo. Valores
esperados:

```
pt  4bf4c28185d6bfeb4ab271017b9b4be51c58fa468f19df777a4c7e55793b1868
en  33d62246f885654e4213dfde159c53f37025bf9b846517f63f4b0beb43ca248c
```
