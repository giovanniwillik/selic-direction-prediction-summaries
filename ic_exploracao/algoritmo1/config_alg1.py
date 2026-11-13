"""
Constantes normativas do Algoritmo 1 (sumarizacao) e do experimento de
determinismo que o antecede.

Fonte: ic_exploracao/latex/relatorio_ic_parcial.tex, entradas do Algoritmo 1
(sumarizacao das atas) e do procedimento TesteDeterminismo. Cada constante
abaixo esta anotada com a entrada do relatorio que a define. Nomes de
constantes e de funcoes do codigo (por exemplo RELATORIO_ETAPA_A) mantem a
nomenclatura da versao anterior da especificacao, com as etapas A, B e C.

Este modulo e deliberadamente separado de ic_exploracao/config.py, que foi
escrito na fase exploratoria e contem parametros desatualizados (num_predict
= 1024, lista de seeds ad hoc, prompts antigos). Nada daqui deve ser lido de la.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Diretorios
# ---------------------------------------------------------------------------
ALG1_DIR = Path(__file__).resolve().parent
IC_DIR = ALG1_DIR.parent
PROJECT_ROOT = IC_DIR.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
MODELOS_DIR = PROJECT_ROOT / "model" / "pre-trained"

RESUMOS_DIR = IC_DIR / "resumos"          # corpus (tabela Resumo), checkpoint
OUTPUT_DIR = IC_DIR / "output" / "algoritmo1"   # Acima, Sim, estatisticas
LOG_DIR = OUTPUT_DIR / "logs"

# ---------------------------------------------------------------------------
# Idiomas
# ---------------------------------------------------------------------------
LANGUAGES = ("pt", "en")

# ---------------------------------------------------------------------------
# D_pt, D_en -- atas de entrada (corpus de Forti, 2025)
# ---------------------------------------------------------------------------
DATASET_CSV = {
    "pt": DATASET_DIR / "dataset_pt.csv",
    "en": DATASET_DIR / "dataset_en.csv",
}
COLUNA_TEXTO = "texto_ata"
COLUNA_DATA = "data"          # inteiro no formato YYYYMMDD
N_ATAS_ESPERADO = 202

# ---------------------------------------------------------------------------
# LLM -- modelo sumarizador, congelado para todo o estudo
# ---------------------------------------------------------------------------
LLM_MODEL = "llama3.1:8b"
LLM_DIGEST_ESPERADO = "46e0c10c039e"

# num_ctx = 32768 e o valor corrigido; ver a secao "Correcao do num_ctx" do
# relatorio.
# O valor anterior (16384) truncava 49 das 202 atas em portugues sem aviso.
NUM_CTX = 32768

# Patamares admitidos para a otimizacao de num_ctx adaptativo. O patamar so e
# rebaixado quando o prompt inteiro cabe nele com folga, e a ausencia de
# truncamento e conferida a posteriori em prompt_eval_count.
NUM_CTX_PATAMARES = (8192, 16384, 32768)

# Parametros de geracao, identicos para os dois idiomas.
LLM_OPTIONS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "num_predict": 900,
}

# Mantem o modelo carregado entre chamadas; evita recarregar 4,9 GB por resumo.
# Nao altera a saida: nao e historico de conversa, e permanencia do processo.
KEEP_ALIVE = "30m"

# ---------------------------------------------------------------------------
# Tokenizadores de destino: BERTimbau em pt, BERT em en (medem o resumo, nao
# a ata)
# ---------------------------------------------------------------------------
TOKENIZER_PATH = {
    "pt": MODELOS_DIR / "bert-base-portuguese-cased",
    "en": MODELOS_DIR / "bert-base-cased",
}

# ---------------------------------------------------------------------------
# L_max -- orcamento de tokens do BERT, incluindo [CLS] e [SEP]
# ---------------------------------------------------------------------------
L_MAX = 512
ADD_SPECIAL_TOKENS = True     # L_max conta [CLS] e [SEP]

# ---------------------------------------------------------------------------
# N_sub, N_sub_max, N_test, K -- numero de submissoes e o teste que o fixou
# ---------------------------------------------------------------------------
# O experimento de determinismo (procedimento TesteDeterminismo do relatorio),
# anterior ao Algoritmo 1, submeteu K atas N_test vezes cada, com o estado do
# servidor limpo antes de cada chamada, e fixou N_sub pela regra:
#
#     se as N_test submissoes de toda ata da amostra forem identicas byte a
#     byte, N_sub = 1; caso contrario, N_sub = N_SUB_MAX.
#
# O Algoritmo 1 recebe esse N_sub como entrada e gera o corpus. A analise de
# variancia sobre o corpus saiu do algoritmo: com N_sub = 1 nao ha pares de
# submissoes a comparar.

# N_sub_max -- submissoes por ata adotadas se o experimento de determinismo
# detectar variacao. Era o valor fixo de N_sub na primeira versao do algoritmo.
N_SUB_MAX = 10

# N_test -- submissoes por ata no experimento de determinismo.
N_SUB_DETERMINISMO = 10

# K -- atas na amostra do experimento (as mesmas datas nos dois idiomas).
N_ATAS_DETERMINISMO = 10

# Valor fixado pelo experimento de determinismo (segunda execucao, 19/10/2026):
# as 20 combinacoes (ata, idioma) produziram dez resumos identicos cada, logo
# N_sub = 1. Ver a secao "Experimento de determinismo" do relatorio.
#
# gerar_resumos.py nao confia nesta constante: le o relatorio da segunda
# execucao do experimento (RELATORIO_ETAPA_A, abaixo) e aborta se ele estiver
# ausente ou se a decisao registrada nele divergir deste valor.
N_SUB = 1

# ---------------------------------------------------------------------------
# S0, P -- sementes espacadas: s_j = S0 + P * j
# ---------------------------------------------------------------------------
S0 = 42
P = 9973

_J_MAXIMO = max(N_SUB_MAX, N_SUB_DETERMINISMO)


def semente(j: int) -> int:
    """
    Semente da submissao j (s_j no procedimento TesteDeterminismo e no
    Algoritmo 1).

    Registre-se que, com temperature = 0, esta semente nao tem efeito algum sobre
    a saida: a decodificacao e gulosa e nenhum numero pseudoaleatorio e
    consumido. Isso foi verificado no experimento de determinismo, em que dez
    sementes distintas produziram o mesmo resumo nas 20 combinacoes. A semente e
    mantida por fidelidade as entradas S0 e P e para que a chamada seja descrita
    por completo, nao porque controle o resultado.
    """
    if not 1 <= j <= _J_MAXIMO:
        raise ValueError(f"j deve estar em [1, {_J_MAXIMO}], recebido {j}")
    return S0 + P * j


SEMENTES = {j: semente(j) for j in range(1, _J_MAXIMO + 1)}

# ---------------------------------------------------------------------------
# Emb -- modelo de embeddings de SimilaridadeEmbedding, congelado
# ---------------------------------------------------------------------------
EMB_MODEL_NOME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMB_MODEL_PATH = MODELOS_DIR / "paraphrase-multilingual-MiniLM-L12-v2"

# ---------------------------------------------------------------------------
# Metricas -- tabela de metricas
# ---------------------------------------------------------------------------
# Consumida pelo experimento de determinismo, que compara as N_test submissoes
# de cada ata da amostra. analisar_variancia.py --fonte corpus tambem a usaria
# se N_sub > 1, caso que nao ocorre com N_sub = 1.
METRICAS = ("lexical", "rougeL", "embedding")

# ---------------------------------------------------------------------------
# Agregacoes do relato de similaridades (RelatarSimilaridades)
# ---------------------------------------------------------------------------
CUTOFF_PERIODO = "2016-01-01"
PERIODOS = ("before_2016", "after_2016", "full")

# ---------------------------------------------------------------------------
# Prompts fixos -- IMUTAVEIS
# ---------------------------------------------------------------------------
# Transcritos da secao "Prompt fixado" do relatorio. O marcador {m} recebe o
# texto completo da ata, nunca um fragmento. Alterar qualquer caractere destes
# prompts invalida a comparabilidade com a rodada de selecao documentada na
# secao "Escolha do prompt".
PROMPT = {
    "pt": """Resuma o documento a seguir em no maximo 300 palavras.
O resumo deve preservar os temas discutidos e as avaliacoes apresentadas, sem
acrescentar informacao que nao esteja no texto original.

Documento:
{m}

Resumo:""",
    "en": """Summarize the document below in at most 300 words.
The summary must preserve the topics discussed and the assessments presented,
without adding information that is not in the original text.

Document:
{m}

Summary:""",
}

# Soma de verificacao dos prompts. Qualquer edicao acidental quebra a checagem
# feita em gerar_resumos.py antes de iniciar a geracao.
PROMPT_SHA256 = {
    "pt": "4bf4c28185d6bfeb4ab271017b9b4be51c58fa468f19df777a4c7e55793b1868",
    "en": "33d62246f885654e4213dfde159c53f37025bf9b846517f63f4b0beb43ca248c",
}

# ---------------------------------------------------------------------------
# Volumes esperados
# ---------------------------------------------------------------------------
# Algoritmo 1 (corpus), com N_SUB = 1:
N_RESUMOS_ESPERADO = N_ATAS_ESPERADO * N_SUB             # 202 por idioma

# Experimento de determinismo, com K atas e N_test submissoes:
N_PARES_DETERMINISMO = (N_SUB_DETERMINISMO
                        * (N_SUB_DETERMINISMO - 1) // 2)  # 45
N_SIMILARIDADES_DETERMINISMO = (N_ATAS_DETERMINISMO
                                * N_PARES_DETERMINISMO
                                * len(METRICAS))          # 1350 por idioma

# ---------------------------------------------------------------------------
# Arquivos de dados
# ---------------------------------------------------------------------------
# Cada condicao experimental tem arquivo proprio. O corpus do estudo (tabela
# Resumo) nao contem nenhum resumo do teste de determinismo, e vice-versa.

# Relatorio da segunda execucao do experimento de determinismo: a decisao de
# N_sub e lida daqui por gerar_resumos.py.
RELATORIO_ETAPA_A = OUTPUT_DIR / "determinismo_frio_relatorio.json"


def arquivo_resumos(lang: str) -> Path:
    """Corpus do estudo (tabela Resumo, Algoritmo 1). Checkpoint append-only."""
    return RESUMOS_DIR / f"resumos_{lang}.jsonl"


def arquivo_frio(lang: str) -> Path:
    """Experimento de determinismo, segunda execucao: estado limpo a cada chamada."""
    return RESUMOS_DIR / f"determinismo_frio_{lang}.jsonl"


def arquivo_padrao(lang: str) -> Path:
    """
    Primeira execucao do teste, sem limpeza de estado entre chamadas.

    Registro historico: foi esta execucao que revelou a dependencia do estado
    do cache e levou a incluir a limpeza (LimparEstado) no teste. Nao e
    reproduzida pelo procedimento atual, e e mantida intacta como evidencia.
    """
    return RESUMOS_DIR / f"determinismo_padrao_{lang}.jsonl"


def arquivo_log(lang: str) -> Path:
    return LOG_DIR / f"geracao_{lang}.log"
