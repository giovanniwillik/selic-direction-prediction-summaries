#!/usr/bin/env python3
"""
Algoritmo 1 -- comparacao par a par dos resumos e relato das distribuicoes.

Sub-rotinas A a D (similaridades) e G (relato em quatro niveis de agregacao),
aplicadas a uma de tres fontes, escolhida por --fonte:

  frio    etapa A, linhas 15 a 21: as N_test submissoes de cada ata da amostra,
          geradas com estado limpo antes de cada chamada. E a fonte que decide
          N_sub.
  padrao  primeira execucao do teste, sem limpeza de estado. Registro historico
          (nota iv); reproduz as estatisticas que levaram ao diagnostico.
  corpus  etapa B, linha 39 (relato de Over[lang], S2) e, apenas se N_sub > 1,
          etapa C, linhas 41 a 51. Com N_sub = 1 nao ha pares, e so S2 e
          produzido.

Nao chama o LLM e pode ser reexecutado quantas vezes for preciso sobre os
mesmos resumos.

Saidas, em output/algoritmo1/, com o nome da fonte como sufixo:
  over_l_max_<fonte>_<lang>.csv        resumos acima de L_max, com o excesso (S2)
  over_l_max_<fonte>_resumo.csv        contagem e estatisticas do excesso
  similaridades_<fonte>_<lang>.parquet um registro por (metrica, ata, a, b)
  similaridades_<fonte>_<lang>.csv.gz  mesma tabela, formato textual
  estatisticas_<fonte>_global.csv      agregacao (i)
  estatisticas_<fonte>_por_ata.csv     agregacao (ii)
  estatisticas_<fonte>_por_ano.csv     agregacao (iii)
  estatisticas_<fonte>_por_periodo.csv agregacao (iv)

Uso:
    python analisar_variancia.py --fonte frio
    python analisar_variancia.py --fonte corpus
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_alg1 as cfg  # noqa: E402


# ---------------------------------------------------------------------------
# Pares de submissoes (a, b) do laco de TesteDeterminismo -- COMBINATIONS(n, k)
# ---------------------------------------------------------------------------
def combinations(n: int, k: int = 2) -> list[tuple[int, int]]:
    """Pares de INDICES (a, b) com 1 <= a < b <= n. Nunca resumos."""
    return list(itertools.combinations(range(1, n + 1), k))


# ---------------------------------------------------------------------------
# Normalizacao de texto para SimilaridadeLexical e SimilaridadeRougeL
# ---------------------------------------------------------------------------
# O corpus de entrada e ASCII puro (as atas foram normalizadas por NFKD com
# remocao de nao-ASCII na origem), mas o resumo e texto livre gerado pelo LLM e
# pode conter acentos. A tokenizacao padrao do pacote rouge-score descarta
# caracteres fora de [a-z0-9], o que fragmentaria palavras acentuadas de forma
# arbitraria. Por isso SimilaridadeLexical e SimilaridadeRougeL recebem o texto
# dobrado para ASCII.
#
# A transformacao e aplicada identicamente aos dois membros de todo par, logo
# nao pode enviesar uma comparacao par a par. Ela nao altera o corpus (tabela
# Resumo): o resumo e gravado como saiu do modelo, e e a versao literal que o
# Algoritmo 2 consome. SimilaridadeEmbedding recebe o resumo literal, porque o
# codificador de sentencas e multilingue e lida com acentuacao nativamente.
_NAO_ALFANUM = re.compile(r"[^a-z0-9]+")


def dobrar_para_ascii(texto: str) -> str:
    return (unicodedata.normalize("NFKD", texto)
            .encode("ascii", "ignore")
            .decode("ascii"))


def tokenizar(texto: str) -> list[str]:
    """Minusculas, pontuacao removida, espacos colapsados."""
    return _NAO_ALFANUM.sub(" ", dobrar_para_ascii(texto).lower()).split()


# ---------------------------------------------------------------------------
# SimilaridadeLexical -- indice de Jaccard sobre conjuntos
# ---------------------------------------------------------------------------
def lexical_similarity(x: str, y: str) -> float:
    X = set(tokenizar(x))
    Y = set(tokenizar(y))
    uniao = X | Y
    if not uniao:
        return 0.0
    return len(X & Y) / len(uniao)


# ---------------------------------------------------------------------------
# SimilaridadeRougeL -- ROUGE-L, variante F1
# ---------------------------------------------------------------------------
def _lcs(a: list[str], b: list[str]) -> int:
    """Comprimento da maior subsequencia comum, em O(|a|*|b|) tempo e O(|b|)
    memoria."""
    if not a or not b:
        return 0
    anterior = [0] * (len(b) + 1)
    for tok_a in a:
        atual = [0] * (len(b) + 1)
        for j, tok_b in enumerate(b, 1):
            if tok_a == tok_b:
                atual[j] = anterior[j - 1] + 1
            else:
                atual[j] = atual[j - 1] if atual[j - 1] >= anterior[j] else anterior[j]
        anterior = atual
    return anterior[-1]


def rougel_similarity(x: str, y: str) -> float:
    Xs = tokenizar(x)
    Ys = tokenizar(y)
    if not Xs or not Ys:
        return 0.0
    l = _lcs(Xs, Ys)
    precision = l / len(Xs)
    recall = l / len(Ys)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# SimilaridadeEmbedding -- cosseno de embeddings de sentenca
# ---------------------------------------------------------------------------
def cosseno(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def carregar_codificador():
    from sentence_transformers import SentenceTransformer
    if cfg.EMB_MODEL_PATH.exists():
        return SentenceTransformer(str(cfg.EMB_MODEL_PATH))
    return SentenceTransformer(cfg.EMB_MODEL_NOME)


# ---------------------------------------------------------------------------
# Conferencia de SimilaridadeRougeL contra a implementacao de referencia
# ---------------------------------------------------------------------------
def conferir_rougel() -> dict:
    """
    O relatorio define SimilaridadeRougeL pela maior subsequencia comum, sem
    remeter a um pacote. A implementacao local existe para receber o texto ja
    dobrado para ASCII, o que a tokenizacao do pacote rouge-score nao permite
    configurar sem reescrever o tokenizador. Esta funcao confere que a
    implementacao local e o rougeL do pacote concordam quando recebem o mesmo
    texto, de modo que o calculo local mede o mesmo que a referencia.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {"conferido": False, "motivo": "rouge-score ausente"}

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    casos = [
        ("o comite avaliou a inflacao corrente e o balanco de riscos",
         "o comite avaliou o balanco de riscos e a inflacao corrente"),
        ("a atividade economica desacelerou no trimestre",
         "a atividade economica desacelerou no trimestre"),
        ("cenario externo adverso", "mercado de trabalho aquecido"),
        ("inflacao acima da meta com expectativas desancoradas",
         "expectativas desancoradas e inflacao acima da meta no periodo"),
    ]
    maior_desvio = 0.0
    for x, y in casos:
        referencia = scorer.score(x, y)["rougeL"].fmeasure
        local = rougel_similarity(x, y)
        maior_desvio = max(maior_desvio, abs(referencia - local))
    return {"conferido": True, "n_casos": len(casos),
            "maior_desvio_absoluto": maior_desvio,
            "concordam": maior_desvio < 1e-9}


# ---------------------------------------------------------------------------
# Leitura dos resumos
# ---------------------------------------------------------------------------
FONTES = {
    # nome: (funcao lang -> arquivo, submissoes por ata esperadas)
    "frio": (cfg.arquivo_frio, cfg.N_SUB_DETERMINISMO),
    "padrao": (cfg.arquivo_padrao, cfg.N_SUB_DETERMINISMO),
    "corpus": (cfg.arquivo_resumos, cfg.N_SUB),
}


def carregar_resumos(lang: str, fonte: str = "corpus") -> pd.DataFrame:
    caminho = FONTES[fonte][0](lang)
    if not caminho.exists():
        raise SystemExit(f"ABORTADO: {caminho} nao existe. Gere os resumos da "
                         f"fonte '{fonte}' antes da analise.")
    registros = []
    with caminho.open(encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                registros.append(json.loads(linha))
    df = pd.DataFrame(registros)

    # Uma interrupcao pode deixar o mesmo (data, j) gravado duas vezes se o
    # script foi reexecutado sem que a linha anterior fosse lida. Mantem-se o
    # registro mais recente, que e o ultimo do arquivo.
    antes = len(df)
    df = df.drop_duplicates(subset=["data", "j"], keep="last")
    if len(df) != antes:
        print(f"aviso: {antes - len(df)} registros duplicados de (data, j) "
              f"descartados em {lang}")

    df["data_dt"] = pd.to_datetime(df["data"], format="%Y%m%d")
    df["ano"] = df["data_dt"].dt.year
    return df.sort_values(["data", "j"]).reset_index(drop=True)


def periodo_da_data(d: pd.Timestamp) -> str:
    return "before_2016" if d < pd.Timestamp(cfg.CUTOFF_PERIODO) else "after_2016"


# ---------------------------------------------------------------------------
# Comparacao par a par, dentro de cada ata (passo 10 de TesteDeterminismo; o
# mesmo calculo se aplicaria ao corpus se N_sub > 1)
# ---------------------------------------------------------------------------
def calcular_similaridades(df: pd.DataFrame, lang: str, n_sub: int,
                           codificador) -> tuple[pd.DataFrame, dict]:
    pares = combinations(n_sub, 2)
    linhas = []
    atas_completas = 0
    atas_incompletas = []

    t0 = time.time()
    grupos = list(df.groupby("data", sort=True))
    for i, (data, g) in enumerate(grupos, 1):
        por_j = {int(r.j): r.resumo for r in g.itertuples()}
        if sorted(por_j) != list(range(1, n_sub + 1)):
            atas_incompletas.append({"data": data,
                                     "j_presentes": sorted(por_j)})
            continue
        atas_completas += 1

        # Codifica os n_sub resumos da ata de uma vez; o cosseno depois e
        # aritmetica. A comparacao permanece estritamente intra-ata.
        emb = codificador.encode([por_j[j] for j in range(1, n_sub + 1)],
                                 batch_size=n_sub, show_progress_bar=False,
                                 convert_to_numpy=True)

        ano = int(str(data)[:4])
        periodo = periodo_da_data(pd.Timestamp(data))
        for a, b in pares:
            x, y = por_j[a], por_j[b]
            valores = {
                "lexical": lexical_similarity(x, y),
                "rougeL": rougel_similarity(x, y),
                "embedding": cosseno(emb[a - 1], emb[b - 1]),
            }
            for nome in cfg.METRICAS:
                linhas.append({"metrica": nome, "lang": lang, "data": data,
                               "ano": ano, "periodo": periodo,
                               "a": a, "b": b, "valor": valores[nome]})

        if i % 20 == 0 or i == len(grupos):
            print(f"  {lang}: {i}/{len(grupos)} atas processadas "
                  f"({time.time() - t0:.0f}s)")

    sim = pd.DataFrame(linhas)
    meta = {"atas_completas": atas_completas,
            "atas_incompletas": atas_incompletas,
            "n_pares_por_ata": len(pares),
            "n_similaridades": len(sim),
            "tempo_s": round(time.time() - t0, 1)}
    return sim, meta


# ---------------------------------------------------------------------------
# Bloco (c) -- relato
# ---------------------------------------------------------------------------
def estatisticas(valores: np.ndarray) -> dict:
    """Sumario de cinco numeros do boxplot, mais media e dispersao.

    Quartis por interpolacao linear (padrao do numpy). Desvio-padrao amostral
    (ddof = 1), que e o padrao do pandas para dispersao descritiva.
    """
    v = np.asarray(valores, dtype=float)
    q1, mediana, q3 = np.percentile(v, [25, 50, 75])
    return {
        "n": int(v.size),
        "minimo": float(v.min()),
        "q1": float(q1),
        "mediana": float(mediana),
        "q3": float(q3),
        "maximo": float(v.max()),
        "iqr": float(q3 - q1),
        "media": float(v.mean()),
        "desvio_padrao": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
    }


def agregar(sim: pd.DataFrame, chaves: list[str]) -> pd.DataFrame:
    linhas = []
    for valores_chave, g in sim.groupby(chaves, sort=True):
        if not isinstance(valores_chave, tuple):
            valores_chave = (valores_chave,)
        registro = dict(zip(chaves, valores_chave))
        registro.update(estatisticas(g["valor"].to_numpy()))
        linhas.append(registro)
    return pd.DataFrame(linhas)


def agregar_por_periodo(sim: pd.DataFrame) -> pd.DataFrame:
    """Agregacao (iv). O periodo 'full' e o conjunto inteiro, nao um recorte,
    e por isso nao sai de um groupby sobre a coluna periodo."""
    partes = [agregar(sim, ["metrica", "lang", "periodo"])]
    completo = agregar(sim, ["metrica", "lang"])
    completo["periodo"] = "full"
    partes.append(completo)
    out = pd.concat(partes, ignore_index=True)
    ordem = {p: i for i, p in enumerate(cfg.PERIODOS)}
    return (out.assign(_o=out["periodo"].map(ordem))
            .sort_values(["metrica", "lang", "_o"])
            .drop(columns="_o")
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Lista Acima[lang] do Algoritmo 1 (resumos acima de L_max)
# ---------------------------------------------------------------------------
def relatar_over(df: pd.DataFrame, lang: str) -> tuple[pd.DataFrame, dict]:
    over = df[df["excede_l_max"]][["data", "j", "n_tokens_bert", "excesso"]]
    over = over.sort_values(["data", "j"]).reset_index(drop=True)
    resumo = {
        "lang": lang,
        "n_resumos": int(len(df)),
        "n_excedem_l_max": int(len(over)),
        "proporcao_excedem": round(len(over) / len(df), 6) if len(df) else 0.0,
        "n_atas_com_excesso": int(over["data"].nunique()),
    }
    if len(over):
        resumo.update({f"excesso_{k}": v
                       for k, v in estatisticas(over["excesso"].to_numpy()).items()})
    return over, resumo


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fonte", choices=list(FONTES), required=True,
                    help="frio: etapa A; padrao: primeira execucao do teste; "
                         "corpus: etapa B (S2) e, se N_sub > 1, etapa C")
    ap.add_argument("--lang", choices=[*cfg.LANGUAGES, "todos"], default="todos")
    args = ap.parse_args()
    args.n_sub = FONTES[args.fonte][1]
    sufixo = f"_{args.fonte}"

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    langs = list(cfg.LANGUAGES) if args.lang == "todos" else [args.lang]
    comparar_pares = args.n_sub > 1
    print(f"fonte: {args.fonte} | submissoes por ata: {args.n_sub} | "
          f"comparacao par a par: {'sim' if comparar_pares else 'nao (N_sub = 1)'}")

    conferencia = {"conferido": False, "motivo": "sem comparacao par a par"}
    codificador = None
    if comparar_pares:
        conferencia = conferir_rougel()
        print(f"conferencia da sub-rotina C contra rouge-score: {conferencia}")
        if conferencia.get("conferido") and not conferencia.get("concordam"):
            raise SystemExit("ABORTADO: a implementacao local de ROUGE-L "
                             "divergiu do pacote rouge-score.")
        codificador = carregar_codificador()
        print(f"codificador de embeddings carregado de "
              f"{cfg.EMB_MODEL_PATH if cfg.EMB_MODEL_PATH.exists() else cfg.EMB_MODEL_NOME}")

    sims = []
    metas = {}
    overs_resumo = []
    nao_ascii = {}

    for lang in langs:
        df = carregar_resumos(lang, args.fonte)
        print(f"\n{lang}: {len(df)} resumos de {df['data'].nunique()} atas")

        over, over_resumo = relatar_over(df, lang)
        over.to_csv(cfg.OUTPUT_DIR / f"over_l_max{sufixo}_{lang}.csv",
                    index=False)
        overs_resumo.append(over_resumo)
        print(f"  S2: {over_resumo['n_excedem_l_max']} resumos excedem "
              f"{cfg.L_MAX} tokens, em {over_resumo['n_atas_com_excesso']} "
              f"de {df['data'].nunique()} atas")

        # Quantos resumos trazem caracteres fora de ASCII: justifica o
        # dobramento aplicado em SimilaridadeLexical e SimilaridadeRougeL.
        nao_ascii[lang] = int(sum(1 for s in df["resumo"]
                                  if any(ord(c) > 127 for c in s)))

        if not comparar_pares:
            continue

        sim, meta = calcular_similaridades(df, lang, args.n_sub, codificador)
        metas[lang] = meta
        print(f"  similaridades: {len(sim)} de {meta['atas_completas']} atas "
              f"completas em {meta['tempo_s']}s")
        if meta["atas_incompletas"]:
            print(f"  aviso: {len(meta['atas_incompletas'])} atas sem as "
                  f"{args.n_sub} submissoes foram ignoradas")

        sim.to_parquet(cfg.OUTPUT_DIR / f"similaridades{sufixo}_{lang}.parquet",
                       index=False)
        sim.to_csv(cfg.OUTPUT_DIR / f"similaridades{sufixo}_{lang}.csv.gz",
                   index=False, compression="gzip")
        sims.append(sim)

    pd.DataFrame(overs_resumo).to_csv(
        cfg.OUTPUT_DIR / f"over_l_max{sufixo}_resumo.csv", index=False)

    if comparar_pares:
        todas = pd.concat(sims, ignore_index=True)
        saidas = {
            "global": agregar(todas, ["metrica", "lang"]),
            "por_ata": agregar(todas, ["metrica", "lang", "data"]),
            "por_ano": agregar(todas, ["metrica", "lang", "ano"]),
            "por_periodo": agregar_por_periodo(todas),
        }
        for nome, tabela in saidas.items():
            arquivo = f"estatisticas{sufixo}_{nome}.csv"
            tabela.to_csv(cfg.OUTPUT_DIR / arquivo, index=False)
            print(f"sub-rotina G: {arquivo} -- {len(tabela)} linhas")

        print("\nagregacao global (i):")
        with pd.option_context("display.width", 140, "display.max_columns", 20):
            print(saidas["global"].to_string(index=False))

    registro = {
        "etapa": "analise_variancia",
        "fonte": args.fonte,
        "executado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argumentos": vars(args),
        "conferencia_rougel": conferencia,
        "modelo_embeddings": cfg.EMB_MODEL_NOME if comparar_pares else None,
        "n_sub": args.n_sub,
        "n_pares_por_ata": len(combinations(args.n_sub, 2)),
        "resumos_com_caracteres_nao_ascii": nao_ascii,
        "por_idioma": metas,
        "over_l_max": overs_resumo,
    }
    caminho = cfg.OUTPUT_DIR / "registro_execucao_analise.jsonl"
    with caminho.open("a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False, default=str) + "\n")
    print(f"\nregistro de execucao anexado em {caminho}")


if __name__ == "__main__":
    main()
