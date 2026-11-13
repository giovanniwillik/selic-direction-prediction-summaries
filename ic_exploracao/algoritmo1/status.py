#!/usr/bin/env python3
"""
Estado da geracao de resumos do Algoritmo 1.

Le o checkpoint e informa quanto falta, sem tocar no LLM. Pode ser rodado a
qualquer momento, inclusive com a geracao em andamento.

Uso:
    python status.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_alg1 as cfg  # noqa: E402

# Modelo de tempo ajustado sobre 173 resumos do corpus (Algoritmo 1, 20/10/2026),
# cada um a primeira e unica chamada da sua ata. O ajuste anterior (25,1 +
# 0,00285 por token) vinha de submissoes repetidas da mesma ata, que reaproveitam
# o prompt ja processado no cache e por isso subestimava o tempo em cerca de 2,5
# vezes.
TEMPO_BASE_S = 9.5
TEMPO_POR_TOKEN_S = 0.01462
CHARS_POR_TOKEN = {"pt": 3.38, "en": 4.57}


def em_execucao() -> list[str]:
    try:
        saida = subprocess.run(["pgrep", "-fl", "gerar_resumos.py"],
                               capture_output=True, text=True, timeout=10).stdout
        return [l for l in saida.splitlines() if "pgrep" not in l]
    except Exception:
        return []


def main() -> None:
    processos = em_execucao()
    print("geracao em execucao: " + ("sim" if processos else "nao"))
    for p in processos:
        print(f"  {p}")
    print()

    total_feito = total_esperado = 0
    segundos_restantes = 0.0

    for lang in cfg.LANGUAGES:
        caminho = cfg.arquivo_resumos(lang)
        df_corpus = pd.read_csv(cfg.DATASET_CSV[lang], index_col=0)
        n_atas = len(df_corpus)
        esperado = n_atas * cfg.N_SUB

        feitos = []
        if caminho.exists():
            with caminho.open(encoding="utf-8") as f:
                for linha in f:
                    linha = linha.strip()
                    if linha:
                        try:
                            feitos.append(json.loads(linha))
                        except json.JSONDecodeError:
                            pass  # ultima linha truncada por interrupcao

        df = pd.DataFrame(feitos)
        if len(df):
            df = df.drop_duplicates(subset=["data", "j"], keep="last")

        n = len(df)
        # ">=" e nao "==": as dez atas do experimento de determinismo tem dez
        # submissoes cada e tambem estao completas para o estudo, que pede uma.
        atas_completas = (
            int((df.groupby("data")["j"].nunique() >= cfg.N_SUB).sum()) if n else 0)
        excedem = int(df["excede_l_max"].sum()) if n else 0
        pct = 100 * n / esperado if esperado else 0

        # Tempo restante: estimado pelas atas que ainda tem submissoes pendentes.
        pendentes_por_ata = {}
        feitos_por_ata = df.groupby("data")["j"].nunique().to_dict() if n else {}
        for _, linha in df_corpus.iterrows():
            data = str(int(linha[cfg.COLUNA_DATA]))
            falta = cfg.N_SUB - feitos_por_ata.get(data, 0)
            if falta > 0:
                n_tok = len(linha[cfg.COLUNA_TEXTO]) / CHARS_POR_TOKEN[lang]
                pendentes_por_ata[data] = falta * (
                    TEMPO_BASE_S + TEMPO_POR_TOKEN_S * n_tok)
        falta_s = sum(pendentes_por_ata.values())
        segundos_restantes += falta_s

        print(f"{lang}: {n} de {esperado} resumos ({pct:.1f}%) | "
              f"{atas_completas} de {n_atas} atas completas")
        print(f"     {excedem} resumos excedem {cfg.L_MAX} tokens | "
              f"faltam ~{falta_s / 3600:.1f} h")
        if n:
            print(f"     tempo medio por resumo: {df['tempo_s'].mean():.0f}s | "
                  f"tokens BERT mediana: {df['n_tokens_bert'].median():.0f}")

        total_feito += n
        total_esperado += esperado

    pct = 100 * total_feito / total_esperado if total_esperado else 0
    print(f"\nTOTAL: {total_feito} de {total_esperado} resumos ({pct:.1f}%) | "
          f"faltam ~{segundos_restantes / 3600:.1f} h de processamento")

    if not processos and total_feito < total_esperado:
        print("\nPara retomar:  ./rodar_geracao.sh")


if __name__ == "__main__":
    main()
