#!/usr/bin/env python3
"""
Verificacao empirica da otimizacao de num_ctx adaptativo.

O Algoritmo 1 declara na entrada LLM um unico valor de num_ctx. Rebaixa-lo por
ata, para o menor patamar que acomode o documento sem truncar, reduz o tempo de
execucao, mas so e legitimo se nao alterar o resumo produzido -- caso contrario o
comprimento da ata ficaria confundido com o patamar usado, e a comparacao entre
atas longas e curtas deixaria de ser limpa.

Este script mede, numa ata que cabe no menor patamar:
  1. se o resumo e identico, caractere por caractere, entre os patamares;
  2. quantos tokens de prompt cada patamar processa (deteccao de truncamento);
  3. o tempo de cada chamada.

Uso:
    python verificar_num_ctx.py --lang pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_alg1 as cfg  # noqa: E402
from gerar_resumos import (  # noqa: E402
    carregar_corpus,
    chamar_llm,
    estimar_tokens_prompt,
    escolher_patamar,
    houve_truncamento,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=cfg.LANGUAGES, default="pt")
    ap.add_argument("--j", type=int, default=1,
                    help="submissao usada no teste (define a semente)")
    args = ap.parse_args()

    lang = args.lang
    df = carregar_corpus(lang)
    df = df.assign(n=df[cfg.COLUNA_TEXTO].str.len()).sort_values("n")

    # Ata curta: tem de caber no menor patamar, senao o teste nao distingue
    # patamar de truncamento.
    linha = None
    for _, cand in df.iterrows():
        prompt = cfg.PROMPT[lang].format(m=cand[cfg.COLUNA_TEXTO])
        if escolher_patamar(estimar_tokens_prompt(prompt, lang)) == \
                cfg.NUM_CTX_PATAMARES[0]:
            linha = cand
            break
    if linha is None:
        raise SystemExit("nenhuma ata cabe no menor patamar; teste inaplicavel")

    texto = linha[cfg.COLUNA_TEXTO]
    prompt = cfg.PROMPT[lang].format(m=texto)
    seed = cfg.semente(args.j)

    print(f"ata {linha['data']} ({lang}): {len(texto)} caracteres, "
          f"prompt estimado em {estimar_tokens_prompt(prompt, lang)} tokens")
    print(f"semente s_{args.j} = {seed}\n")

    resultados = []
    for patamar in cfg.NUM_CTX_PATAMARES:
        res = chamar_llm(prompt, seed, patamar)
        sha = hashlib.sha256(res["resumo"].encode("utf-8")).hexdigest()
        truncou = houve_truncamento(res["prompt_eval_count"], patamar,
                                    len(prompt), lang)
        resultados.append({
            "num_ctx": patamar,
            "sha256_resumo": sha,
            "n_chars_resumo": len(res["resumo"]),
            "prompt_eval_count": res["prompt_eval_count"],
            "eval_count": res["eval_count"],
            "tempo_s": res["tempo_s"],
            "truncamento_detectado": truncou,
        })
        print(f"num_ctx={patamar:>6}  pec={res['prompt_eval_count']:>6}  "
              f"saida={res['eval_count']:>4} tok  {res['tempo_s']:>6.1f}s  "
              f"sha={sha[:16]}  truncou={truncou}")
        # Pausa curta: a troca de num_ctx faz o Ollama recarregar o modelo.
        time.sleep(2)

    shas = {r["sha256_resumo"] for r in resultados}
    identicos = len(shas) == 1
    print()
    if identicos:
        print("RESULTADO: resumo IDENTICO nos tres patamares. A otimizacao de "
              "num_ctx adaptativo nao altera a saida e pode ser adotada.")
    else:
        print("RESULTADO: resumo DIFERENTE entre patamares. A otimizacao NAO "
              "pode ser adotada -- use --num-ctx-modo fixo.")

    saida = {
        "verificacao": "num_ctx_adaptativo",
        "executado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lang": lang,
        "ata": linha["data"],
        "n_chars_ata": int(len(texto)),
        "seed": seed,
        "llm_model": cfg.LLM_MODEL,
        "resumo_identico_entre_patamares": identicos,
        "medicoes": resultados,
    }
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    caminho = cfg.OUTPUT_DIR / f"verificacao_num_ctx_{lang}.json"
    caminho.write_text(json.dumps(saida, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"\nmedicoes salvas em {caminho}")


if __name__ == "__main__":
    main()
