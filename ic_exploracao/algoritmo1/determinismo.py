#!/usr/bin/env python3
"""
Algoritmo 1, etapa A -- teste de determinismo. Decide N_sub.

Implementa as linhas 1 a 23 da especificacao: submete K atas N_test vezes cada,
com o estado do servidor limpo antes de cada chamada (sub-rotina E), verifica se
as submissoes de cada ata sao identicas byte a byte e aplica a regra da linha 23:

    todas identicas  ->  N_sub = 1
    alguma diferente ->  N_sub = N_SUB_MAX

A decisao fica gravada em output/algoritmo1/determinismo_frio_relatorio.json,
que gerar_resumos.py (etapa B) le antes de gerar o corpus.

Modos da etapa A, na ordem de uso:

    python determinismo.py --selecionar
        Fixa a amostra (sub-rotina F) e imprime o comando de execucao.

    python determinismo.py --rodar-frio       (ou ./rodar_determinismo_frio.sh)
        As K * N_test chamadas por idioma, com estado limpo antes de cada uma.
        Ao terminar, emite o veredito e a decisao de N_sub.

    python determinismo.py --verificar-frio
        Refaz o veredito e a decisao a partir dos resumos ja gravados.

Modos da primeira execucao, sem limpeza de estado (registro historico, nota iv):

    python determinismo.py --verificar
        Veredito sobre as submissoes da primeira execucao, gravadas em
        resumos/determinismo_padrao_<lang>.jsonl.

    python determinismo.py --diagnosticar
        Para as atas que variaram naquela execucao, refaz a submissao j = 1 com
        a MESMA semente duas vezes seguidas -- a primeira com o modelo recem
        carregado, a segunda com o prompt ja no cache. Foi este diagnostico que
        mostrou que a variacao vinha do estado do cache e levou a incluir a
        sub-rotina E na etapa A.

O teste foi definido pelo orientador: dez atas, dez submissoes cada,
temperature = 0. Aqui ele vale para os dois idiomas, porque a comparacao entre
portugues e ingles e um dos eixos do estudo e a decisao vale para os dois
corpora.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_alg1 as cfg  # noqa: E402
from gerar_resumos import (  # noqa: E402
    carregar_corpus,
    descarregar_modelo,
    escolher_patamar,
    estimar_tokens_prompt,
    modelo_residente,
)

N_ATAS_AMOSTRA = cfg.N_ATAS_DETERMINISMO          # K
ARQUIVO_AMOSTRA = cfg.OUTPUT_DIR / "amostra_determinismo.json"

# Cada condicao grava em arquivo proprio (ver config_alg1): misturar resumos do
# teste com o corpus do estudo invalidaria os dois.
arquivo_frio = cfg.arquivo_frio
arquivo_padrao = cfg.arquivo_padrao


def selecionar_amostra() -> dict:
    """
    AmostraPorComprimento: amostra fixada por regra, nao por escolha caso a caso.

    As atas sao ordenadas pelo comprimento do texto e tomadas em dez posicoes
    igualmente espacadas da ordenacao, incluindo os dois extremos. O comprimento
    e a variavel relevante aqui: ele determina o tamanho do prompt e o patamar
    de num_ctx, que sao os fatores capazes de alterar o caminho de execucao
    dentro do servidor e, com ele, a saida.

    Criterio definido antes de observar qualquer resumo, e reproduzivel: rodar
    de novo devolve as mesmas dez atas.
    """
    pt = carregar_corpus("pt")
    en = carregar_corpus("en")
    assert set(pt["data"]) == set(en["data"]), "corpora com datas divergentes"

    ordenado = pt.assign(n=pt[cfg.COLUNA_TEXTO].str.len()).sort_values(
        ["n", "data"]).reset_index(drop=True)
    total = len(ordenado)
    posicoes = [round(i * (total - 1) / (N_ATAS_AMOSTRA - 1))
                for i in range(N_ATAS_AMOSTRA)]

    itens = []
    for posicao in posicoes:
        linha = ordenado.loc[posicao]
        data = linha["data"]
        item = {"data": data, "posicao_na_ordenacao": int(posicao),
                "n_chars_pt": int(linha["n"])}
        for lang in cfg.LANGUAGES:
            corpus = pt if lang == "pt" else en
            texto = corpus.loc[corpus["data"] == data,
                               cfg.COLUNA_TEXTO].iloc[0]
            prompt = cfg.PROMPT[lang].format(m=texto)
            item[f"patamar_{lang}"] = escolher_patamar(
                estimar_tokens_prompt(prompt, lang))
        itens.append(item)

    return {
        "experimento": "determinismo_temperature_0",
        "definido_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "criterio": ("dez posicoes igualmente espacadas na ordenacao das 202 "
                     "atas por comprimento do texto em portugues, extremos "
                     "incluidos"),
        "n_atas": N_ATAS_AMOSTRA,
        "n_sub": cfg.N_SUB_DETERMINISMO,
        "idiomas": list(cfg.LANGUAGES),
        "temperature": cfg.LLM_OPTIONS["temperature"],
        "atas": itens,
    }


def _ja_gerados(lang: str, datas: list[str]) -> dict[str, int]:
    """Submissoes do experimento (segunda execucao) ja gravadas, por ata."""
    caminho = cfg.arquivo_frio(lang)
    contagem = {d: 0 for d in datas}
    if not caminho.exists():
        return contagem
    vistos = set()
    with caminho.open(encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                r = json.loads(linha)
            except json.JSONDecodeError:
                continue
            if r["data"] in contagem:
                chave = (r["data"], r["j"])
                if chave not in vistos:
                    vistos.add(chave)
                    contagem[r["data"]] += 1
    return contagem


def modo_selecionar() -> None:
    amostra = selecionar_amostra()
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARQUIVO_AMOSTRA.write_text(
        json.dumps(amostra, ensure_ascii=False, indent=1), encoding="utf-8")

    datas = [a["data"] for a in amostra["atas"]]
    print(f"amostra de {len(datas)} atas, criterio: {amostra['criterio']}\n")
    print(f"{'data':<10} {'pos':>4} {'chars PT':>9} {'ctx pt':>7} {'ctx en':>7}")
    for a in amostra["atas"]:
        print(f"{a['data']:<10} {a['posicao_na_ordenacao']:>4} "
              f"{a['n_chars_pt']:>9} {a['patamar_pt']:>7} {a['patamar_en']:>7}")

    print(f"\namostra salva em {ARQUIVO_AMOSTRA}")

    lista = ",".join(datas)
    falta_total = 0
    print("\nsubmissoes ja geradas desta amostra:")
    for lang in cfg.LANGUAGES:
        contagem = _ja_gerados(lang, datas)
        feito = sum(contagem.values())
        falta = len(datas) * cfg.N_SUB_DETERMINISMO - feito
        falta_total += falta
        print(f"  {lang}: {feito} de {len(datas) * cfg.N_SUB_DETERMINISMO} "
              f"(faltam {falta} chamadas)")

    print(f"\ntotal a gerar: {falta_total} chamadas\n")
    print(f"amostra: {lista}\n")
    print("execucao da etapa A (estado limpo antes de cada chamada):")
    print("  ./rodar_determinismo_frio.sh")
    print("\nao terminar, o veredito e a decisao de N_sub saem no fim do log;")
    print("para refaze-los:  python determinismo.py --verificar-frio")


def modo_verificar(fonte=None, sufixo: str = "", condicao: str = "padrao") -> None:
    """
    Veredito por ata. Serve as duas condicoes experimentais.

    fonte  -- funcao lang -> caminho do JSONL a ler. Por omissao, a primeira
              execucao, sem limpeza de estado (registro historico).
    sufixo -- diferencia os arquivos de saida das duas condicoes.
    """
    if fonte is None:
        fonte = cfg.arquivo_padrao
    if not ARQUIVO_AMOSTRA.exists():
        raise SystemExit(f"ABORTADO: {ARQUIVO_AMOSTRA} nao existe. Rode "
                         "'python determinismo.py --selecionar' antes.")
    amostra = json.loads(ARQUIVO_AMOSTRA.read_text(encoding="utf-8"))
    datas = [a["data"] for a in amostra["atas"]]

    linhas = []
    incompletas = []

    for lang in cfg.LANGUAGES:
        caminho = fonte(lang)
        if not caminho.exists():
            incompletas.append({"lang": lang, "motivo": "sem arquivo"})
            continue
        registros = {}
        with caminho.open(encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if not linha:
                    continue
                try:
                    r = json.loads(linha)
                except json.JSONDecodeError:
                    continue
                if r["data"] in datas:
                    registros[(r["data"], r["j"])] = r

        for data in datas:
            submissoes = [registros[(data, j)] for j in range(1, cfg.N_SUB_DETERMINISMO + 1)
                          if (data, j) in registros]
            if len(submissoes) < cfg.N_SUB_DETERMINISMO:
                incompletas.append({"lang": lang, "data": data,
                                    "submissoes": len(submissoes)})
                continue
            shas = [hashlib.sha256(s["resumo"].encode("utf-8")).hexdigest()
                    for s in submissoes]
            distintos = sorted(set(shas))
            tamanhos = {s["n_tokens_bert"] for s in submissoes}
            linhas.append({
                "lang": lang,
                "data": data,
                "n_submissoes": len(submissoes),
                "n_resumos_distintos": len(distintos),
                "identicos": len(distintos) == 1,
                "n_tokens_bert_distintos": len(tamanhos),
                "sementes": ";".join(str(s["seed"]) for s in submissoes),
                "sha256_resumos": ";".join(s[:12] for s in distintos),
            })

    if incompletas:
        print("atas sem as dez submissoes, excluidas da conclusao:")
        for item in incompletas:
            print(f"  {item}")
        print()

    if not linhas:
        raise SystemExit("ABORTADO: nenhuma ata da amostra tem as dez "
                         "submissoes. Rode a geracao antes de verificar.")

    df = pd.DataFrame(linhas)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.OUTPUT_DIR / f"determinismo{sufixo}_por_ata.csv", index=False)

    print(f"condicao: {condicao}\n")
    print(f"{'lang':<5} {'data':<10} {'subm':>5} {'distintos':>10} {'veredito':>10}")
    for r in linhas:
        print(f"{r['lang']:<5} {r['data']:<10} {r['n_submissoes']:>5} "
              f"{r['n_resumos_distintos']:>10} "
              f"{'identicos' if r['identicos'] else 'VARIARAM':>10}")

    todos_identicos = bool(df["identicos"].all())
    n_atas_variaram = int((~df["identicos"]).sum())

    print()
    if todos_identicos:
        print(f"CONCLUSAO: nas {len(df)} combinacoes (ata, idioma) verificadas, "
              f"as {cfg.N_SUB_DETERMINISMO} submissoes produziram UM UNICO "
              "resumo distinto.")
        print("A replicacao e vazia com temperature = 0: a semente nao tem "
              "efeito porque nao ha amostragem a semear.")
        if condicao == "cache_frio":
            print("Com o estado de cache uniformizado, a variancia do "
                  "sumarizador e exatamente zero -- nao ha residuo a atribuir "
                  "ao caminho de execucao do servidor.")
    else:
        print(f"CONCLUSAO: {n_atas_variaram} de {len(df)} combinacoes "
              "(ata, idioma) produziram resumos diferentes entre submissoes.")
        if condicao == "cache_frio":
            print("Com o estado limpo antes de cada chamada, a variacao nao "
                  "pode ser atribuida ao cache: e variancia do proprio "
                  "sumarizador, e a repeticao passa a ser informativa.")
        else:
            print("Nesta condicao o estado do cache nao e controlado. Rode "
                  "'python determinismo.py --diagnosticar': se a mesma semente "
                  "produzir saidas diferentes conforme o estado do cache, a "
                  "variacao nao vem do sumarizador.")

    # Passo 12 de TesteDeterminismo: so a condicao com estado limpo decide N_sub.
    n_sub_decidido = None
    if condicao == "cache_frio":
        n_sub_decidido = 1 if todos_identicos else cfg.N_SUB_MAX
        print(f"\nDECISAO (linha 23): N_sub = {n_sub_decidido}")

    relatorio = {
        "experimento": amostra["experimento"],
        "condicao": condicao,
        "verificado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "criterio_amostra": amostra["criterio"],
        "n_sub": cfg.N_SUB_DETERMINISMO,
        "temperature": cfg.LLM_OPTIONS["temperature"],
        "llm_model": cfg.LLM_MODEL,
        "llm_digest": cfg.LLM_DIGEST_ESPERADO,
        "n_combinacoes_verificadas": len(df),
        "n_combinacoes_com_variacao": n_atas_variaram,
        "todas_identicas": todos_identicos,
        "regra_decisao": ("N_sub = 1 se todas as submissoes de toda ata da "
                          "amostra forem identicas byte a byte; senao "
                          f"N_sub = {cfg.N_SUB_MAX}"),
        "n_sub_decidido": n_sub_decidido,
        "incompletas": incompletas,
        "por_ata": linhas,
    }
    caminho = cfg.OUTPUT_DIR / f"determinismo{sufixo}_relatorio.json"
    caminho.write_text(json.dumps(relatorio, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"\nrelatorio em {caminho}")
    print(f"tabela por ata em "
          f"{cfg.OUTPUT_DIR / f'determinismo{sufixo}_por_ata.csv'}")


def modo_diagnosticar() -> None:
    """
    Isola a causa da variacao observada.

    Para cada combinacao (ata, idioma) que variou, refaz a submissao j = 1 com a
    sua propria semente, duas vezes seguidas. A unica diferenca entre as duas
    chamadas e o estado do cache de prefixo do servidor: a primeira ocorre com o
    modelo recem carregado e a segunda com o prompt identico ja residente.

    Leitura do resultado:
      - as duas saidas diferem  -> a semente NAO explica a variacao; o estado do
        cache explica, e a submissao nao e reproduzivel apenas fixando a semente;
      - as duas saidas coincidem -> o cache nao e a causa, e a variacao registrada
        tem outra origem, a investigar.
    """
    caminho_relatorio = cfg.OUTPUT_DIR / "determinismo_relatorio.json"
    if not caminho_relatorio.exists():
        raise SystemExit("ABORTADO: rode 'python determinismo.py --verificar' "
                         "antes, para saber quais atas variaram.")
    relatorio = json.loads(caminho_relatorio.read_text(encoding="utf-8"))
    variaram = [r for r in relatorio["por_ata"] if not r["identicos"]]
    if not variaram:
        print("nenhuma combinacao (ata, idioma) variou; nada a diagnosticar.")
        return

    # Importado aqui para que os modos --selecionar e --verificar nao dependam
    # do servidor do LLM estar no ar.
    from gerar_resumos import chamar_llm

    print(f"diagnostico de {len(variaram)} combinacoes que variaram\n")
    linhas = []
    for caso in variaram:
        lang, data = caso["lang"], caso["data"]
        corpus = carregar_corpus(lang)
        texto = corpus.loc[corpus["data"] == data, cfg.COLUNA_TEXTO].iloc[0]
        prompt = cfg.PROMPT[lang].format(m=texto)

        registros = {}
        with cfg.arquivo_padrao(lang).open(encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if not linha:
                    continue
                try:
                    r = json.loads(linha)
                except json.JSONDecodeError:
                    continue
                if r["data"] == data:
                    registros[r["j"]] = r

        sha_j1 = hashlib.sha256(
            registros[1]["resumo"].encode("utf-8")).hexdigest()
        sha_resto = hashlib.sha256(
            registros[2]["resumo"].encode("utf-8")).hexdigest()
        semente = cfg.semente(1)
        num_ctx = registros[1]["num_ctx"]

        print(f"=== {lang} {data} | semente fixa s_1 = {semente} | "
              f"num_ctx = {num_ctx} ===")
        print(f"  registrado: j=1 -> {sha_j1[:12]} | "
              f"j=2..10 -> {sha_resto[:12]}")

        obtidos = []
        for k, estado in enumerate(("fria", "quente"), start=1):
            r = chamar_llm(prompt, semente, num_ctx)
            sha = hashlib.sha256(r["resumo"].encode("utf-8")).hexdigest()
            if sha == sha_j1:
                qual = "reproduziu j=1"
            elif sha == sha_resto:
                qual = "reproduziu j=2..10"
            else:
                qual = "terceira variante"
            obtidos.append({"ordem": k, "estado_cache": estado,
                            "sha256": sha, "corresponde_a": qual,
                            "eval_count": r["eval_count"],
                            "tempo_s": r["tempo_s"]})
            print(f"  chamada {k} ({estado:<6}): {sha[:12]} -> {qual}  "
                  f"{r['tempo_s']:.0f}s")

        semente_explica = obtidos[0]["sha256"] == obtidos[1]["sha256"]
        print(f"  veredito: a semente {'EXPLICA' if semente_explica else 'NAO explica'} "
              f"a variacao\n")
        linhas.append({"lang": lang, "data": data, "semente": semente,
                       "num_ctx": num_ctx, "sha256_j1": sha_j1,
                       "sha256_j2_a_j10": sha_resto,
                       "semente_explica_variacao": semente_explica,
                       "chamadas": obtidos})

    nenhuma_explicada = all(not l["semente_explica_variacao"] for l in linhas)
    print("CONCLUSAO DO DIAGNOSTICO:")
    if nenhuma_explicada:
        print("  Em nenhuma das combinacoes a semente explica a variacao. Duas "
              "chamadas com a MESMA semente produziram saidas diferentes, "
              "diferindo apenas no estado do cache de prefixo do servidor.")
        print("  Consequencia: fixar a semente NAO torna a submissao "
              "reproduzivel; e preciso fixar tambem o estado do cache.")
    else:
        print("  Ha combinacoes em que a semente reproduz a variacao; o cache "
              "nao e a unica causa. Investigar caso a caso.")

    saida = {
        "diagnostico": "causa_da_variacao",
        "executado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hipotese": ("a variacao observada decorre do estado do cache de "
                     "prefixo do servidor, nao da semente"),
        "semente_nunca_explica": nenhuma_explicada,
        "casos": linhas,
    }
    caminho = cfg.OUTPUT_DIR / "determinismo_diagnostico.json"
    caminho.write_text(json.dumps(saida, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"\ndiagnostico em {caminho}")


def modo_rodar_frio(langs: list[str]) -> None:
    """
    Procedimento TesteDeterminismo: o teste com estado limpo (segunda execucao).

    A primeira execucao do teste, sem limpeza de estado, deixava uma assimetria:
    a primeira submissao de cada ata processava o prompt do zero e as nove
    seguintes reaproveitavam o cache de prefixo. Eram dois caminhos de execucao,
    e a diferenca entre eles era a unica variacao observada.

    Aqui o modelo e descarregado antes de CADA chamada (LimparEstado), de modo
    que todas as submissoes partam do mesmo estado. O que sobrar de variacao nao
    pode ser atribuido ao cache: e variancia do proprio sumarizador.

    Grava em arquivo proprio, com checkpoint por (data, j), e e retomavel.
    """
    from gerar_resumos import chamar_llm, digest_modelo, versao_ollama
    from transformers import BertTokenizerFast

    # Procedencia gravada em cada registro. A primeira execucao desta etapa
    # (19/10/2026) nao gravava a versao do servidor; ela teve de ser inferida
    # depois, pelo horario de partida do processo do Ollama.
    versao_servidor = versao_ollama()
    digest = digest_modelo()

    if not ARQUIVO_AMOSTRA.exists():
        raise SystemExit("ABORTADO: rode 'python determinismo.py --selecionar' "
                         "antes, para fixar a amostra.")
    amostra = json.loads(ARQUIVO_AMOSTRA.read_text(encoding="utf-8"))
    datas = [a["data"] for a in amostra["atas"]]

    cfg.RESUMOS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for lang in langs:
        caminho = arquivo_frio(lang)
        feitos = set()
        if caminho.exists():
            with caminho.open(encoding="utf-8") as f:
                for linha in f:
                    linha = linha.strip()
                    if not linha:
                        continue
                    try:
                        r = json.loads(linha)
                    except json.JSONDecodeError:
                        continue
                    feitos.add((r["data"], int(r["j"])))

        corpus = carregar_corpus(lang)
        tokenizer = BertTokenizerFast.from_pretrained(
            str(cfg.TOKENIZER_PATH[lang]), local_files_only=True)

        pendentes = [(d, j) for d in datas
                     for j in range(1, cfg.N_SUB_DETERMINISMO + 1)
                     if (d, j) not in feitos]
        print(f"\n{lang}: {len(pendentes)} de "
              f"{len(datas) * cfg.N_SUB_DETERMINISMO} submissoes pendentes")

        t0 = time.time()
        for i, (data, j) in enumerate(pendentes, 1):
            texto = corpus.loc[corpus["data"] == data,
                               cfg.COLUNA_TEXTO].iloc[0]
            prompt = cfg.PROMPT[lang].format(m=texto)
            num_ctx = amostra["atas"][datas.index(data)][f"patamar_{lang}"]

            # O teste so vale para uma versao do servidor: se ela mudar no meio,
            # as submissoes de uma mesma ata deixariam de ser comparaveis.
            if versao_ollama() != versao_servidor:
                raise SystemExit(
                    f"ABORTADO: o Ollama mudou de versao durante a etapa A "
                    f"({versao_servidor} -> {versao_ollama()}). Refaca a etapa "
                    "inteira na versao nova.")

            # Estado frio, verificado e nao presumido.
            descarregar_modelo()
            residentes_antes = modelo_residente()
            if residentes_antes:
                raise RuntimeError(
                    f"estado frio nao obtido antes de ({data}, j={j}): "
                    f"residentes {residentes_antes}")

            # keep_alive = 0 faz o servidor liberar o modelo ao terminar, de modo
            # que a proxima chamada tambem parta de frio.
            res = chamar_llm(prompt, cfg.semente(j), num_ctx, keep_alive=0)
            n_tok = len(tokenizer.encode(
                res["resumo"], add_special_tokens=cfg.ADD_SPECIAL_TOKENS))

            registro = {
                "condicao": "cache_frio",
                "lang": lang,
                "data": data,
                "j": j,
                "seed": cfg.semente(j),
                "resumo": res["resumo"],
                "n_tokens_bert": n_tok,
                "excede_l_max": n_tok > cfg.L_MAX,
                "excesso": max(0, n_tok - cfg.L_MAX),
                "n_chars_resumo": len(res["resumo"]),
                "n_chars_ata": len(texto),
                "num_ctx": num_ctx,
                "prompt_eval_count": res["prompt_eval_count"],
                "eval_count": res["eval_count"],
                "done_reason": res["done_reason"],
                "modelo_residente_antes": False,
                "keep_alive": 0,
                "tempo_s": res["tempo_s"],
                "llm_model": cfg.LLM_MODEL,
                "llm_digest": digest,
                "ollama_version": versao_servidor,
                "prompt_sha256": cfg.PROMPT_SHA256[lang],
                "gerado_em": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"),
            }
            with caminho.open("a", encoding="utf-8") as f:
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

            media = (time.time() - t0) / i
            restam = len(pendentes) - i
            print(f"  [{lang} {i}/{len(pendentes)}] ata {data} j={j} "
                  f"ctx={num_ctx} pec={res['prompt_eval_count']} "
                  f"tok_bert={n_tok} {res['tempo_s']:.0f}s | media "
                  f"{media:.0f}s | restam {restam} (~{restam * media / 3600:.1f} h)",
                  flush=True)

    print("\ngeracao em cache frio concluida. Veredito:")
    modo_verificar(fonte=arquivo_frio, sufixo="_frio", condicao="cache_frio")


def modo_verificar_versao() -> None:
    """
    Verifica, na versao atual do servidor, a equivalencia de que depende o
    Algoritmo 1.

    O Algoritmo 1 com N_sub = 1 dispensa limpar o estado porque a primeira
    chamada de uma ata -- com o cache ocupado pelo prompt de OUTRA ata -- produz o
    mesmo resumo que uma chamada em estado limpo. Isso foi demonstrado na 0.35.1.
    Este modo testa a mesma coisa na versao em uso, nas combinacoes que variaram
    na primeira execucao do teste, que sao as sensiveis ao estado do cache.

    Para cada caso, reproduz a condicao exata do corpus:
      1. limpa o estado;
      2. chama a ata que a precedeu na geracao do corpus, no mesmo patamar de
         num_ctx -- o modelo fica carregado e o cache passa a conter outra ata;
      3. chama a ata-alvo: e a "primeira chamada", o caminho do corpus;
      4. chama a ata-alvo de novo: e a "chamada quente", que reaproveita o
         prompt identico. Mostra se a dependencia do cache persiste nesta versao.
    O resultado de (3) tem de ser identico ao da segunda execucao do
    experimento de determinismo (estado limpo).
    """
    from gerar_resumos import chamar_llm, versao_ollama

    rel_padrao = json.loads(
        (cfg.OUTPUT_DIR / "determinismo_relatorio.json").read_text(encoding="utf-8"))
    casos = [(r["lang"], r["data"]) for r in rel_padrao["por_ata"]
             if not r["identicos"]]
    amostra = json.loads(ARQUIVO_AMOSTRA.read_text(encoding="utf-8"))
    patamar = {(lang, a["data"]): a[f"patamar_{lang}"]
               for a in amostra["atas"] for lang in cfg.LANGUAGES}
    versao = versao_ollama()

    def sha(texto: str) -> str:
        return hashlib.sha256(texto.encode("utf-8")).hexdigest()

    def ler(caminho: Path) -> dict:
        out = {}
        with caminho.open(encoding="utf-8") as f:
            for linha in f:
                if linha.strip():
                    r = json.loads(linha)
                    out.setdefault((r["data"], r["j"]), r)
        return out

    print(f"servidor {versao} | {len(casos)} casos sensiveis ao cache\n")
    resultados = []
    for lang, data in casos:
        corpus = carregar_corpus(lang)
        texto = dict(zip(corpus["data"], corpus[cfg.COLUNA_TEXTO]))
        ctx = patamar[(lang, data)]

        # Ata que precedeu a alvo na geracao do corpus: o Algoritmo 1 ordena por
        # (patamar, data), entao e a ata de data imediatamente anterior no
        # mesmo patamar.
        mesmo_patamar = sorted(
            d for d in corpus["data"]
            if escolher_patamar(estimar_tokens_prompt(
                cfg.PROMPT[lang].format(m=texto[d]), lang)) == ctx)
        anterior = mesmo_patamar[mesmo_patamar.index(data) - 1]

        frio = ler(arquivo_frio(lang))[(data, 1)]
        padrao = ler(arquivo_padrao(lang))
        sha_limpo = sha(frio["resumo"])
        sha_quente_antigo = sha(padrao[(data, 2)]["resumo"])
        prompt = cfg.PROMPT[lang].format(m=texto[data])

        print(f"=== {lang} {data} | num_ctx {ctx} | ata anterior {anterior}")
        descarregar_modelo()
        chamar_llm(cfg.PROMPT[lang].format(m=texto[anterior]),
                   cfg.semente(1), ctx)
        primeira = chamar_llm(prompt, cfg.semente(1), ctx)
        quente = chamar_llm(prompt, cfg.semente(1), ctx)

        sha_primeira, sha_quente = sha(primeira["resumo"]), sha(quente["resumo"])
        ok = sha_primeira == sha_limpo
        print(f"  primeira chamada = estado limpo (etapa A)? "
              f"{'SIM' if ok else 'NAO'}  {primeira['tempo_s']:.0f}s")
        print(f"  chamada quente   = primeira chamada? "
              f"{'sim' if sha_quente == sha_primeira else 'nao'}"
              f"{' | = variante quente da 0.35.1' if sha_quente == sha_quente_antigo else ''}"
              f"  {quente['tempo_s']:.0f}s\n")
        resultados.append({
            "lang": lang, "data": data, "num_ctx": ctx, "ata_anterior": anterior,
            "primeira_igual_estado_limpo": ok,
            "quente_igual_primeira": sha_quente == sha_primeira,
            "quente_igual_variante_quente_0351": sha_quente == sha_quente_antigo,
            "sha256_primeira": sha_primeira, "sha256_quente": sha_quente,
            "sha256_estado_limpo": sha_limpo,
        })

    todos = all(r["primeira_igual_estado_limpo"] for r in resultados)
    print("CONCLUSAO:", "em todos os casos a primeira chamada reproduz o estado "
          "limpo nesta versao." if todos else "ha caso em que a primeira chamada "
          "NAO reproduz o estado limpo nesta versao.")
    saida = {"verificacao": "primeira_chamada_equivale_estado_limpo",
             "executado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "ollama_version": versao, "todos_equivalentes": todos,
             "casos": resultados}
    caminho = cfg.OUTPUT_DIR / f"verificacao_versao_{versao}.json"
    caminho.write_text(json.dumps(saida, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"\nresultado em {caminho}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    grupo = ap.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--selecionar", action="store_true")
    grupo.add_argument("--verificar", action="store_true")
    grupo.add_argument("--diagnosticar", action="store_true")
    grupo.add_argument("--rodar-frio", action="store_true",
                       help="repete o experimento descarregando o modelo antes "
                            "de cada chamada, uniformizando o estado do cache")
    grupo.add_argument("--verificar-frio", action="store_true",
                       help="veredito sobre os resumos gerados em cache frio")
    grupo.add_argument("--verificar-versao", action="store_true",
                       help="testa, na versao atual do servidor, que a primeira "
                            "chamada de uma ata equivale ao estado limpo")
    ap.add_argument("--lang", choices=[*cfg.LANGUAGES, "todos"], default="todos")
    args = ap.parse_args()

    langs = list(cfg.LANGUAGES) if args.lang == "todos" else [args.lang]

    if args.selecionar:
        modo_selecionar()
    elif args.verificar:
        modo_verificar()
    elif args.diagnosticar:
        modo_diagnosticar()
    elif args.rodar_frio:
        modo_rodar_frio(langs)
    elif args.verificar_versao:
        modo_verificar_versao()
    else:
        modo_verificar(fonte=arquivo_frio, sufixo="_frio",
                       condicao="cache_frio")


if __name__ == "__main__":
    main()
