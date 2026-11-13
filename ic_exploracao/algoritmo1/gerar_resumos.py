#!/usr/bin/env python3
"""
Algoritmo 1, etapa B -- geracao do corpus de resumos.

Implementa as linhas 25 a 39 da especificacao: para cada idioma e cada ata,
produz N_sub resumos com o LLM local, mede o comprimento de cada um com o
tokenizador BERT do idioma e registra os que excedem L_max.

N_sub nao e escolhido aqui. Ele e a decisao da etapa A (linha 23), lida do
relatorio que determinismo.py grava. Sem esse relatorio a geracao nao roda.

Estado de execucao (nota v da especificacao): com N_sub = 1, cada chamada e a
primeira e unica da sua ata, e a etapa A mostrou que esse caminho coincide com
o de uma chamada em estado limpo. Para que isso valha tambem depois de
interrupcoes, o modelo e descarregado na partida e antes de cada retentativa --
assim nenhuma parte do prompt da ata em andamento fica no cache. Se N_sub > 1,
o estado e limpo antes de toda chamada, como na etapa A.

Saidas:
  resumos/resumos_<lang>.jsonl   S1, append-only, um JSON por resumo
  output/algoritmo1/logs/        log de progresso e falhas

Nao calcula similaridades nem o relato de S2: isso e feito por
analisar_variancia.py, que le o JSONL. A geracao leva horas, a analise leva
segundos, e refazer a analise nao deve exigir regerar nada.

Checkpoint: a chave de um resumo e (lang, data, j). Na partida o script le o
JSONL existente e pula o que ja esta lá, de modo que uma interrupcao custa, no
pior caso, o resumo em andamento.

Uso:
    python gerar_resumos.py --lang pt
    python gerar_resumos.py --lang pt --atas 20250129,20241211   # piloto
    python gerar_resumos.py --lang pt --dry-run                  # so o plano
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ollama
import pandas as pd
from transformers import BertTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_alg1 as cfg  # noqa: E402


# ---------------------------------------------------------------------------
# Estimativa de tokens do prompt
# ---------------------------------------------------------------------------
# O Ollama nao expoe endpoint de tokenizacao (/api/tokenize responde 404 na
# versao usada) e o tokenizador do Llama 3.1 e distribuido sob licenca restrita,
# portanto nao ha como contar os tokens do prompt antes de submete-lo.
#
# A estimativa abaixo e usada apenas para escolher o patamar de num_ctx e e
# deliberadamente pessimista: o divisor e menor que a razao caracteres/token
# medida (PT 3,38 e EN 4,57), de modo a superestimar o numero de tokens. A
# ausencia de truncamento nao depende desta estimativa: ela e conferida depois,
# em prompt_eval_count, pela funcao houve_truncamento().
CHARS_POR_TOKEN_CONSERVADOR = {"pt": 3.10, "en": 4.15}
CHARS_POR_TOKEN_MEDIDO = {"pt": 3.38, "en": 4.57}
OVERHEAD_TOKENS = 64          # cabecalho do prompt + template de chat
FOLGA_PATAMAR = 0.95          # o estimado tem de caber em 95% do patamar

MAX_TENTATIVAS = 4            # por chamada, antes de registrar falha
ESPERA_BASE_S = 15


def estimar_tokens_prompt(prompt: str, lang: str) -> int:
    return int(len(prompt) / CHARS_POR_TOKEN_CONSERVADOR[lang]) + OVERHEAD_TOKENS


def escolher_patamar(n_tokens_estimado: int) -> int:
    """Menor patamar de num_ctx que acomoda o prompt estimado com folga."""
    for patamar in cfg.NUM_CTX_PATAMARES:
        if n_tokens_estimado <= patamar * FOLGA_PATAMAR:
            return patamar
    return cfg.NUM_CTX_PATAMARES[-1]


def houve_truncamento(prompt_eval_count: int, num_ctx: int,
                      n_chars_prompt: int, lang: str) -> bool:
    """
    Detecta o truncamento silencioso do Ollama.

    Quando o prompt excede num_ctx, o servidor descarta o excesso e processa
    num_ctx/2 tokens, sem erro. Medicao na maior ata PT: num_ctx = 8192 deu
    prompt_eval_count = 4098; num_ctx = 16384 deu 8194 -- isto e, num_ctx/2 + 2.

    Duas condicoes precisam valer para caracterizar truncamento, porque um
    prompt legitimo pode por coincidencia ter comprimento proximo de num_ctx/2:

      1. prompt_eval_count cai na janela estreita [num_ctx/2, num_ctx/2 + 8];
      2. a razao tokens/caractere observada e muito menor que a medida para o
         idioma -- sinal de que boa parte do texto nao foi processada.
    """
    metade = num_ctx // 2
    if not (metade <= prompt_eval_count <= metade + 8):
        return False
    if n_chars_prompt <= 0:
        return False
    razao_esperada = 1.0 / CHARS_POR_TOKEN_MEDIDO[lang]
    razao_observada = prompt_eval_count / n_chars_prompt
    return razao_observada < 0.80 * razao_esperada


# ---------------------------------------------------------------------------
# Estado do servidor -- procedimento LimparEstado do relatorio
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434"
ESPERA_DESCARGA_S = 30


def modelo_residente() -> list[str]:
    """Modelos atualmente carregados na memoria do servidor."""
    import urllib.request
    with urllib.request.urlopen(f"{OLLAMA_URL}/api/ps", timeout=10) as r:
        return [m["name"] for m in json.load(r).get("models", [])]


def descarregar_modelo() -> None:
    """
    LimparEstado: descarrega o modelo e espera a confirmacao.

    Uma requisicao com keep_alive = 0 e sem prompt instrui o servidor a liberar
    o modelo, o que destroi o cache de prefixo junto. A confirmacao vem de
    /api/ps, para que o estado limpo seja verificado e nao presumido.
    """
    import urllib.request
    corpo = json.dumps({"model": cfg.LLM_MODEL, "keep_alive": 0}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=corpo,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=60).read()
    except Exception:
        pass  # o servidor pode fechar a conexao ao descarregar

    limite = time.time() + ESPERA_DESCARGA_S
    while time.time() < limite:
        if not modelo_residente():
            return
        time.sleep(1)
    raise RuntimeError(
        f"o modelo seguiu residente apos {ESPERA_DESCARGA_S}s; o estado limpo "
        "nao pode ser garantido")


# ---------------------------------------------------------------------------
# Decisao do experimento de determinismo (N_sub)
# ---------------------------------------------------------------------------
def decisao_etapa_a() -> dict:
    """
    Le o relatorio da segunda execucao do experimento de determinismo e devolve
    o N_sub fixado por ele (passo 12 do procedimento TesteDeterminismo).

    Regra: se as N_test submissoes de todas as atas da amostra, nos dois
    idiomas, foram identicas byte a byte, N_sub = 1; senao, N_sub = N_SUB_MAX.
    So vale o relatorio da condicao com estado limpo antes de cada chamada,
    que e o teste como especificado.
    """
    caminho = cfg.RELATORIO_ETAPA_A
    if not caminho.exists():
        raise SystemExit(
            f"ABORTADO: relatorio da etapa A ausente ({caminho}).\n"
            "A etapa B usa o N_sub decidido pela etapa A. Execute antes:\n"
            "  ./rodar_determinismo_frio.sh")
    rel = json.loads(caminho.read_text(encoding="utf-8"))
    if rel.get("condicao") != "cache_frio":
        raise SystemExit(f"ABORTADO: {caminho} nao e da condicao com estado "
                         f"limpo (condicao = {rel.get('condicao')!r}).")
    esperado = cfg.N_ATAS_DETERMINISMO * len(cfg.LANGUAGES)
    if rel.get("n_combinacoes_verificadas") != esperado or rel.get("incompletas"):
        raise SystemExit(f"ABORTADO: a etapa A esta incompleta em {caminho}: "
                         f"{rel.get('n_combinacoes_verificadas')} de {esperado} "
                         "combinacoes verificadas.")
    n_sub = 1 if rel["todas_identicas"] else cfg.N_SUB_MAX
    return {"n_sub": n_sub, "todas_identicas": rel["todas_identicas"],
            "verificado_em": rel.get("verificado_em"),
            "relatorio": str(caminho)}


# ---------------------------------------------------------------------------
# Verificacoes de integridade antes de gerar qualquer coisa
# ---------------------------------------------------------------------------
def versao_ollama() -> str:
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/version",
                                    timeout=5) as r:
            return json.load(r).get("version", "desconhecida")
    except Exception:
        return "desconhecida"


def digest_modelo() -> str:
    try:
        saida = subprocess.run(["ollama", "list"], capture_output=True,
                               text=True, timeout=20).stdout
        for linha in saida.splitlines():
            campos = linha.split()
            if campos and campos[0] == cfg.LLM_MODEL:
                return campos[1]
    except Exception:
        pass
    return "desconhecido"


def verificar_ambiente(log: logging.Logger) -> dict:
    """Falha cedo e alto se qualquer constante do estudo estiver fora de lugar."""
    for lang, prompt in cfg.PROMPT.items():
        sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if sha != cfg.PROMPT_SHA256[lang]:
            raise SystemExit(
                f"ABORTADO: o prompt {lang} foi alterado.\n"
                f"  esperado {cfg.PROMPT_SHA256[lang]}\n"
                f"  obtido   {sha}\n"
                "O prompt e constante do estudo (restricao metodologica 1). "
                "Restaure o texto original antes de gerar."
            )
        if "{m}" not in prompt:
            raise SystemExit(f"ABORTADO: prompt {lang} sem o marcador {{m}}.")

    digest = digest_modelo()
    if digest != cfg.LLM_DIGEST_ESPERADO:
        log.warning("digest do modelo e %s, esperado %s -- o modelo deveria "
                    "estar congelado", digest, cfg.LLM_DIGEST_ESPERADO)

    ambiente = {
        "ollama_version": versao_ollama(),
        "llm_model": cfg.LLM_MODEL,
        "llm_digest": digest,
        "python": platform.python_version(),
        "plataforma": f"{platform.system()} {platform.release()} {platform.machine()}",
        "prompt_sha256": dict(cfg.PROMPT_SHA256),
        "llm_options": dict(cfg.LLM_OPTIONS),
    }
    log.info("ambiente: %s", json.dumps(ambiente, ensure_ascii=False))
    return ambiente


# ---------------------------------------------------------------------------
# Corpus e checkpoint
# ---------------------------------------------------------------------------
def carregar_corpus(lang: str) -> pd.DataFrame:
    df = pd.read_csv(cfg.DATASET_CSV[lang], index_col=0)
    faltando = {cfg.COLUNA_DATA, cfg.COLUNA_TEXTO} - set(df.columns)
    if faltando:
        raise SystemExit(f"ABORTADO: {cfg.DATASET_CSV[lang]} sem colunas {faltando}")
    df = df.copy()
    df["data"] = df[cfg.COLUNA_DATA].astype(int).astype(str)
    if df["data"].duplicated().any():
        dup = df.loc[df["data"].duplicated(), "data"].tolist()
        raise SystemExit(f"ABORTADO: datas duplicadas no corpus {lang}: {dup}")
    if len(df) != cfg.N_ATAS_ESPERADO:
        raise SystemExit(f"ABORTADO: corpus {lang} tem {len(df)} atas, "
                         f"esperado {cfg.N_ATAS_ESPERADO}")
    return df[["data", cfg.COLUNA_TEXTO]]


def carregar_checkpoint(lang: str) -> tuple[set[tuple[str, int]], dict[str, int]]:
    """Devolve as chaves (data, j) ja geradas e o num_ctx usado por ata."""
    caminho = cfg.arquivo_resumos(lang)
    feitos: set[tuple[str, int]] = set()
    ctx_por_ata: dict[str, int] = {}
    if not caminho.exists():
        return feitos, ctx_por_ata
    with caminho.open(encoding="utf-8") as f:
        for n_linha, linha in enumerate(f, 1):
            linha = linha.strip()
            if not linha:
                continue
            try:
                r = json.loads(linha)
            except json.JSONDecodeError:
                # Linha truncada por interrupcao durante a escrita: so pode ser
                # a ultima. Ignorar faz o par (data, j) ser regerado.
                print(f"aviso: linha {n_linha} de {caminho.name} ilegivel, "
                      "sera regerada", file=sys.stderr)
                continue
            feitos.add((r["data"], int(r["j"])))
            ctx_por_ata.setdefault(r["data"], int(r["num_ctx"]))
    return feitos, ctx_por_ata


def anexar_resumo(lang: str, registro: dict) -> None:
    """Grava um resumo e forca a descida ao disco, para o checkpoint ser real."""
    caminho = cfg.arquivo_resumos(lang)
    with caminho.open("a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Chamada ao LLM
# ---------------------------------------------------------------------------
def chamar_llm(prompt: str, seed: int, num_ctx: int,
               keep_alive=None) -> dict:
    """
    Uma submissao ao LLM, em sessao nova.

    A lista de mensagens tem um unico elemento: nao ha historico de conversa
    entre submissoes (cada chamada e feita em sessao nova, no Algoritmo 1 e no
    TesteDeterminismo). O cache de prefixo do servidor nao e historico de
    conversa, mas altera o caminho de execucao; ver a nota do Algoritmo 1 sobre
    a limpeza do estado.

    keep_alive controla por quanto tempo o modelo permanece residente depois da
    resposta. O valor 0 faz o servidor descarrega-lo imediatamente, de modo que a
    chamada seguinte parta de estado limpo; e o que o experimento de
    determinismo usa. Passando None, vale cfg.KEEP_ALIVE.
    """
    opcoes = dict(cfg.LLM_OPTIONS)
    opcoes["seed"] = seed
    opcoes["num_ctx"] = num_ctx

    t0 = time.time()
    r = ollama.chat(
        model=cfg.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options=opcoes,
        keep_alive=cfg.KEEP_ALIVE if keep_alive is None else keep_alive,
    )
    elapsed = time.time() - t0

    return {
        "resumo": r["message"]["content"],
        "tempo_s": round(elapsed, 2),
        "prompt_eval_count": int(r.get("prompt_eval_count") or 0),
        "eval_count": int(r.get("eval_count") or 0),
        "done_reason": r.get("done_reason") or "",
    }


def chamar_com_retentativa(prompt: str, seed: int, num_ctx: int,
                           log: logging.Logger, keep_alive=None) -> dict:
    """
    Retenta falhas transitorias do servidor. Nao retenta por conteudo.

    Antes de cada retentativa o modelo e descarregado: uma chamada que falhou no
    meio pode ter deixado parte do prompt desta ata no cache, e repeti-la com o
    prefixo reaproveitado seria outro caminho de execucao (nota v).
    """
    ultimo_erro = None
    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            return chamar_llm(prompt, seed, num_ctx, keep_alive=keep_alive)
        except Exception as e:                       # noqa: BLE001
            ultimo_erro = e
            espera = ESPERA_BASE_S * tentativa
            log.warning("falha na chamada (tentativa %d/%d): %s -- nova "
                        "tentativa em %ds", tentativa, MAX_TENTATIVAS, e, espera)
            time.sleep(espera)
            try:
                descarregar_modelo()
            except Exception as e2:                  # noqa: BLE001
                log.warning("nao foi possivel limpar o estado antes da "
                            "retentativa: %s", e2)
    raise RuntimeError(f"{MAX_TENTATIVAS} tentativas falharam: {ultimo_erro}")


# ---------------------------------------------------------------------------
# Laco principal
# ---------------------------------------------------------------------------
def gerar_idioma(lang: str, args, log: logging.Logger, ambiente: dict) -> dict:
    df = carregar_corpus(lang)

    if args.atas:
        alvo = {a.strip() for a in args.atas.split(",")}
        desconhecidas = alvo - set(df["data"])
        if desconhecidas:
            raise SystemExit(f"ABORTADO: atas inexistentes no corpus {lang}: "
                             f"{sorted(desconhecidas)}")
        df = df[df["data"].isin(alvo)]

    tokenizer = BertTokenizerFast.from_pretrained(
        str(cfg.TOKENIZER_PATH[lang]), local_files_only=True)

    feitos, ctx_por_ata = carregar_checkpoint(lang)
    n_sub = args.n_sub

    # Plano: patamar de num_ctx por ata. Em modo fixo, o valor unico da entrada
    # LLM do Algoritmo 1 (num_ctx = 32768).
    plano = []
    for _, linha in df.iterrows():
        data = linha["data"]
        texto = linha[cfg.COLUNA_TEXTO]
        prompt = cfg.PROMPT[lang].format(m=texto)
        if args.num_ctx_modo == "fixo":
            patamar = cfg.NUM_CTX
        else:
            patamar = ctx_por_ata.get(
                data, escolher_patamar(estimar_tokens_prompt(prompt, lang)))
        pendentes = [j for j in range(1, n_sub + 1) if (data, j) not in feitos]
        plano.append({"data": data, "prompt": prompt, "n_chars_ata": len(texto),
                      "patamar": patamar, "pendentes": pendentes})

    # Ordenar por patamar agrupa as atas que compartilham num_ctx. Trocar
    # num_ctx obriga o Ollama a recarregar o modelo, e a ordem de processamento
    # nao afeta resultado algum: cada ata e independente e a saida e chaveada
    # por (lang, data, j).
    plano.sort(key=lambda p: (p["patamar"], p["data"]))

    total_pendente = sum(len(p["pendentes"]) for p in plano)
    por_patamar = {}
    for p in plano:
        por_patamar[p["patamar"]] = por_patamar.get(p["patamar"], 0) + 1
    log.info("idioma %s: %d atas, %d resumos pendentes de %d; "
             "distribuicao de num_ctx por ata: %s",
             lang, len(plano), total_pendente, len(plano) * n_sub,
             dict(sorted(por_patamar.items())))

    if args.dry_run:
        return {"lang": lang, "dry_run": True, "pendentes": total_pendente,
                "patamares": dict(sorted(por_patamar.items()))}

    gerados = 0
    falhas = []
    excedentes = 0
    t_inicio = time.time()

    for p in plano:
        data, prompt, patamar = p["data"], p["prompt"], p["patamar"]
        if not p["pendentes"]:
            continue
        patamar_confirmado = data in ctx_por_ata

        for j in p["pendentes"]:
            # O app do Ollama se atualiza sozinho (aconteceu em 16/10/2026,
            # 0.35.1 -> 0.40.5). Se o servidor mudar de versao no meio da
            # geracao, para aqui: retomar e uma decisao, nao um efeito colateral.
            versao_agora = versao_ollama()
            if versao_agora != ambiente["ollama_version"]:
                log.error("o Ollama mudou de versao durante a geracao (%s -> "
                          "%s). Geracao interrompida; os resumos ja gravados "
                          "estao preservados. Para continuar na versao nova, "
                          "rode de novo ./rodar_geracao.sh.",
                          ambiente["ollama_version"], versao_agora)
                raise SystemExit(3)
            seed = cfg.semente(j)
            # Passo 6 do Algoritmo 1: com mais de uma submissao por ata, o estado
            # e limpo antes de cada uma, para que todas sigam o mesmo caminho de
            # execucao.
            keep_alive = None
            if n_sub > 1:
                descarregar_modelo()
                keep_alive = 0
            try:
                res = chamar_com_retentativa(prompt, seed, patamar, log,
                                             keep_alive=keep_alive)
            except Exception as e:                   # noqa: BLE001
                log.error("FALHA definitiva em (%s, %s, j=%d): %s",
                          lang, data, j, e)
                falhas.append({"lang": lang, "data": data, "j": j,
                               "erro": str(e)})
                continue

            # Conferencia de truncamento: feita uma vez por ata, na primeira
            # submissao efetivamente gerada. Nas seguintes o prompt e identico e
            # o num_ctx e o mesmo, de modo que o resultado da conferencia vale
            # para todas -- e o cache de prefixo tornaria prompt_eval_count
            # menor que o real, inutilizando a verificacao.
            if not patamar_confirmado:
                while (houve_truncamento(res["prompt_eval_count"], patamar,
                                         len(prompt), lang)
                       and patamar < cfg.NUM_CTX_PATAMARES[-1]):
                    proximo = next(t for t in cfg.NUM_CTX_PATAMARES if t > patamar)
                    log.warning("ata %s: prompt truncado em num_ctx=%d "
                                "(prompt_eval_count=%d); elevando para %d e "
                                "descartando o resumo truncado",
                                data, patamar, res["prompt_eval_count"], proximo)
                    patamar = proximo
                    res = chamar_com_retentativa(prompt, seed, patamar, log,
                                                 keep_alive=keep_alive)

                truncado_no_teto = houve_truncamento(
                    res["prompt_eval_count"], patamar, len(prompt), lang)
                if truncado_no_teto:
                    log.error("ata %s: truncamento persiste no maior patamar "
                              "(num_ctx=%d, prompt_eval_count=%d). O resumo "
                              "sera marcado como suspeito.",
                              data, patamar, res["prompt_eval_count"])
                patamar_confirmado = True
                ctx_por_ata[data] = patamar
            else:
                truncado_no_teto = False

            resumo = res["resumo"]
            n_tok = len(tokenizer.encode(
                resumo, add_special_tokens=cfg.ADD_SPECIAL_TOKENS))
            excede = n_tok > cfg.L_MAX
            if excede:
                excedentes += 1

            registro = {
                "lang": lang,
                "data": data,
                "j": j,
                "seed": seed,
                "resumo": resumo,
                "n_tokens_bert": n_tok,
                "excede_l_max": excede,
                # Excesso sobre L_max; 0 quando cabe. Relatado na lista Acima.
                "excesso": max(0, n_tok - cfg.L_MAX),
                "n_chars_resumo": len(resumo),
                "n_chars_ata": p["n_chars_ata"],
                "num_ctx": patamar,
                "prompt_eval_count": res["prompt_eval_count"],
                "eval_count": res["eval_count"],
                "done_reason": res["done_reason"],
                "truncamento_suspeito": bool(truncado_no_teto),
                "tempo_s": res["tempo_s"],
                "llm_model": cfg.LLM_MODEL,
                "llm_digest": ambiente["llm_digest"],
                "ollama_version": ambiente["ollama_version"],
                "prompt_sha256": cfg.PROMPT_SHA256[lang],
                "gerado_em": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"),
            }
            anexar_resumo(lang, registro)
            gerados += 1

            decorrido = time.time() - t_inicio
            media = decorrido / gerados
            restantes = total_pendente - gerados
            log.info("[%s %d/%d] ata %s j=%d ctx=%d pec=%d tok_bert=%d%s "
                     "%.0fs | media %.0fs | restam %d (~%.1f h)",
                     lang, gerados, total_pendente, data, j, patamar,
                     res["prompt_eval_count"], n_tok,
                     " EXCEDE" if excede else "", res["tempo_s"], media,
                     restantes, restantes * media / 3600)

    if falhas:
        caminho_falhas = cfg.LOG_DIR / f"falhas_{lang}.jsonl"
        with caminho_falhas.open("a", encoding="utf-8") as f:
            for item in falhas:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        log.error("%d falhas registradas em %s -- reexecute o script para "
                  "retomar apenas elas", len(falhas), caminho_falhas)

    resumo_exec = {
        "lang": lang,
        "resumos_gerados_nesta_execucao": gerados,
        "falhas": len(falhas),
        "excedem_l_max_nesta_execucao": excedentes,
        "tempo_total_s": round(time.time() - t_inicio, 1),
        "num_ctx_modo": args.num_ctx_modo,
        "patamares": dict(sorted(por_patamar.items())),
    }
    log.info("concluido %s: %s", lang, json.dumps(resumo_exec, ensure_ascii=False))
    return resumo_exec


def construir_logger(lang_tag: str) -> logging.Logger:
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("alg1.geracao")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(cfg.LOG_DIR / f"geracao_{lang_tag}.log",
                             encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=[*cfg.LANGUAGES, "todos"], default="todos")
    ap.add_argument("--atas", default=None,
                    help="lista de datas YYYYMMDD separadas por virgula; "
                         "restringe a execucao (usado no piloto)")
    ap.add_argument("--n-sub", type=int, default=None,
                    help="submissoes por ata; por omissao, o valor decidido "
                         "pela etapa A. Passar um valor diferente aborta.")
    ap.add_argument("--num-ctx-modo", choices=["adaptativo", "fixo"],
                    default="adaptativo",
                    help="adaptativo: menor patamar que acomode a ata sem "
                         "truncar; fixo: sempre o valor de E4")
    ap.add_argument("--dry-run", action="store_true",
                    help="apenas relata o plano, sem chamar o LLM")
    args = ap.parse_args()

    cfg.RESUMOS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

    langs = list(cfg.LANGUAGES) if args.lang == "todos" else [args.lang]
    log = construir_logger(args.lang)

    # N_sub e entrada do Algoritmo 1, fixada pelo experimento de determinismo,
    # nao escolha do operador.
    decisao = decisao_etapa_a()
    if decisao["n_sub"] != cfg.N_SUB:
        raise SystemExit(
            f"ABORTADO: a etapa A decidiu N_sub = {decisao['n_sub']}, mas "
            f"config_alg1.N_SUB = {cfg.N_SUB}. Alinhe a constante ao relatorio "
            f"{decisao['relatorio']}.")
    if args.n_sub is None:
        args.n_sub = decisao["n_sub"]
    elif args.n_sub != decisao["n_sub"]:
        raise SystemExit(
            f"ABORTADO: --n-sub {args.n_sub} contraria a decisao da etapa A "
            f"(N_sub = {decisao['n_sub']}, {decisao['relatorio']}).")

    log.info("inicio | lang=%s atas=%s n_sub=%d (decidido pela etapa A em %s) "
             "num_ctx_modo=%s", args.lang, args.atas or "todas", args.n_sub,
             decisao["verificado_em"], args.num_ctx_modo)

    ambiente = verificar_ambiente(log)
    ambiente["decisao_etapa_a"] = decisao

    # Passo 1 do Algoritmo 1 (nota sobre a limpeza do estado): nenhuma parte do
    # prompt da ata em andamento pode estar no cache quando a chamada comeca.
    # Uma execucao anterior interrompida no meio de uma ata deixaria esse
    # prefixo residente; descarregar na partida elimina o caso.
    if not args.dry_run:
        descarregar_modelo()
        log.info("estado do servidor limpo na partida (nenhum modelo residente)")

    registros = []
    t0 = time.time()
    for lang in langs:
        registros.append(gerar_idioma(lang, args, log, ambiente))

    if not args.dry_run:
        registro_exec = {
            "etapa": "geracao",
            "inicio_utc": datetime.fromtimestamp(t0, timezone.utc).isoformat(
                timespec="seconds"),
            "fim_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tempo_total_s": round(time.time() - t0, 1),
            "argumentos": vars(args),
            "ambiente": ambiente,
            "por_idioma": registros,
        }
        caminho = cfg.OUTPUT_DIR / "registro_execucao_geracao.jsonl"
        with caminho.open("a", encoding="utf-8") as f:
            f.write(json.dumps(registro_exec, ensure_ascii=False) + "\n")
        log.info("registro de execucao anexado em %s", caminho)


if __name__ == "__main__":
    main()
