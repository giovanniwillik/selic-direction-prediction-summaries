"""
Validacao de prompts de sumarizacao em atas do COPOM posteriores a mar/2025.
Conjunto de validacao disjunto do dataset do trabalho base (jan/2002 - mar/2025).
"""
import json, time, re, sys
import ollama
from transformers import BertTokenizer

MODEL = "llama3.1:8b"
ATAS = json.load(open("atas_pt_pos_mar2025.json"))

# --- Candidatos de prompt (PT) ---
P = {}

P["A_neutro"] = """Resuma o documento a seguir em no maximo 350 palavras.
O resumo deve preservar os temas discutidos e as avaliacoes apresentadas, sem
acrescentar informacao que nao esteja no texto original.

Documento:
{texto}

Resumo:"""

P["B_tematico"] = """Resuma o documento a seguir em no maximo 350 palavras.
Cubra, quando presentes no texto, os seguintes temas: cenario externo, atividade
economica, mercado de trabalho, inflacao corrente e expectativas de inflacao,
condicoes financeiras e de credito, e o balanco de riscos descrito.
Restrinja-se ao conteudo do documento, sem inferencias ou conclusoes proprias.

Documento:
{texto}

Resumo:"""

P["C_estruturado"] = """Resuma o documento a seguir preenchendo as secoes abaixo.
Use no maximo 3 frases por secao e restrinja-se ao conteudo do documento.
Se uma secao nao for abordada no texto, escreva "Nao abordado".

1. Cenario externo:
2. Atividade economica e mercado de trabalho:
3. Inflacao e expectativas:
4. Condicoes financeiras e de credito:
5. Balanco de riscos:

Documento:
{texto}

Resumo:"""

P["D_extrativo"] = """Do documento a seguir, selecione os 10 trechos mais
informativos sobre a avaliacao economica apresentada e reproduza-os literalmente,
na ordem em que aparecem, um por linha, sem alterar as palavras e sem comentarios.

Documento:
{texto}

Trechos:"""

# --- Termos que indicariam antecipacao da decisao (vies / vazamento) ---
TERMOS_VIES = [
    "proxima reuniao", "proximo passo", "proximos passos", "devera elevar",
    "devera reduzir", "devera manter", "provavelmente", "espera-se que a taxa",
    "recomenda", "sugere elevar", "sugere reduzir", "previsao para a selic",
    "deve subir", "deve cair", "tendencia de alta da taxa", "tendencia de queda da taxa",
    "hawkish", "dovish", "conclui-se que", "portanto o comite ira",
]

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_BERT_PT = os.path.join(_HERE, "..", "..", "model", "pre-trained",
                        "bert-base-portuguese-cased")
tok = BertTokenizer.from_pretrained(_BERT_PT, local_files_only=True)

def gerar(prompt_tpl, texto, seed):
    t0 = time.time()
    r = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt_tpl.format(texto=texto)}],
        options={"temperature": 0.0, "top_p": 1.0, "num_predict": 900,
                 "seed": seed, "num_ctx": 16384},
    )
    return r["message"]["content"], time.time() - t0

def avaliar(resumo, original):
    n_tok = len(tok.encode(resumo, add_special_tokens=True))
    low = resumo.lower()
    vies = [t for t in TERMOS_VIES if t in low]
    # cobertura tematica simples
    temas = {
        "externo": ["externo", "global", "internacional"],
        "atividade": ["atividade", "pib", "consumo", "investimento"],
        "trabalho": ["trabalho", "emprego", "desemprego", "salari"],
        "inflacao": ["inflacao", "ipca", "precos"],
        "expectativas": ["expectativa", "projec"],
        "credito": ["credito", "financeir", "juros"],
        "riscos": ["risco", "balanco de riscos"],
    }
    cob = sum(1 for k, vs in temas.items() if any(v in low for v in vs))
    return {"n_tokens_bert": n_tok, "cabe_512": n_tok <= 512,
            "termos_vies": vies, "cobertura_temas": f"{cob}/7",
            "n_chars": len(resumo)}

if __name__ == "__main__":
    amostra = [a for a in ATAS if a["nro"] in (270, 274, 279)]
    saida = {}
    for nome, tpl in P.items():
        print(f"\n{'='*70}\nPROMPT {nome}\n{'='*70}")
        saida[nome] = []
        for a in amostra:
            resumo, dt = gerar(tpl, a["texto"], seed=7)
            m = avaliar(resumo, a["texto"])
            m.update({"ata": a["nro"], "tempo_s": round(dt, 1), "resumo": resumo})
            saida[nome].append(m)
            print(f"  ata {a['nro']}: {m['n_tokens_bert']} tok "
                  f"(cabe512={m['cabe_512']}) cobertura={m['cobertura_temas']} "
                  f"vies={m['termos_vies']} {dt:.0f}s")
        json.dump(saida, open("resultado_prompts.json", "w"),
                  ensure_ascii=False, indent=1)
    print("\nsalvo em resultado_prompts.json")
