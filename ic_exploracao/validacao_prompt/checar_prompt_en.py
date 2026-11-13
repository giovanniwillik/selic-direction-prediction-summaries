"""
Checagem TECNICA do orcamento de tokens do prompt EN (traducao literal do prompt
PT ja selecionado nas atas pos-mar/2025). Nao e etapa de selecao de prompt.
"""
import os, json, time
import ollama, pandas as pd
from transformers import BertTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
BERT_EN = os.path.join(HERE, "..", "..", "model", "pre-trained", "bert-base-cased")
DS_EN = os.path.join(HERE, "..", "..", "dataset", "dataset_en.csv")

PROMPT_EN = """Summarize the document below by filling in the sections listed.
Use at most 3 sentences per section and restrict yourself to the content of the
document. If a section is not addressed in the text, write "Not addressed".

1. External environment:
2. Economic activity and labour market:
3. Inflation and expectations:
4. Financial and credit conditions:
5. Balance of risks:

Document:
{texto}

Summary:"""

tok = BertTokenizer.from_pretrained(BERT_EN, local_files_only=True)
df = pd.read_csv(DS_EN, index_col=0)

out = []
for idx in (0, 40, 90):
    texto = df.iloc[idx]["texto_ata"]
    t0 = time.time()
    r = ollama.chat(model="llama3.1:8b",
                    messages=[{"role": "user",
                               "content": PROMPT_EN.format(texto=texto)}],
                    options={"temperature": 0.0, "top_p": 1.0,
                             "num_predict": 900, "seed": 7, "num_ctx": 16384})
    s = r["message"]["content"]
    n = len(tok.encode(s, add_special_tokens=True))
    out.append({"idx": int(idx), "data": int(df.iloc[idx]["data"]),
                "n_tokens_bert": n, "cabe_512": n <= 512,
                "tempo_s": round(time.time() - t0, 1), "resumo": s})
    print(f"idx={idx} data={df.iloc[idx]['data']} tokens={n} cabe512={n<=512}")

json.dump(out, open(os.path.join(HERE, "resultado_prompt_en.json"), "w"),
          ensure_ascii=False, indent=1)
print("salvo")
