# IC — Predizendo a Taxa SELIC a partir de Resumos de Atas do COPOM

Iniciação Científica: extensão de Forti (2025), introduzindo uma etapa de sumarização
com LLM local antes da classificação com BERT/BERTimbau. Visão geral, créditos e
instruções de reprodução no `README.md` da raiz do repositório.

## Pipeline da IC

```
Ata completa (PT/EN)  →  LLM local (sumarização)  →  Resumo (≤512 tokens)  →  BERT/BERTimbau (fine-tuning)  →  Predição (alta/manut/queda)
```

O trabalho é dividido em dois algoritmos independentes, especificados em
`latex/relatorio_ic_parcial.tex`:

- **Algoritmo 1 — sumarização.** Gera um resumo por ata, nos dois idiomas, com o
  `llama3.1:8b`. O número de submissões por ata (`N_sub = 1`) foi fixado antes,
  por um experimento de determinismo, que mediu variância nula com
  `temperature = 0`. Concluído: 404 resumos.
- **Algoritmo 2 — classificação.** Fine-tuning de BERT/BERTimbau sobre os resumos,
  replicando o protocolo de Forti (2025). Especificado no relatório, **ainda não
  executado**.

## Estrutura

```
ic_exploracao/
├── README.md                     # Este arquivo
├── latex/
│   ├── relatorio_ic_parcial.tex  # Relatório parcial da IC: especificação e resultados
│   └── sbc-template.sty          # Estilo da SBC usado pelo relatório
├── algoritmo1/                   # Implementação do Algoritmo 1 e do experimento de determinismo
├── resumos/                      # Corpus de resumos e resumos do experimento de determinismo
├── output/algoritmo1/            # Decisão de N_sub, lista Acima, similaridades, registros de execução
├── validacao_prompt/             # Seleção do prompt em atas fora do recorte do estudo
├── setup_ambiente.sh             # Script de instalação do ambiente (macOS)
└── requirements.txt              # Dependências Python (versões fixadas)
```

## Quick Start

```bash
# 1. Ambiente
chmod +x setup_ambiente.sh
./setup_ambiente.sh
source .venv/bin/activate

# 2. Garantir que o Ollama esta rodando
ollama list            # se nao responder: ollama serve

# 3. Algoritmo 1 -- ver algoritmo1/README.md
cd algoritmo1
./rodar_geracao.sh     # gera o corpus; le o N_sub fixado pelo experimento de determinismo
python status.py
```

## Requisitos de Hardware

- macOS Apple Silicon (M4 recomendado)
- 24GB RAM (mínimo 16GB para modelo 8B)
- ~12GB de disco (modelos + venv)
- Python 3.10+
