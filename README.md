# Predizendo a taxa SELIC a partir de resumos de atas do COPOM

Iniciação Científica de Giovanni Willik Del Piccolo (EACH/USP). O trabalho
estende o TCC de Forti (2025), que previu a direção da Selic na reunião
seguinte com BERT/BERTimbau sobre as atas do COPOM truncadas por *head+tail*
(255 + 255 tokens) e obteve F1-macro ≈ 0,25. A hipótese é que um resumo gerado
por um LLM local cabe nos 512 tokens do BERT e preserva mais informação do que
o truncamento.

```
ata (PT/EN) -> Algoritmo 1: llama3.1:8b -> resumo -> Algoritmo 2: BERT/BERTimbau -> {queda, manutenção, alta}
```

## Créditos

O corpus de atas, o notebook de fine-tuning e os resultados do *baseline* são
de Ryan A. Forti, do TCC *Predição da direção futura da taxa SELIC a partir de
atas de reunião do COPOM* (EACH/USP, 2025), publicados em
[RyanForti/Selic-Direction-Prediciton](https://github.com/RyanForti/Selic-Direction-Prediciton).
Aquele repositório não declara licença, por isso nenhum arquivo dele é
redistribuído aqui: o passo 1 de "Como reproduzir" mostra como obter os dados
de entrada. O restante foi produzido na IC, exceto o estilo da SBC (ver
"Licença").

## Estrutura

```
.
├── README.md, LICENSE, LICENSE-CC-BY-4.0.txt
└── ic_exploracao/
    ├── latex/
    │   ├── relatorio_ic_parcial.tex   relatório parcial: especificação e resultados
    │   └── sbc-template.sty           estilo da SBC usado pelo relatório
    ├── algoritmo1/                    Algoritmo 1 e experimento de determinismo (ver o README da pasta)
    ├── resumos/                       corpus de resumos (202 PT + 202 EN) e resumos do experimento
    ├── output/algoritmo1/             decisão de N_sub, resumos acima de 512 tokens, similaridades, registros
    ├── validacao_prompt/              escolha do prompt em atas fora do recorte do estudo
    ├── setup_ambiente.sh              instalação do ambiente (macOS)
    └── requirements.txt               dependências Python, com versões fixadas
```

Pastas criadas na reprodução e fora do controle de versão: `dataset/` (atas de
entrada, do repositório de Forti), `model/pre-trained/` (pesos do BERT, do
BERTimbau e do codificador de sentenças) e `ic_exploracao/.venv/`.

## Como reproduzir

Os scripts foram escritos para macOS com Apple Silicon (usam Homebrew e
`caffeinate`), com 16 GB de RAM no mínimo e Python 3.10.

**1. Dados de entrada.** O código lê `dataset/dataset_pt.csv` e
`dataset/dataset_en.csv` (202 atas cada) na raiz do repositório. O commit
abaixo é o que o estudo usou.

```bash
git clone https://github.com/giovanniwillik/selic-direction-prediction-summaries.git
cd selic-direction-prediction-summaries
git clone https://github.com/RyanForti/Selic-Direction-Prediciton.git ../Selic-Direction-Prediciton
git -C ../Selic-Direction-Prediciton checkout b3b7642bbe30241afe651c86cea4b2d394385437
mkdir -p dataset
cp ../Selic-Direction-Prediciton/dataset/dataset_pt.csv ../Selic-Direction-Prediciton/dataset/dataset_en.csv dataset/
shasum -a 256 dataset/dataset_pt.csv dataset/dataset_en.csv
# de9b00a2d3b19219272b977449e0d9e44fbed59a71cb126a59acee9c4be13f7c  dataset/dataset_pt.csv
# 3b215cdea00a11378a2342d2d203df15a97d2463016d3c30300dfa3064274c5f  dataset/dataset_en.csv
```

**2. Ambiente Python e pesos.** `setup_ambiente.sh` instala o Ollama pelo
Homebrew, baixa os modelos do Ollama, cria `ic_exploracao/.venv`, instala
`requirements.txt` e salva `bert-base-cased` e
`neuralmind/bert-base-portuguese-cased` em `model/pre-trained/` (cerca de
420 MB cada). O codificador `paraphrase-multilingual-MiniLM-L12-v2` é baixado
do Hugging Face na primeira execução de `analisar_variancia.py`, se não estiver
em `model/pre-trained/`.

```bash
./ic_exploracao/setup_ambiente.sh
source ic_exploracao/.venv/bin/activate
```

**3. Ollama.** O estudo usou `llama3.1:8b`, quantização Q4_K_M, digest
`46e0c10c039e`, no Ollama 0.40.5 (partes anteriores em 0.34.2 e 0.35.1).
Confira o digest na coluna `ID` de `ollama list`; `gerar_resumos.py` avisa se
ele for outro. Desative a atualização automática do app do Ollama antes de
gerar: a geração para se a versão do servidor mudar.

```bash
ollama pull llama3.1:8b
ollama list            # ID deve começar por 46e0c10c039e
```

**4. Algoritmo 1.** Os comandos, os arquivos e as decisões de implementação
estão em `ic_exploracao/algoritmo1/README.md`.

```bash
cd ic_exploracao/algoritmo1

# Experimento de determinismo (já executado; refazer só se modelo, Ollama ou prompt mudarem)
python determinismo.py --selecionar
./rodar_determinismo_frio.sh             # 200 chamadas, ~8 h
python determinismo.py --verificar-frio  # decide N_sub

# Geração do corpus
./rodar_geracao.sh                       # ~20 h para as 404 atas
python status.py
python analisar_variancia.py --fonte corpus
```

A geração é retomável e pula as atas que já estão em
`ic_exploracao/resumos/resumos_{pt,en}.jsonl`. Como o corpus publicado já está
completo, para regerá-lo do zero mova esses dois arquivos para fora da pasta
antes de rodar.

**Relatório.** `ic_exploracao/latex/relatorio_ic_parcial.tex` compila com
pdfLaTeX junto com `sbc-template.sty` (por exemplo, no Overleaf), em duas
passadas.

## Estado atual

- **Algoritmo 1 (sumarização): concluído.** 404 resumos, 202 em português e
  202 em inglês, gerados em 19,7 h, sem falhas.
- **Algoritmo 2 (classificação com BERT/BERTimbau): ainda não executado.** A
  especificação aguarda a aprovação do orientador e entrará na seção 6.1 do
  relatório, que está em branco nesta versão, assim como as de resultados da
  classificação e de resultados gerais.

## Licença

- Código (`*.py`, `*.sh`): MIT, em `LICENSE`.
- Textos, resumos gerados e arquivos de resultados: CC BY 4.0, em
  `LICENSE-CC-BY-4.0.txt`.
- `ic_exploracao/latex/sbc-template.sty` é o modelo da Sociedade Brasileira de
  Computação (Jomi Hübner e Rafael Bordini) e não é coberto por estas licenças.
- Os resumos foram gerados com o Llama 3.1 8B, da Meta, cujo uso segue a Llama
  3.1 Community License. As atas são publicações do Banco Central do Brasil.

## Como citar

```bibtex
@misc{delpiccolo2026selic,
  author       = {Del Piccolo, Giovanni Willik},
  title        = {Predizendo a taxa {SELIC} a partir de resumos de atas do {COPOM}},
  year         = {2026},
  howpublished = {\url{https://github.com/giovanniwillik/selic-direction-prediction-summaries}},
  note         = {Iniciação Científica, EACH/USP. Relatório parcial}
}
```

Cite também o trabalho de origem dos dados e do *baseline*:

```bibtex
@misc{forti2025selic,
  author       = {Forti, Ryan A.},
  title        = {Predição da direção futura da taxa {SELIC} a partir de atas de reunião do {COPOM}},
  year         = {2025},
  howpublished = {Trabalho de Conclusão de Curso, EACH/USP},
  note         = {\url{https://github.com/RyanForti/Selic-Direction-Prediciton}}
}
```
