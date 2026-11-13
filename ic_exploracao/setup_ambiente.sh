#!/bin/bash

# Setup do ambiente para IC — Sumarização de Atas do COPOM com LLM Local
# Pré-requisito: macOS com Homebrew instalado

set -e

echo "=========================================="
echo " IC — Setup do Ambiente de Testes"
echo " Predizendo a SELIC a partir de resumos"
echo "=========================================="

# 1. Instalar Ollama
echo ""
echo "[1/6] Verificando Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "  → Instalando Ollama via Homebrew..."
    brew install ollama
else
    echo "  ✓ Ollama já instalado"
fi

# 2. Baixar modelos LLM (Ollama)
echo ""
echo "[2/6] Baixando modelos LLM (Ollama)..."

# Iniciar o servidor Ollama em background se não estiver rodando
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  → Iniciando servidor Ollama em background..."
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    sleep 3  # Aguardar o servidor subir
else
    echo "  ✓ Servidor Ollama já está rodando"
    OLLAMA_PID=""
fi

echo "  → llama3.2:3b (modelo rápido para desenvolvimento — ~2GB)..."
ollama pull llama3.2:3b

echo "  → llama3.1:8b (modelo principal para sumarização — ~4.7GB)..."
ollama pull llama3.1:8b

# Parar servidor que nós iniciamos (usuário pode iniciar manualmente depois)
if [ -n "$OLLAMA_PID" ]; then
    echo "  → Parando servidor Ollama temporário..."
    kill $OLLAMA_PID 2>/dev/null || true
fi

# 3. Criar ambiente virtual Python
echo ""
echo "[3/6] Criando ambiente virtual Python..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
MODELS_DIR="$PROJECT_ROOT/model/pre-trained"

if [ -d "$VENV_DIR" ]; then
    echo "  ✓ Ambiente virtual já existe em $VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
    echo "  ✓ Ambiente virtual criado"
fi

# 4. Instalar dependências Python
echo ""
echo "[4/6] Instalando dependências Python..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements.txt" -q

# 5. Baixar modelos BERT e BERTimbau (necessários para classificação)
echo ""
echo "[5/6] Baixando modelos BERT/BERTimbau..."

BERT_EN_DIR="$MODELS_DIR/bert-base-cased"
BERT_PT_DIR="$MODELS_DIR/bert-base-portuguese-cased"

mkdir -p "$BERT_EN_DIR"
mkdir -p "$BERT_PT_DIR"

# Verificar se BERT (inglês) já está baixado
if [ -z "$(ls -A "$BERT_EN_DIR" 2>/dev/null)" ]; then
    echo "  → Baixando bert-base-cased (inglês, ~420MB)..."
    python3 -c "
from transformers import BertTokenizer, BertForSequenceClassification
print('    Baixando tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer.save_pretrained('$BERT_EN_DIR')
print('    Baixando modelo...')
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3)
model.save_pretrained('$BERT_EN_DIR')
print('    ✓ bert-base-cased salvo em $BERT_EN_DIR')
"
else
    echo "  ✓ bert-base-cased já existe"
fi

# Verificar se BERTimbau (português) já está baixado
if [ -z "$(ls -A "$BERT_PT_DIR" 2>/dev/null)" ]; then
    echo "  → Baixando bert-base-portuguese-cased / BERTimbau (~420MB)..."
    python3 -c "
from transformers import BertTokenizer, BertForSequenceClassification
print('    Baixando tokenizer...')
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
tokenizer.save_pretrained('$BERT_PT_DIR')
print('    Baixando modelo...')
model = BertForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased', num_labels=3)
model.save_pretrained('$BERT_PT_DIR')
print('    ✓ BERTimbau salvo em $BERT_PT_DIR')
"
else
    echo "  ✓ bert-base-portuguese-cased já existe"
fi

# 6. Verificar instalação
echo ""
echo "[6/6] Verificando instalação..."
python3 -c "
import ollama
import torch
import transformers
import pandas as pd
import sklearn
from importlib.metadata import version as pkg_version
print('  ✓ ollama:', pkg_version('ollama'))
print('  ✓ torch:', torch.__version__)
print('  ✓ transformers:', transformers.__version__)
print('  ✓ pandas:', pd.__version__)
print('  ✓ scikit-learn:', sklearn.__version__)
print('  ✓ MPS (GPU Apple Silicon):', torch.backends.mps.is_available())

import os
bert_en = '$BERT_EN_DIR'
bert_pt = '$BERT_PT_DIR'
en_ok = os.path.exists(os.path.join(bert_en, 'config.json'))
pt_ok = os.path.exists(os.path.join(bert_pt, 'config.json'))
print(f'  ✓ BERT (EN) disponível: {en_ok}')
print(f'  ✓ BERTimbau (PT) disponível: {pt_ok}')
"

echo ""
echo ""
echo "=========================================="
echo " Setup concluído!"
echo "=========================================="
echo ""
echo " Modelos instalados:"
echo "   • Ollama: llama3.2:3b (testes rápidos), llama3.1:8b (sumarização)"
echo "   • BERT: bert-base-cased (EN), bert-base-portuguese-cased (PT)"
echo ""
echo " Para ativar o ambiente:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo " Para iniciar o Ollama (em outro terminal):"
echo "   ollama serve"
echo ""
echo " Para testar o LLM:"
echo "   ollama run llama3.1:8b 'Resuma em 2 frases: O COPOM decidiu manter a taxa Selic em 10,5%.'"
echo ""
echo " Para rodar o notebook:"
echo "   jupyter notebook $SCRIPT_DIR/01_teste_sumarizacao.ipynb"
echo ""
