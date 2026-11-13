#!/bin/bash
# Dispara a geracao de resumos do Algoritmo 1 desacoplada do terminal.
#
# Resolve tres fragilidades de rodar o script no terminal do editor:
#   - nohup: o processo sobrevive ao fechamento do terminal e do editor;
#   - subshell + disown: o processo e reparentado para o launchd (PPID 1), de
#     modo que encerrar o editor nao o leva junto;
#   - caffeinate -is: impede que o macOS suspenda a maquina durante a execucao
#     (a configuracao atual e "pmset sleep 1", um minuto de inatividade).
#
# A tela pode apagar e a tampa NAO pode ser fechada: fechar a tampa suspende a
# maquina mesmo com caffeinate.
#
# Esta e a geracao do corpus pelo Algoritmo 1. O numero de submissoes por ata nao
# e escolhido aqui: e o N_sub fixado pelo experimento de determinismo
# (./rodar_determinismo_frio.sh), lido do relatorio dele. Sem esse relatorio a
# geracao aborta.
#
# Uso:
#   ./rodar_geracao.sh                    # todas as atas, os dois idiomas
#   ./rodar_geracao.sh --lang pt          # so portugues
#
# Qualquer argumento e repassado a gerar_resumos.py.
#
# Retomada: o script e idempotente. Se for interrompido, basta rodar de novo
# com os mesmos argumentos -- o checkpoint em resumos/resumos_<lang>.jsonl faz
# ele pular o que ja existe. O custo maximo de uma interrupcao e o resumo que
# estava em andamento.
set -euo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$AQUI/../.venv/bin/activate"
LOG_DIR="$AQUI/../output/algoritmo1/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/nohup_geracao_$(date +%Y%m%d_%H%M%S).log"

if ! curl -fsS --max-time 5 http://localhost:11434/api/version >/dev/null; then
  echo "ERRO: o Ollama nao respondeu em localhost:11434. Suba com 'ollama serve'." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV"

cd "$AQUI"
(
  nohup caffeinate -is python gerar_resumos.py "$@" > "$LOG" 2>&1 &
  echo "$!" > "$LOG_DIR/geracao.pid"
  disown
) &
sleep 3

PID="$(cat "$LOG_DIR/geracao.pid")"
echo "geracao iniciada  pid=$PID"
echo "log:  $LOG"
echo
echo "acompanhar:  tail -f \"$LOG\""
echo "interromper: kill $PID     (o progresso ja gravado e preservado)"
