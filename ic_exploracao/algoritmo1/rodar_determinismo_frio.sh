#!/bin/bash
# Dispara o experimento de determinismo (procedimento TesteDeterminismo, com o
# estado do servidor limpo antes de cada chamada), desacoplado do terminal. Ao
# terminar, grava o relatorio do qual a geracao do corpus (Algoritmo 1) le o
# N_sub.
#
# Mesmas protecoes de rodar_geracao.sh: nohup para sobreviver ao fechamento do
# editor, subshell com disown para ser reparentado ao launchd, e caffeinate -is
# para impedir a suspensao do macOS. A tampa NAO pode ser fechada.
#
# Esta execucao e mais lenta que a geracao normal por construcao: o modelo e
# descarregado e recarregado antes de cada chamada, justamente para que todas
# partam do mesmo estado de cache.
#
# Uso:
#   ./rodar_determinismo_frio.sh                # os dois idiomas
#   ./rodar_determinismo_frio.sh --lang pt      # so portugues
#
# Retomada: idempotente. O checkpoint fica em
# resumos/determinismo_frio_<lang>.jsonl e o script pula o que ja existe.
set -euo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$AQUI/../.venv/bin/activate"
LOG_DIR="$AQUI/../output/algoritmo1/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/nohup_determinismo_frio_$(date +%Y%m%d_%H%M%S).log"

if ! curl -fsS --max-time 5 http://localhost:11434/api/version >/dev/null; then
  echo "ERRO: o Ollama nao respondeu em localhost:11434. Suba com 'ollama serve'." >&2
  exit 1
fi

if pgrep -f "gerar_resumos.py" >/dev/null; then
  echo "ERRO: a geracao do estudo esta rodando. As duas execucoes disputariam" >&2
  echo "o mesmo servidor, e descarregar o modelo aqui corromperia o estado da" >&2
  echo "outra. Pare a geracao antes: pkill -f gerar_resumos.py" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV"

cd "$AQUI"
(
  nohup caffeinate -is python determinismo.py --rodar-frio "$@" > "$LOG" 2>&1 &
  echo "$!" > "$LOG_DIR/determinismo_frio.pid"
  disown
) &
sleep 3

PID="$(cat "$LOG_DIR/determinismo_frio.pid")"
echo "experimento em cache frio iniciado  pid=$PID"
echo "log:  $LOG"
echo
echo "acompanhar:  tail -f \"$LOG\""
echo "interromper: kill $PID     (o progresso ja gravado e preservado)"
