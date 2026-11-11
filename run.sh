#!/bin/sh
# Tự chọn backend theo GPU của máy rồi chạy docker compose: NVIDIA -> cuda, Intel -> openvino, còn lại -> cpu.
# Dùng:  ./run.sh   |   ./run.sh down   |   MODE=cpu ./run.sh
cd "$(dirname "$0")" || exit 1
if [ -z "$MODE" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then MODE=cuda
  elif [ -e /dev/dri/renderD128 ] && lspci 2>/dev/null | grep -qi 'intel.*\(vga\|display\)'; then MODE=intel
  else MODE=cpu; fi
fi
echo "Backend: $MODE"
files="-f docker-compose.yml"
[ "$MODE" != cpu ] && files="$files -f docker-compose.$MODE.yml"
[ $# -eq 0 ] && set -- up -d --build
# shellcheck disable=SC2086
docker compose $files "$@"
