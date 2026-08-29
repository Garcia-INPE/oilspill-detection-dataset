#!/usr/bin/env bash
set -euo pipefail

usage() {
	echo "Usage: bash 01-install.sh [cpu|gpu]"
}

if [[ $# -ne 1 ]]; then
	usage
	exit 1
fi

PROFILE="$1"

python -m pip install --upgrade pip
python -m pip install -r requirements-common.txt

case "$PROFILE" in
	cpu)
		python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-cpu.txt
		;;
	gpu)
		python -m pip install --index-url https://download.pytorch.org/whl/cu121 -r requirements-gpu.txt
		;;
	*)
		usage
		exit 1
		;;
esac

echo "Installation complete (profile=$PROFILE)."
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
