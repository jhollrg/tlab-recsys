#!/usr/bin/env bash
# Полный прогон. Использование: bash run_all.sh [work_dir] [n_users]   (Colab CPU: ~3-4 ч; кэш в work_dir)
set -euo pipefail
W=${1:-work}; N=${2:-2000}
python -m pytest -q tests
python run_experiments.py --n-users "$N" --work-dir "$W"                    # этап 1
python diagnostics.py --work-dir "$W"
python run_stage2.py --work-dir "$W" --seeds 3 --tune 6                     # этап 2
python run_windows.py --work-dir "$W" --seeds 3                             # окна (с ДИ)
python run_stage3.py --work-dir "$W" --seeds 2                              # этап 3
python run_experiments.py --n-users "$N" --work-dir "${W}_hard" \
       --neg-pop 50 --neg-hist 50 --neg-art 50 --skip-greedy                # трудный протокол
python diagnostics.py --work-dir "${W}_hard"
python run_stage3.py --work-dir "${W}_hard" --seeds 1 --quick --skip-timing
python summarize.py --work-dirs "$W" "${W}_hard"
