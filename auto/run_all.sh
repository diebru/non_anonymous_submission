#!/usr/bin/env bash
# Master entry point. Runs setup -> reasoning -> mceval(dispatch) -> plots.
# Each sub-script is independently re-runnable; this just chains them.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$HERE/00_setup.sh"
bash "$HERE/10_reasoning.sh"
bash "$HERE/20_mceval.sh"
bash "$HERE/30_plots.sh"
