#!/usr/bin/env bash
# Bit-exact simulation of dmn_lqg_core with GHDL (Linux/macOS/Git Bash).
# Needs ghdl in PATH and the generated files from:  python main.py model hardware
set -euo pipefail
cd "$(dirname "$0")/vhdl"
[ -f gen/lqg_params_pkg.vhd ] || { echo "run  python main.py model hardware  first"; exit 1; }
mkdir -p work
ghdl -a --std=08 --workdir=work gen/lqg_params_pkg.vhd neuron_pkg.vhd dmn_lqg_core.vhd tb_dmn_lqg_core.vhd
ghdl -e --std=08 --workdir=work tb_dmn_lqg_core
ghdl -r --std=08 --workdir=work tb_dmn_lqg_core -gVEC_FILE=gen/vectors.txt --assert-level=error
