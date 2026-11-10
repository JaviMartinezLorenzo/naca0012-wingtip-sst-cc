#!/bin/bash
#------------------------------------------------------------------------------
# Run pv_vortex.py with ParaView's pvpython.
#
#   scripts/postprocess/pv_vortex.sh <case dir> <label> [pv_vortex.py options]
#
# Examples:
#   scripts/postprocess/pv_vortex.sh ~/naca_server_500 nutk_medium_500
#   scripts/postprocess/pv_vortex.sh OpenFOAM spalding_medium --decomposed
#   scripts/postprocess/pv_vortex.sh ~/naca_server_500 test --no-images
#
# pvpython is taken from $PVPYTHON, else from PATH, else from the newest
# Windows ParaView install (C:\Program Files\ParaView *). With the Windows
# pvpython.exe under WSL, paths are converted with wslpath automatically.
# Results go to scripts/results/<label>/.
#------------------------------------------------------------------------------
set -euo pipefail

[ $# -ge 2 ] || { sed -n '3,16p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

here=$(cd "$(dirname "$0")" && pwd)
case_dir=$(readlink -f "$1"); label=$2; shift 2
[ -e "$case_dir" ] || { echo "case not found: $case_dir" >&2; exit 1; }

pv=${PVPYTHON:-}
if [ -z "$pv" ]; then
    pv=$(command -v pvpython 2>/dev/null || true)
fi
if [ -z "$pv" ]; then
    pv=$(ls -d "/mnt/c/Program Files/ParaView "*/bin/pvpython.exe 2>/dev/null | sort -V | tail -n 1 || true)
fi
[ -n "$pv" ] || { echo "pvpython not found: install ParaView or set PVPYTHON" >&2; exit 1; }

script="$here/pv_vortex.py"
if [[ "$pv" == *.exe ]]; then                      # Windows pvpython from WSL
    script=$(wslpath -w "$script")
    case_dir=$(wslpath -w "$case_dir")
fi

echo "pvpython: $pv"
exec "$pv" --force-offscreen-rendering "$script" "$case_dir" "$label" "$@"
