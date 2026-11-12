#!/bin/bash
#------------------------------------------------------------------------------
# Unattended mesh + steady RANS run for the NACA 0012 wingtip vortex case.
#
# Usage:
#   ./Allrun.sh --detach --mesh-only     # mesh in the background, then stop
#   ./Allrun.sh --detach                 # mesh, then solve, in the background
#   tail -f Allrun.log                   # watch progress (Ctrl-C only stops tail)
#
# Options:
#   --detach       run in the background, detached from the terminal, with the
#                  output in Allrun.log. Survives logging out and closing SSH.
#                  The quick checks run first, so errors show immediately.
#   --mesh-only    stop after the mesh has passed its checks
#   --solve-only   use the existing mesh; resumes if a run was interrupted
#   --fresh        with --solve-only: move the previous run (processor*, time
#                  folders, postProcessing, solver logs) to previous/<date>/
#                  and start the solver from iteration 0. Needed after
#                  changing anything in 0/ or the models.
#   --clean        delete previous mesh, results and logs first
#   -h, --help     show this help
#
# OpenFOAM: the version already loaded in the shell is used (with a warning
# if it is not FOAM_VERSION). If none is loaded, FOAM_VERSION (default 2312,
# the version the case was set up and validated with) is sourced; the newest
# installed one only if that is missing. Another version: FOAM_VERSION=2606.
#
# Environment overrides:
#   FOAM_VERSION      OpenFOAM version to source if none is loaded (default 2312)
#   MPI_OPTS          mpirun options (default: --map-by socket --bind-to core)
#                     e.g. MPI_OPTS="--use-hwthread-cpus" on a laptop
#   MIN_FREE_GB       free disk needed to start (default: 30)
#   MIN_MEAN_LAYERS   mean prism layers on the wing needed to go on to the
#                     solver (default: 22 of 25; the coarse mesh reaches
#                     about 19, so use MIN_MEAN_LAYERS=19 for it)
#
# The number of MPI ranks is numberOfSubdomains in system/decomposeParDict.
# The script stops at the first failure, before spending solver time on a bad
# mesh. Each step writes log.<step> in this directory. Only one run per case
# directory at a time (.Allrun.pid).
#------------------------------------------------------------------------------

SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")"

#- Load OpenFOAM if this shell has not (its bashrc is not 'set -eu' safe).
#  Run a tool rather than look for it: PATH can be set without the libraries.
foam_ok() { [ -n "${WM_PROJECT_DIR:-}" ] && foamDictionary -help > /dev/null 2>&1; }

FOAM_VERSION=${FOAM_VERSION:-2312}

if ! foam_ok; then
    for rc in $(ls -d /usr/lib/openfoam/openfoam"$FOAM_VERSION"/etc/bashrc \
                      /opt/openfoam"$FOAM_VERSION"/etc/bashrc \
                      "$HOME"/OpenFOAM/OpenFOAM-v"$FOAM_VERSION"/etc/bashrc 2>/dev/null) \
              $(ls -d /usr/lib/openfoam/openfoam[0-9]*/etc/bashrc \
                      /opt/openfoam[0-9]*/etc/bashrc \
                      "$HOME"/OpenFOAM/OpenFOAM-*/etc/bashrc 2>/dev/null | sort -V -r); do
        case "$rc" in
            *"$FOAM_VERSION"/etc/bashrc) echo "OpenFOAM not loaded, sourcing $rc" ;;
            *) echo "WARNING: OpenFOAM v$FOAM_VERSION not found, sourcing $rc" ;;
        esac
        # shellcheck disable=SC1090
        source "$rc"
        break
    done
elif [ "${WM_PROJECT_VERSION:-}" != "v$FOAM_VERSION" ] && [ "${WM_PROJECT_VERSION:-}" != "$FOAM_VERSION" ]; then
    echo "WARNING: OpenFOAM ${WM_PROJECT_VERSION:-?} is loaded, not v$FOAM_VERSION; using it"
fi

set -euo pipefail

#------------------------------------------------------------------------------
# Helpers
#------------------------------------------------------------------------------

info() { echo "[$(date '+%F %T')] $*"; }

die() {
    echo "[$(date '+%F %T')] ERROR: $*" >&2
    echo "[$(date '+%F %T')] Allrun.sh STOPPED" >&2
    exit 1
}

trap 'die "unexpected failure at line $LINENO of Allrun.sh"' ERR

usage() { sed -n '3,/^#---/p' "$SELF" | sed '$d; s/^# \{0,1\}//'; }

hms() { printf '%dh%02dm%02ds' $(($1/3600)) $(($1%3600/60)) $(($1%60)); }

# run <step name> <command...>: run with output in log.<step>, stop on failure
run() {
    local name=$1; shift
    local log="log.$name" start=$SECONDS
    if [ -e "$log" ]; then mv "$log" "$log.$(date +%Y%m%d-%H%M%S)"; fi

    info "START $name"
    if ! "$@" > "$log" 2>&1 || grep -q "FOAM FATAL" "$log"; then
        grep -m1 -A8 "FOAM FATAL" "$log" >&2 || tail -n 15 "$log" >&2
        die "$name failed, see $PWD/$log"
    fi
    info "DONE  $name ($(hms $((SECONDS-start))))"
}

run_par() {
    local name=$1 app=$2; shift 2
    run "$name" mpirun -np "$NP" "${MPI[@]}" "$app" -parallel "$@"
}

time_dirs() {
    find "$1" -maxdepth 1 -type d \( -name '[1-9]*' -o -name '0.[0-9]*' \) \
        -printf '%f\n' | grep -E '^[0-9.eE+-]+$' || true
}

# checkMesh exits 0 even when checks fail, so read its verdict. Aspect ratio
# above its fixed limit of 1000 is expected with a 10 um first cell on the
# larger faces of the coarse mesh: warn only. Anything else stops the run.
check_mesh_verdict() {
    grep -q "^Mesh OK\." "$1" && return 0

    local other
    other=$(grep -F '***' "$1" | grep -v "High aspect ratio" || true)
    if [ -z "$other" ] && grep -q "High aspect ratio" "$1"; then
        info "WARNING: only the aspect-ratio check failed (expected for thin wall layers)"
        return 0
    fi
    [ -z "$other" ] || echo "$other" >&2
    die "checkMesh reported failures, see $PWD/$1"
}

# True if the mesh in <case dir> has the wake monitor cellZones (topoSetDict)
has_wake_zones() {
    grep -qs wake_x "$1/constant/polyMesh/cellZones"
}

# Mean layers per face on the wing, from the final table in log.snappyHexMesh
mean_layers() {
    awk '/^patch +faces +layers +overall thickness/ { t=1; next }
         t && $1=="wing" { v=$4; t=0 }
         END { print v }' log.snappyHexMesh
}

#------------------------------------------------------------------------------
# Arguments and pre-flight checks
#------------------------------------------------------------------------------

doMesh=true doSolve=true doClean=false doDetach=false doFresh=false
childArgs=()                      # options passed on to a detached run
while [ $# -gt 0 ]; do
    case "$1" in
        --detach)     doDetach=true ;;
        --mesh-only)  doSolve=false ;;
        --solve-only) doMesh=false ;;
        --clean)      doClean=true ;;
        --fresh)      doFresh=true ;;
        -h|--help)    usage; exit 0 ;;
        *)            usage; die "unknown option: $1" ;;
    esac
    [ "$1" = --detach ] || childArgs+=("$1")
    shift
done

[ "$doMesh" = true ] || [ "$doSolve" = true ] \
    || die "--mesh-only and --solve-only cannot be combined"
[ "$doMesh" = true ] || [ "$doClean" = false ] \
    || die "--clean would delete the mesh that --solve-only needs"
[ "$doFresh" = false ] || [ "$doMesh" = false ] \
    || die "--fresh only works with --solve-only"

foam_ok || die "OpenFOAM not found or not working: source its etc/bashrc first"

for f in system/blockMeshDict system/snappyHexMeshDict system/decomposeParDict \
         system/controlDict constant/triSurface/wing.stl 0/U 0/p 0/k 0/omega 0/nut
do
    [ -f "$f" ] || die "missing $f (run this from a complete case)"
done

NP=$(foamDictionary -entry numberOfSubdomains -value system/decomposeParDict)
read -r -a MPI <<< "${MPI_OPTS---map-by socket --bind-to core}"

# 'set' test: a deliberately empty MPI_OPTS also skips this check
if [ -z "${MPI_OPTS+set}" ]; then
    cores=$(lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l)
    [ "$NP" -le "$cores" ] || die "numberOfSubdomains is $NP but this machine \
has $cores physical cores. Lower it in system/decomposeParDict, or set \
MPI_OPTS=\"--use-hwthread-cpus\" to use hardware threads."
fi

MIN_FREE_GB=${MIN_FREE_GB:-30}
freeGB=$(( $(df -Pk . | awk 'NR==2 {print $4}') / 1024 / 1024 ))
[ "$freeGB" -ge "$MIN_FREE_GB" ] || die "only ${freeGB} GB free on \
$(df -P . | awk 'NR==2 {print $6}'), need ${MIN_FREE_GB} GB. Move the case to \
a larger disk or set MIN_FREE_GB."

MIN_MEAN_LAYERS=${MIN_MEAN_LAYERS:-22}

PIDFILE=.Allrun.pid
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    die "Allrun.sh is already running in this case (PID $(cat "$PIDFILE")). \
Follow it with: tail -f Allrun.log"
fi

if [ "$doMesh" = true ]; then
    if [ "$doClean" = false ] && { [ -e constant/polyMesh ] \
       || compgen -G "processor*" > /dev/null || [ -n "$(time_dirs .)" ]; }; then
        die "existing mesh or results found. Nothing was changed. Use --clean \
to delete them, or --solve-only to reuse the mesh."
    fi
else
    [ -f constant/polyMesh/faces ] || [ -f constant/polyMesh/faces.gz ] \
        || die "no mesh in constant/polyMesh for --solve-only"

    # Without --fresh, old results would be picked up as the starting point
    # (decomposePar takes every time folder), with the old 0/ settings.
    if [ "$doFresh" = false ] && [ -z "$(time_dirs processor0 2>/dev/null)" ] \
       && [ -n "$(time_dirs .)" ]; then
        die "results from a previous run found ($(time_dirs . | tr '\n' ' ')). \
Use --fresh to move them to previous/ and start again from iteration 0."
    fi
fi

#- Detach: relaunch in a new session (setsid) with no terminal, so that
#  logging out cannot send it SIGHUP. Open MPI's mpirun handles SIGHUP itself
#  and stops the job even under nohup.
if [ "$doDetach" = true ]; then
    if [ -e Allrun.log ]; then mv Allrun.log "Allrun.log.$(date +%Y%m%d-%H%M%S)"; fi
    setsid nohup "$SELF" ${childArgs[@]+"${childArgs[@]}"} \
        > Allrun.log 2>&1 < /dev/null &
    # wait for the background run to take the lock (or to fail its checks)
    for _ in $(seq 60); do
        if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            info "Running in the background (PID $(cat "$PIDFILE")). You can log out."
            echo "    follow:  tail -f $PWD/Allrun.log"
            echo "    stop:    kill -- -$(cat "$PIDFILE")"
            exit 0
        fi
        if grep -q "Allrun.sh STOPPED" Allrun.log 2>/dev/null; then
            cat Allrun.log >&2
            die "the background run stopped, see $PWD/Allrun.log"
        fi
        sleep 0.5
    done
    die "could not confirm the background run started, check $PWD/Allrun.log"
fi

echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

if [ "$doClean" = true ]; then
    info "Cleaning previous mesh, results and logs"
    rm -rf processor* constant/polyMesh constant/extendedFeatureEdgeMesh \
           constant/triSurface/wing.eMesh postProcessing subsetBox.obj log.*
    time_dirs . | while read -r t; do rm -rf "./$t"; done
fi

info "Case:  $PWD"
info "$WM_PROJECT $WM_PROJECT_VERSION, $NP ranks, mpirun options: ${MPI[*]:-none}, ${freeGB} GB free"
totalStart=$SECONDS

#------------------------------------------------------------------------------
# Mesh
#------------------------------------------------------------------------------

if [ "$doMesh" = true ]; then
    run blockMesh blockMesh
    run surfaceFeatureExtract surfaceFeatureExtract
    run decomposePar.mesh decomposePar -force
    run_par snappyHexMesh snappyHexMesh -overwrite
    run reconstructParMesh reconstructParMesh -constant
    rm -rf processor*

    run checkMesh checkMesh
    grep -E "cells: |non-orthogonality Max|Max skewness|Max aspect" log.checkMesh \
        | sed 's/^ */    /'
    check_mesh_verdict log.checkMesh

    layers=$(mean_layers)
    [ -n "$layers" ] || die "no layer summary in log.snappyHexMesh"
    info "Prism layers on wing: mean $layers per face"
    awk -v a="$layers" -v b="$MIN_MEAN_LAYERS" 'BEGIN { exit !(a >= b) }' \
        || die "mean layers $layers is below MIN_MEAN_LAYERS=$MIN_MEAN_LAYERS: \
the first cell would be too thick on part of the wing. Mesh kept for inspection."

    run topoSet topoSet           # wake monitor zones, see system/topoSetDict
    info "MESH READY ($(hms $((SECONDS-totalStart))))"
fi

#------------------------------------------------------------------------------
# Solve
#------------------------------------------------------------------------------

if [ "$doSolve" = true ]; then
    if [ "$doFresh" = true ]; then
        archive="previous/$(date +%Y%m%d-%H%M%S)"
        info "Moving the previous run to $archive"
        mkdir -p "$archive"
        for f in processor* postProcessing log.decomposePar.solve* \
                 log.potentialFoam* log.simpleFoam* log.reconstructPar* \
                 $(time_dirs .); do
            if [ -e "$f" ]; then mv "$f" "$archive/"; fi
        done
    fi

    if [ -d processor0 ] && [ -n "$(time_dirs processor0)" ]; then
        info "Previous run found in processor*: continuing from its latest time"
        has_wake_zones processor0 || run_par topoSet topoSet
    else
        rm -rf processor*
        has_wake_zones . || run topoSet topoSet
        run decomposePar.solve decomposePar
        run_par potentialFoam potentialFoam -writephi
    fi

    run_par simpleFoam simpleFoam
    run reconstructPar reconstructPar -latestTime

    grep -E "SIMPLE solution converged" log.simpleFoam | sed 's/^/    /' || true
    grep -E "patch wing y\+" log.simpleFoam | tail -n 1 | sed 's/^ */    /' || true
    grep -E "^ +C[ld]:" log.simpleFoam | tail -n 2 | sed 's/^ */    /' || true
fi

info "ALL DONE ($(hms $((SECONDS-totalStart))))"
