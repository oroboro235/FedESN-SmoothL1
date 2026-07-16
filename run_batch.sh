#!/usr/bin/env bash
# Purpose: run a built-in list of commands one by one (serially); by default
#          it re-launches itself into the background via nohup.
# Usage:
#   ./run_batch.sh            # auto background (survives closing the terminal)
#   FG=1 ./run_batch.sh       # run in the foreground (for debugging)

set -euo pipefail

# Directory this script lives in
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p ./logs

# ---------- Auto background ----------
# On the first run (_BATCH_BG unset), re-launch itself into the background via
# nohup and exit the current invocation. Set FG=1 to skip and run in foreground.
if [[ -z "${_BATCH_BG:-}" && -z "${FG:-}" ]]; then
    MASTER_LOG="./logs/batch_$(date +%Y%m%d_%H%M%S).log"
    _BATCH_BG=1 nohup bash "${BASH_SOURCE[0]}" "$@" > "${MASTER_LOG}" 2>&1 &
    echo "Started run_batch.sh in the background"
    echo "  PID      : $!"
    echo "  Master log: ${MASTER_LOG}"
    echo "  Progress : tail -f ${MASTER_LOG}"
    echo "  Stop     : kill $!   (or pkill -f run_batch.sh)"
    exit 0
fi
# -------------------------------------

# ---------- Signal handling: clean up child processes when the main process is killed ----------
# Each command is launched with setsid into its own process group, so CHILD_PGID
# equals its PID. On TERM/INT we kill the whole process group with
# kill -- -CHILD_PGID (killing python and its n_jobs worker children too).
CHILD_PGID=""
cleanup() {
    echo ""
    echo ">> Received termination signal, cleaning up the running child processes..."
    if [[ -n "${CHILD_PGID}" ]] && kill -0 "-${CHILD_PGID}" 2>/dev/null; then
        kill -TERM "-${CHILD_PGID}" 2>/dev/null || true
        sleep 3
        kill -KILL "-${CHILD_PGID}" 2>/dev/null || true
    fi
    echo ">> Cleanup done, exiting."
    exit 130
}
trap cleanup TERM INT
# -----------------------------------------------------------------------------------------------

# ---------- Load the command list from a file ----------
# The command file is one command per line; blank lines and lines starting with
# '#' are ignored. Override the path with CMD_FILE=... if needed.
# This file is git-ignored; commands.sample.txt is the tracked template.
CMD_FILE="${CMD_FILE:-${SCRIPT_DIR}/commands.txt}"

if [[ ! -f "${CMD_FILE}" ]]; then
    echo "Command file not found: ${CMD_FILE}" >&2
    echo "Copy the template and edit it:" >&2
    echo "  cp ${SCRIPT_DIR}/commands.sample.txt ${CMD_FILE}" >&2
    exit 1
fi

COMMANDS=()
while IFS= read -r line || [[ -n "${line}" ]]; do
    # strip leading/trailing whitespace
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    # skip blank lines and comments
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    COMMANDS+=("${line}")
done < "${CMD_FILE}"

if [[ ${#COMMANDS[@]} -eq 0 ]]; then
    echo "No commands to run in ${CMD_FILE} (all blank/comments)." >&2
    exit 1
fi

echo ">> Loaded ${#COMMANDS[@]} command(s) from ${CMD_FILE}"
# -------------------------------------------------------

# Run each command serially, one log file per command.
for cmd in "${COMMANDS[@]}"; do
    TS="$(date +%Y%m%d_%H%M%S)"
    SAFE_CMD="$(echo "${cmd}" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/_+/_/g; s/^_+//; s/_+$//' | cut -c1-100)"
    LOG="./logs/${TS}_${SAFE_CMD}.log"
    echo "==================================================="
    echo ">> Running: ${cmd}"
    echo "   Log: ${LOG}"

    # setsid puts the command in its own process group; run it in the background,
    # record the PGID, then wait to keep execution serial.
    setsid bash -c "${cmd}" > "${LOG}" 2>&1 &
    CHILD_PGID=$!
    rc=0
    wait "${CHILD_PGID}" || rc=$?
    echo "   Done, exit code: ${rc}"
    CHILD_PGID=""
done

echo "==================================================="
echo "All ${#COMMANDS[@]} command(s) finished (serial execution)."
