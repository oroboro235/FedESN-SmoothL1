#!/usr/bin/env bash
# Usage: ./run_nohup.sh "python ./tsr_centralized_search.py --arg1 val1"
# Purpose: run the given command in the background via nohup, naming the log
#          file by "datetime_command".

set -euo pipefail

# Join all arguments into one full command (works with or without quotes).
CMD="$*"

if [[ -z "${CMD}" ]]; then
    echo "Usage: $0 \"command to run\"" >&2
    echo "Example: $0 python ./tsr_centralized_search.py --lr 0.01" >&2
    exit 1
fi

# Log directory
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "${LOG_DIR}"

# Timestamp
TS="$(date +%Y%m%d_%H%M%S)"

# Sanitize the command into a safe filename fragment:
#   - replace any non [alnum . - _] characters with underscore
#   - collapse repeated underscores
#   - strip leading/trailing underscores
#   - truncate to avoid overly long filenames
SAFE_CMD="$(echo "${CMD}" \
    | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/_+/_/g; s/^_+//; s/_+$//' \
    | cut -c1-100)"

LOG_FILE="${LOG_DIR}/${TS}_${SAFE_CMD}.log"

# Write some metadata to the top of the log file.
{
    echo "# CMD : ${CMD}"
    echo "# TIME: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# CWD : $(pwd)"
    echo "#----------------------------------------"
} > "${LOG_FILE}"

# Run in the background via nohup; stdout/stderr both appended to the log.
nohup bash -c "${CMD}" >> "${LOG_FILE}" 2>&1 &
PID=$!

echo "Started: ${CMD}"
echo "  PID : ${PID}"
echo "  Log : ${LOG_FILE}"
echo "  View: tail -f ${LOG_FILE}"
echo "  Stop: kill ${PID}"

# Also expose the PID (optional, handy for other scripts to capture).
echo "${PID}" > "${LOG_FILE}.pid"
