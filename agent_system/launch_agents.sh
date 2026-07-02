#!/usr/bin/env bash
# Launch N parallel Claude Code agents for a task.
#
# Simple mode (default):
#   bash launch_agents.sh -t clean_books -n 5 -r 40
#
# Daemon mode (unattended, for launchd):
#   bash launch_agents.sh -t clean_books -n 5 -r 40 --daemon -i 300
#
# Pause (works in both modes):
#   bash launch_agents.sh -t clean_books pause 12    # pause 12 hours

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Launcher lives at interpret/agent_system/launch_agents.sh, so the project
# root defaults to two levels up (a past off-by-one here sent agents to
# $HOME and silently broke every queued task). AGENT_PROJECT_ROOT overrides
# it, e.g. for a standalone toolkit checkout where the root is one level up.
PROJECT_ROOT="${AGENT_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
QUEUE_PY="$SCRIPT_DIR/job_queue.py"
# Task JSONs are searched in order: $AGENT_TASKS_DIR (if set), the toolkit's
# own tasks/, then $PROJECT_ROOT/scripts/agent_tasks (project-side tasks).
# The same order lives in config._tasks_search_dirs for job_queue.py.
find_task_config() {
  local name="$1"
  local candidates=()
  [[ -n "${AGENT_TASKS_DIR:-}" ]] && candidates+=("$AGENT_TASKS_DIR/${name}.json")
  candidates+=("$SCRIPT_DIR/tasks/${name}.json")
  candidates+=("$PROJECT_ROOT/scripts/agent_tasks/${name}.json")
  for c in "${candidates[@]}"; do
    if [[ -f "$c" ]]; then echo "$c"; return 0; fi
  done
  return 1
}

# Defaults
TASK=""
NUM_AGENTS=5
REPS=40
DAEMON=false
INTERVAL_MINUTES=300
VARIANT=""             # subfolder substituted into {variant} in task JSON paths
MODEL_OVERRIDE=""      # if set, overrides the model declared in the task JSON

# Parse --daemon before getopts (getopts doesn't handle long opts)
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--daemon" ]]; then
    DAEMON=true
  else
    ARGS+=("$arg")
  fi
done
set -- ${ARGS[@]+"${ARGS[@]}"}

while getopts "t:n:r:i:V:m:" opt; do
  case $opt in
    t) TASK="$OPTARG" ;;
    n) NUM_AGENTS="$OPTARG" ;;
    r) REPS="$OPTARG" ;;
    i) INTERVAL_MINUTES="$OPTARG" ;;
    V) VARIANT="$OPTARG" ;;
    m) MODEL_OVERRIDE="$OPTARG" ;;
    *) echo "Usage: $0 -t <task> [-n num_agents] [-r reps] [--daemon] [-i interval_min] [-V variant] [-m model]" >&2; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

# Task is always required
if [[ -z "$TASK" ]]; then
  echo "Error: -t <task> is required" >&2
  echo "Usage: $0 -t <task> [-n num_agents] [-r reps] [--daemon] [-i interval_min] [-V variant] [-m model]" >&2
  exit 1
fi

# Read agent config from task JSON. Path fields are resolved via
# `job_queue.py paths` so the {variant} substitution is centralised in
# config.py rather than re-implemented in bash. Agent + model are not
# variant-dependent, so we still read them from the raw JSON.
if ! TASK_CONFIG="$(find_task_config "$TASK")"; then
  echo "Error: task config not found: ${TASK}.json (searched \$AGENT_TASKS_DIR, $SCRIPT_DIR/tasks, $PROJECT_ROOT/scripts/agent_tasks)" >&2
  exit 1
fi

# Helper: every internal queue invocation needs the variant baked in so
# init/status/paths see the same folders the agents will write to.
QUEUE_CMD=(uv run python "$QUEUE_PY" --task "$TASK")
if [[ -n "$VARIANT" ]]; then
  QUEUE_CMD+=(--variant "$VARIANT")
fi

PATHS_JSON=$(cd "$PROJECT_ROOT" && "${QUEUE_CMD[@]}" paths)
JOBS_FOLDER=$(python3 -c "import json,sys; print(json.loads('''$PATHS_JSON''')['jobs_folder'])")
AGENT=$(python3 -c "import json,sys; print(json.loads('''$PATHS_JSON''')['agent'])")
MODEL=$(python3 -c "import json,sys; print(json.loads('''$PATHS_JSON''')['model'])")
if [[ -n "$MODEL_OVERRIDE" ]]; then
  MODEL="$MODEL_OVERRIDE"
fi

LOG_DIR="$JOBS_FOLDER/logs"
PAUSE_FILE="$LOG_DIR/.pause_until"
LAST_RUN_FILE="$LOG_DIR/.last_run"

# --- Pause command ---
if [[ "${1:-}" == "pause" ]]; then
  hours="${2:-1}"
  mkdir -p "$LOG_DIR"
  pause_until=$(( $(date +%s) + hours * 3600 ))
  echo "$pause_until" > "$PAUSE_FILE"
  resume_time=$(date -r "$pause_until" '+%Y-%m-%d %H:%M')
  echo "Task '$TASK' paused for ${hours}h. Will resume at ${resume_time}."
  pgrep -f "launch_agents.sh.*-t $TASK" | grep -v $$ | xargs kill 2>/dev/null || true
  exit 0
fi

if [[ -z "$AGENT" ]]; then
  echo "Error: no 'agent' field in $TASK_CONFIG" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

# --- Honor pause file ---
if [[ -f "$PAUSE_FILE" ]]; then
  pause_until=$(cat "$PAUSE_FILE")
  now=$(date +%s)
  if [[ $now -lt $pause_until ]]; then
    remaining=$(( (pause_until - now) / 60 ))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task '$TASK' paused for ${remaining}m. Waiting..."
    while [[ $(date +%s) -lt $pause_until ]]; do
      remaining=$(( (pause_until - $(date +%s)) / 60 ))
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Paused: ${remaining}m remaining..."
      sleep 60
    done
  fi
  rm -f "$PAUSE_FILE"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pause expired. Resuming."
fi

# --- Daemon mode: cooldown ---
if $DAEMON && [[ -f "$LAST_RUN_FILE" ]]; then
  last_run_ts=$(awk '{print $1}' "$LAST_RUN_FILE")
  last_run_reason=$(awk '{print $2}' "$LAST_RUN_FILE")
  if [[ "$last_run_reason" == "rate_limited" ]]; then
    now=$(date +%s)
    elapsed_min=$(( (now - last_run_ts) / 60 ))
    if [[ $elapsed_min -lt $INTERVAL_MINUTES ]]; then
      wait_until=$(( last_run_ts + INTERVAL_MINUTES * 60 ))
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rate-limited ${elapsed_min}m ago. Waiting until cooldown expires..."
      while [[ $(date +%s) -lt $wait_until ]]; do
        remaining=$(( (wait_until - $(date +%s)) / 60 ))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cooldown: ${remaining}m remaining..."
        sleep 60
      done
    fi
  fi
fi

# Record run start
if $DAEMON; then
  echo "$(date +%s) normal" > "$LAST_RUN_FILE"
fi

# Track worker PIDs by slot index
declare -a PIDS
for ((i=1; i<=NUM_AGENTS; i++)); do
  PIDS[$i]=0
done

DONE=false
RATE_LIMITED=false
RATE_LIMIT_GRACE=1800  # 30 min grace for stragglers
RATE_LIMITED_AT=0

cleanup() {
  echo ""
  echo "Shutting down agents..."
  for ((i=1; i<=NUM_AGENTS; i++)); do
    if [[ ${PIDS[$i]} -ne 0 ]] && kill -0 "${PIDS[$i]}" 2>/dev/null; then
      kill "${PIDS[$i]}" 2>/dev/null || true
    fi
  done
  sleep 2
  for ((i=1; i<=NUM_AGENTS; i++)); do
    if [[ ${PIDS[$i]} -ne 0 ]] && kill -0 "${PIDS[$i]}" 2>/dev/null; then
      kill -9 "${PIDS[$i]}" 2>/dev/null || true
    fi
  done
  # Record exit reason for daemon mode
  if $DAEMON; then
    if $RATE_LIMITED; then
      echo "$(date +%s) rate_limited" > "$LAST_RUN_FILE"
    else
      echo "$(date +%s) normal" > "$LAST_RUN_FILE"
    fi
  fi
  echo "Final status:"
  cd "$PROJECT_ROOT" && "${QUEUE_CMD[@]}" status 2>/dev/null || true
  exit 0
}

trap cleanup SIGINT SIGTERM

check_all_done() {
  local status_json
  status_json=$(cd "$PROJECT_ROOT" && "${QUEUE_CMD[@]}" status --json 2>/dev/null) || return 1
  local pending in_progress failed
  pending=$(echo "$status_json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('items_pending',1))" 2>/dev/null) || return 1
  in_progress=$(echo "$status_json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('items_in_progress',0))" 2>/dev/null) || return 1
  failed=$(echo "$status_json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('items_failed',0))" 2>/dev/null) || return 1
  [[ "$pending" -eq 0 && "$in_progress" -eq 0 && "$failed" -eq 0 ]]
}

check_rate_limited() {
  local slot=$1
  local log_file="$LOG_DIR/worker_${slot}.log"
  if [[ -f "$log_file" ]]; then
    if tail -20 "$log_file" | grep -qi "hit your limit\|rate.limit\|usage.limit\|resets [0-9]"; then
      return 0
    fi
  fi
  return 1
}

launch_agent() {
  local slot=$1
  local worker_id="${TASK}-worker-$slot"
  local log_file="$LOG_DIR/worker_${slot}.log"
  local timestamp
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')

  echo "[$timestamp] Launching $worker_id (log: $log_file)"

  echo "" >> "$log_file"
  echo "=== $worker_id started at $timestamp ===" >> "$log_file"

  (
    unset CLAUDECODE
    cd "$PROJECT_ROOT"
    # Agents in .claude/agents/<name>.md hardcode `--task X` in their CLI
    # examples; AGENT_QUEUE_VARIANT propagates the per-batch variant
    # into job_queue.py without each agent file having to learn the flag.
    export AGENT_QUEUE_VARIANT="$VARIANT"
    claude \
      --agent "$AGENT" \
      --print \
      --dangerously-skip-permissions \
      --model "$MODEL" \
      --no-session-persistence \
      "You are worker $worker_id. Execute the worker loop using --worker-id $worker_id --task $TASK. Process up to $REPS items, then stop." \
      >> "$log_file" 2>&1
  ) &

  PIDS[$slot]=$!
}

# Initial launch
echo "Starting $NUM_AGENTS agents for task '$TASK' (agent: $AGENT, model: $MODEL, reps: $REPS)..."
if $DAEMON; then
  echo "Daemon mode: cooldown ${INTERVAL_MINUTES}m between runs"
fi
echo "Project root: $PROJECT_ROOT"
echo "Logs: $LOG_DIR"
echo ""

cd "$PROJECT_ROOT" && "${QUEUE_CMD[@]}" status 2>/dev/null || true
echo ""

for ((i=1; i<=NUM_AGENTS; i++)); do
  launch_agent "$i"
  sleep 2
done

echo ""
echo "All agents launched. Monitoring... (Ctrl+C to stop)"
echo ""

# Monitor loop
while ! $DONE; do
  sleep 60

  all_exited=true
  for ((i=1; i<=NUM_AGENTS; i++)); do
    pid=${PIDS[$i]}
    if [[ $pid -ne 0 ]] && ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      PIDS[$i]=0

      if check_rate_limited "$i"; then
        echo "[$(date '+%H:%M:%S')] Agent $i hit rate limit — will not relaunch."
        RATE_LIMITED=true
        [[ $RATE_LIMITED_AT -eq 0 ]] && RATE_LIMITED_AT=$(date +%s)
      elif ! $RATE_LIMITED; then
        if check_all_done; then
          echo "[$(date '+%H:%M:%S')] Agent $i finished — all work complete."
          DONE=true
          break
        else
          echo "[$(date '+%H:%M:%S')] Agent $i exited — relaunching..."
          launch_agent "$i"
        fi
      else
        echo "[$(date '+%H:%M:%S')] Agent $i exited (rate limit active — not relaunching)."
      fi
    fi

    if [[ ${PIDS[$i]} -ne 0 ]] && kill -0 "${PIDS[$i]}" 2>/dev/null; then
      all_exited=false
    fi
  done

  # Daemon mode: kill stragglers after grace period
  if $DAEMON && $RATE_LIMITED && ! $all_exited && [[ $RATE_LIMITED_AT -gt 0 ]]; then
    elapsed_since_rl=$(( $(date +%s) - RATE_LIMITED_AT ))
    if [[ $elapsed_since_rl -ge $RATE_LIMIT_GRACE ]]; then
      echo "[$(date '+%H:%M:%S')] Rate limit grace period expired. Killing remaining agents..."
      for ((i=1; i<=NUM_AGENTS; i++)); do
        if [[ ${PIDS[$i]} -ne 0 ]] && kill -0 "${PIDS[$i]}" 2>/dev/null; then
          kill "${PIDS[$i]}" 2>/dev/null || true
        fi
      done
      all_exited=true
    fi
  fi

  if $all_exited && ! $DONE; then
    if $RATE_LIMITED; then
      echo "[$(date '+%H:%M:%S')] All agents exited (rate limited). Exiting to cooldown."
      DONE=true
    elif check_all_done; then
      DONE=true
      echo "[$(date '+%H:%M:%S')] All work complete!"
    else
      echo "[$(date '+%H:%M:%S')] All agents exited but work remains — relaunching..."
      for ((i=1; i<=NUM_AGENTS; i++)); do
        launch_agent "$i"
        sleep 2
      done
    fi
  fi
done

# Clean shutdown
cleanup
