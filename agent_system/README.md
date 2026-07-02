# Agent System

A generic, task-agnostic job queue for running parallel Claude Code agents on batch processing tasks. Each task is defined by a JSON config file and a Claude Code agent definition — no framework code changes needed to add new tasks.

## Architecture

```
                     ┌──────────────────┐
                     │ launch_agents.sh │
                     │ spawns N agents  │
                     └────────┬─────────┘
                              │
           ┌──────────┬───────┼───────┬──────────┐
           │          │       │       │          │
      ┌────▼───┐ ┌────▼──┐ ┌─▼────┐ ┌▼─────┐ ┌──▼────┐
      │agent-1 │ │agent-2│ │agent-3│ │agent-4│ │agent-5│
      └────┬───┘ └────┬──┘ └──┬───┘ └──┬───┘ └──┬────┘
           │          │       │        │         │
           └──────────┴───────┼────────┴─────────┘
                              │
                    ┌─────────▼──────────┐
                    │   job_queue.py      │
                    │  (file-locked queue)│
                    └─────────┬──────────┘
                              │
                 ┌────────────┼────────────┐
                 │            │            │
          ┌──────▼──────┐ ┌──▼────────┐ ┌─▼────────┐
          │manifest.json│ │  input/   │ │ results/ │
          │  (state)    │ │  (files)  │ │ (output) │
          └─────────────┘ └──────────-┘ └──────────┘
```

## How It Works

1. **One file = one work item.** The queue scans an input folder and creates a manifest listing all files to process. It makes no assumptions about file format.
2. **Agents claim items.** Each agent calls `next` to claim the next available file, processes it, then calls `submit` with the result. The manifest tracks which items are pending, in progress, completed, or failed.
3. **File-level locking.** `fcntl.flock()` on `manifest.lock` ensures only one agent modifies the manifest at a time. The lock is held only during the brief manifest read/write, never during processing.
4. **Stale detection.** Items claimed but not completed within the configured timeout (default 30 min) are automatically reset to pending for another agent to pick up.

## Files

| File | Purpose |
|------|---------|
| `job_queue.py` | Core queue engine: init, next, submit, fail, status, reset |
| `config.py` | Loads and validates task configs (searches the task dirs below) |
| `launch_agents.sh` | Launches and monitors N parallel Claude Code agents |
| `tasks/*.json` | Toolkit task configs (autointerpret label/eval, steering judge, `example.json` template) |
| `agents/*.md` | Source of truth for the toolkit's Claude Code agent definitions (see below) |
| `examples/chain_label_all_sae.sh` | Reference: a serial, rate-limit-resumable campaign driver (`nohup caffeinate` launch pattern) |

## Paths and overrides

- **Project root** — task-JSON paths resolve against the project root, which
  defaults to two levels above this folder (the parent repo root when the
  toolkit lives at `<root>/interpret/agent_system`). Override with
  `AGENT_PROJECT_ROOT` (e.g. a standalone toolkit checkout, where the root is
  one level up).
- **Task lookup order** — `$AGENT_TASKS_DIR` (if set) → this folder's `tasks/`
  → `<project root>/scripts/agent_tasks/` (project-side tasks that don't
  belong in the toolkit, e.g. `colour-bridge`, `book-triage`).
- **Agent definitions** — Claude Code discovers agents at the repo root's
  `.claude/agents/`. The toolkit's agent `.md` files live in `agents/` here
  (so they travel with the toolkit); the parent repo's `.claude/agents/`
  contains symlinks to them. In a standalone toolkit checkout, symlink or copy
  `agent_system/agents/*.md` into that repo's `.claude/agents/`.

## Quick Start

### 1. Create a task config

Copy `tasks/example.json` to `tasks/mytask.json`:

```json
{
    "task_name": "mytask",
    "agent": "mytask-worker",
    "model": "opus",
    "input_folder": "path/to/input/files/",
    "output_folder": "path/to/output/",
    "jobs_folder": "resources/jobs/mytask/",
    "stale_timeout_minutes": 30
}
```

Paths are relative to the project root unless absolute.

**Required fields:**
- `task_name` — identifier for this task
- `agent` — name of the Claude Code agent definition (`.claude/agents/<agent>.md`)
- `input_folder` — folder containing input files (each file is one work item)
- `jobs_folder` — where manifest, results, and logs are stored

**Optional fields:**
- `model` — Claude model (default: `"opus"`)
- `output_folder` — if set, files already present here are skipped during init
- `stale_timeout_minutes` — reset uncompleted items after this many minutes (default: `30`)
- `on_complete` — shell command to run when all items are done (supports `{jobs_folder}`, `{input_folder}`, `{output_folder}` substitution)

### 2. Create an agent definition

Create `.claude/agents/mytask-worker.md`:

```markdown
---
name: mytask-worker
description: "Worker agent for mytask"
tools: Bash, Write
model: opus
---

# Worker Instructions

You operate as a worker in a parallel job queue. Your loop:

1. Claim an item:
   ```bash
   uv run python interpret/agent_system/job_queue.py --task mytask next --worker-id <your-worker-id>
   ```
   This returns JSON with `item_filename` and `input_path`, or `{"done": true}`.

2. Process the file at `input_path`. [Your task-specific instructions here.]

3. Write your result to a file and submit:
   ```bash
   uv run python interpret/agent_system/job_queue.py --task mytask submit --item <item_filename> --file /tmp/result.json
   ```

4. Repeat from step 1 until `next` returns `{"done": true}`.

If processing fails, report it:
```bash
uv run python interpret/agent_system/job_queue.py --task mytask fail --item <item_filename> --error "description"
```
```

### 3. Initialize and run

```bash
# Initialize the queue (scans input folder, creates manifest)
uv run python interpret/agent_system/job_queue.py --task mytask init

# Check status
uv run python interpret/agent_system/job_queue.py --task mytask status

# Launch 5 agents, each processing up to 40 items
bash interpret/agent_system/launch_agents.sh -t mytask -n 5 -r 40
```

## CLI Reference

All commands require `--task <name>`.

```bash
# Initialize queue from input folder
uv run python job_queue.py --task mytask init

# Claim next item (returns JSON to stdout)
uv run python job_queue.py --task mytask next --worker-id agent-1

# Submit completed result
uv run python job_queue.py --task mytask submit --item file.json --file /tmp/result.json

# Mark item as failed (will be retried by another agent)
uv run python job_queue.py --task mytask fail --item file.json --error "reason"

# Progress report
uv run python job_queue.py --task mytask status
uv run python job_queue.py --task mytask status --json

# Reset failed/stale items to pending
uv run python job_queue.py --task mytask reset
uv run python job_queue.py --task mytask reset --item file.json
```

## Launcher Reference

### Simple mode (default)

Run agents manually, exit when done:

```bash
bash launch_agents.sh -t mytask                  # 5 agents, 40 reps each
bash launch_agents.sh -t mytask -n 10 -r 100     # 10 agents, 100 reps each
```

| Flag | Description | Default |
|------|-------------|---------|
| `-t <task>` | Task name (required) | — |
| `-n <num>` | Number of parallel agents | 5 |
| `-r <reps>` | Items per agent before it stops | 40 |

Agents that exit are relaunched if work remains. Exits when all agents finish their repetitions or all items are done. Rate-limited agents are not relaunched.

### Daemon mode

For unattended operation (e.g., with launchd). Adds cooldown between runs and rate-limit handling:

```bash
bash launch_agents.sh -t mytask -n 5 -r 40 --daemon -i 300
```

| Flag | Description | Default |
|------|-------------|---------|
| `--daemon` | Enable daemon mode | off |
| `-i <min>` | Cooldown minutes between runs (only after rate limit) | 300 |

When agents hit a rate limit, the launcher stops relaunching and waits for remaining agents to finish (30-minute grace period). On exit, it writes the reason to `.last_run`. On next startup, if the previous exit was rate-limited and the cooldown hasn't elapsed, it waits.

### Pause / resume

```bash
bash launch_agents.sh -t mytask pause 12    # pause for 12 hours
rm resources/jobs/mytask/logs/.pause_until   # unpause immediately
```

## Job Queue State

Each task stores its state in `jobs_folder/`:

```
resources/jobs/mytask/
├── manifest.json      # Central state: item queue + statuses
├── manifest.lock      # File lock (fcntl.flock)
├── results/           # Submitted result files
└── logs/
    ├── worker_1.log   # Per-agent output
    ├── worker_2.log
    ├── .last_run       # Daemon mode: timestamp + exit reason
    └── .pause_until    # Pause control file
```

### Item lifecycle

```
pending → in_progress → completed
                ↓
             failed → pending (on reset or stale timeout)
```

## Concurrency Model

- **File locking**: `fcntl.flock()` exclusive lock held only during manifest read/write (microseconds), never during processing
- **Atomic writes**: Manifest updates use temp file + rename
- **Stale detection**: Items in_progress beyond the timeout are auto-reset to pending
- **No coordination needed**: Agents are fully independent; the manifest is the only shared state

## Design Decisions

- **Format-agnostic**: The queue never reads or parses input files. It just hands out filenames. This means it works with JSON, XML, plain text, images, or any other format.
- **Smallest-first ordering**: Items are queued by file size (ascending) to maximize early throughput and give quick progress feedback.
- **No chunking**: If you need to split large files into smaller work units, do that in preprocessing before running `init`. This keeps the framework simple.
- **No built-in API workers**: All workers are Claude Code agents. For API-based processing, write a task-specific script that calls `next`/`submit` in a loop.
