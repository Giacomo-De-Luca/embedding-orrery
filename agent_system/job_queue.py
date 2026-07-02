"""
Generic task-agnostic agent job queue.

Manages a file-based job queue where multiple Claude Code agents work in parallel,
each claiming and processing separate items. Items are processed smallest-to-largest
by file size. A manifest file tracks progress with file-level locking for safety.

This is format-agnostic: the queue hands out filenames without reading or parsing
input files. One file = one work item.

Usage:
    python job_queue.py --task clean_books init
    python job_queue.py --task clean_books next --worker-id agent-1
    python job_queue.py --task clean_books submit --item file.json --file /tmp/result.json
    python job_queue.py --task clean_books fail --item file.json --error "bad input"
    python job_queue.py --task clean_books status [--json]
    python job_queue.py --task clean_books reset [--item file.json]
"""

import os
import sys
import json
import fcntl
import shutil
import uuid
import socket
import argparse
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

try:
    from .config import load_config
except ImportError:
    from config import load_config


# ============= LOCKING =============

def _acquire_manifest(jobs_folder):
    """Acquire exclusive lock and read manifest. Returns (manifest, lock_fh)."""
    jobs_folder = Path(jobs_folder)
    lock_path = jobs_folder / "manifest.lock"
    lock_fh = open(lock_path, "w")
    fcntl.flock(lock_fh, fcntl.LOCK_EX)
    try:
        manifest = json.loads((jobs_folder / "manifest.json").read_text("utf-8"))
    except Exception:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()
        raise
    return manifest, lock_fh


def _release_manifest(jobs_folder, manifest, lock_fh):
    """Write manifest atomically and release lock."""
    jobs_folder = Path(jobs_folder)
    tmp = jobs_folder / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    tmp.rename(jobs_folder / "manifest.json")
    fcntl.flock(lock_fh, fcntl.LOCK_UN)
    lock_fh.close()


def _unlock_manifest(lock_fh):
    """Release lock without writing (used on error paths)."""
    fcntl.flock(lock_fh, fcntl.LOCK_UN)
    lock_fh.close()


def _atomic_write(path, data_str):
    """Write a file atomically via temp + rename."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(data_str, "utf-8")
    tmp.rename(path)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _generate_worker_id():
    host = socket.gethostname()[:8]
    return f"{host}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


# ============= STALE DETECTION =============

def _reset_stale_items(manifest):
    """Reset items that have been in_progress for too long."""
    timeout = manifest.get("stale_timeout_minutes", 30)
    now = datetime.now(timezone.utc)
    for item_info in manifest["items"].values():
        if item_info["status"] != "in_progress":
            continue
        claimed = item_info.get("claimed_at")
        if not claimed:
            continue
        claimed_dt = datetime.fromisoformat(claimed)
        elapsed = (now - claimed_dt).total_seconds() / 60
        if elapsed > timeout:
            item_info["status"] = "pending"
            item_info["claimed_at"] = None
            item_info["worker_id"] = None


# ============= INIT =============

def init_jobs(config):
    """
    One-time setup: scan input_folder for files, build manifest.
    Items are sorted smallest-to-largest by file size.
    Skips items already present in output_folder (if set).
    """
    input_folder = Path(config["input_folder"])
    jobs_folder = Path(config["jobs_folder"])
    output_folder = Path(config["output_folder"]) if config.get("output_folder") else None

    if (jobs_folder / "manifest.json").exists():
        print(f"Jobs already initialized at {jobs_folder}")
        print("Delete manifest.json to reinitialize, or use 'reset' to fix stale items.")
        return None

    if not input_folder.exists():
        print(f"Error: input folder not found: {input_folder}")
        return None

    # Scan input folder
    items = []
    for f in input_folder.iterdir():
        if f.is_file() and not f.name.startswith("."):
            # Skip if already in output
            if output_folder and (output_folder / f.name).exists():
                continue
            items.append({
                "filename": f.name,
                "file_size": f.stat().st_size,
            })

    items.sort(key=lambda x: x["file_size"])

    if not items:
        print("No items to process.")
        return None

    print(f"Found {len(items)} items to process")

    # Build manifest
    manifest = {
        "version": 1,
        "task_name": config["task_name"],
        "created_at": _now(),
        "input_folder": str(input_folder),
        "output_folder": str(output_folder) if output_folder else None,
        "stale_timeout_minutes": config.get("stale_timeout_minutes", 30),
        "on_complete": config.get("on_complete"),
        "item_queue": [item["filename"] for item in items],
        "items": {},
        "current_item_index": 0,
    }

    for item in items:
        manifest["items"][item["filename"]] = {
            "status": "pending",
            "file_size": item["file_size"],
            "claimed_at": None,
            "completed_at": None,
            "worker_id": None,
        }

    # Create directory structure
    jobs_folder.mkdir(parents=True, exist_ok=True)
    (jobs_folder / "results").mkdir(exist_ok=True)
    (jobs_folder / "logs").mkdir(exist_ok=True)

    (jobs_folder / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8"
    )
    (jobs_folder / "manifest.lock").touch()

    print(f"Initialized {len(items)} items in {jobs_folder}")
    print(f"Smallest: {items[0]['filename']} ({items[0]['file_size']} bytes)")
    print(f"Largest:  {items[-1]['filename']} ({items[-1]['file_size']} bytes)")
    return manifest


# ============= GET NEXT ITEM =============

def get_next_item(config, worker_id=None):
    """
    Claim the next available item. Returns a dict with item info,
    or None if all work is done.
    """
    jobs_folder = Path(config["jobs_folder"])
    worker_id = worker_id or _generate_worker_id()

    manifest, lock_fh = _acquire_manifest(jobs_folder)
    try:
        input_folder = Path(manifest["input_folder"])
        queue = manifest["item_queue"]

        # Reset stale in-progress items
        _reset_stale_items(manifest)

        # Scan from current_item_index forward
        for idx in range(manifest["current_item_index"], len(queue)):
            item_filename = queue[idx]
            item_info = manifest["items"][item_filename]

            if item_info["status"] == "completed":
                continue

            if item_info["status"] in ("pending", "failed"):
                # Claim it
                item_info["status"] = "in_progress"
                item_info["claimed_at"] = _now()
                item_info["worker_id"] = worker_id

                input_path = input_folder / item_filename

                _release_manifest(jobs_folder, manifest, lock_fh)
                lock_fh = None

                return {
                    "item_filename": item_filename,
                    "input_path": str(input_path),
                    "worker_id": worker_id,
                }

            # Item is in_progress (claimed by another worker) — skip
            continue

        # Advance current_item_index past completed items
        while (manifest["current_item_index"] < len(queue) and
               manifest["items"][queue[manifest["current_item_index"]]]["status"] == "completed"):
            manifest["current_item_index"] += 1

        # Nothing left
        _release_manifest(jobs_folder, manifest, lock_fh)
        lock_fh = None
        return None

    finally:
        if lock_fh is not None:
            _unlock_manifest(lock_fh)


# ============= SUBMIT ITEM =============

def submit_item(config, item_filename, result_file):
    """
    Submit a completed result for an item. Copies the result file to
    jobs_folder/results/ and marks the item as completed.
    """
    jobs_folder = Path(config["jobs_folder"])
    result_file = Path(result_file)

    # Normalize filenames for macOS NFC/NFD compatibility
    item_nfc = unicodedata.normalize("NFC", item_filename)
    item_nfd = unicodedata.normalize("NFD", item_filename)

    manifest, lock_fh = _acquire_manifest(jobs_folder)
    try:
        if item_nfc in manifest["items"]:
            item_filename = item_nfc
        elif item_nfd in manifest["items"]:
            item_filename = item_nfd
        else:
            _unlock_manifest(lock_fh)
            lock_fh = None
            return {"error": f"Item not found: {item_filename}"}

        item_info = manifest["items"][item_filename]

        # Idempotent: skip if already completed
        if item_info["status"] == "completed":
            _release_manifest(jobs_folder, manifest, lock_fh)
            lock_fh = None
            return {"status": "completed", "already_completed": True}

        # Mark completed
        item_info["status"] = "completed"
        item_info["completed_at"] = _now()

        # Advance current_item_index past completed items
        queue = manifest["item_queue"]
        while (manifest["current_item_index"] < len(queue) and
               manifest["items"][queue[manifest["current_item_index"]]]["status"] == "completed"):
            manifest["current_item_index"] += 1

        # Check if all items are done
        all_done = all(
            info["status"] == "completed"
            for info in manifest["items"].values()
        )

        on_complete = manifest.get("on_complete")

        _release_manifest(jobs_folder, manifest, lock_fh)
        lock_fh = None

        # Copy result file to results/ (outside lock)
        if result_file.exists():
            dest = jobs_folder / "results" / item_filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(result_file), str(dest))

        # Run on_complete hook if all done
        if all_done and on_complete:
            _run_on_complete(config, on_complete)

        return {"status": "completed", "already_completed": False, "all_done": all_done}

    finally:
        if lock_fh is not None:
            _unlock_manifest(lock_fh)


# ============= FAIL ITEM =============

def fail_item(config, item_filename, error=""):
    """Mark an item as failed so another worker can retry it."""
    jobs_folder = Path(config["jobs_folder"])

    item_nfc = unicodedata.normalize("NFC", item_filename)
    item_nfd = unicodedata.normalize("NFD", item_filename)

    manifest, lock_fh = _acquire_manifest(jobs_folder)
    try:
        if item_nfc in manifest["items"]:
            item_filename = item_nfc
        elif item_nfd in manifest["items"]:
            item_filename = item_nfd
        else:
            _unlock_manifest(lock_fh)
            lock_fh = None
            return {"error": f"Item not found: {item_filename}"}

        item_info = manifest["items"][item_filename]
        item_info["status"] = "failed"
        item_info["error"] = error
        item_info["failed_at"] = _now()
        _release_manifest(jobs_folder, manifest, lock_fh)
        lock_fh = None
    finally:
        if lock_fh is not None:
            _unlock_manifest(lock_fh)


# ============= ON COMPLETE HOOK =============

def _run_on_complete(config, on_complete_template):
    """Run the on_complete shell command with variable substitution."""
    cmd = on_complete_template.format(
        jobs_folder=config["jobs_folder"],
        input_folder=config["input_folder"],
        output_folder=config.get("output_folder", ""),
    )
    print(f"Running on_complete hook: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"on_complete hook failed (exit {result.returncode}): {result.stderr}", file=sys.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"on_complete hook error: {e}", file=sys.stderr)
        return False


# ============= STATUS =============

def get_status(config):
    """Return a progress report dict."""
    jobs_folder = Path(config["jobs_folder"])

    if not (jobs_folder / "manifest.json").exists():
        return {"error": f"Not initialized. Run: python job_queue.py init --task {config['task_name']}"}

    manifest, lock_fh = _acquire_manifest(jobs_folder)
    try:
        total = len(manifest["item_queue"])
        completed = 0
        in_progress = 0
        failed = 0
        pending = 0

        for item_info in manifest["items"].values():
            s = item_info["status"]
            if s == "completed":
                completed += 1
            elif s == "in_progress":
                in_progress += 1
            elif s == "failed":
                failed += 1
            else:
                pending += 1

        _unlock_manifest(lock_fh)
        lock_fh = None

        return {
            "task_name": manifest["task_name"],
            "items_total": total,
            "items_completed": completed,
            "items_in_progress": in_progress,
            "items_failed": failed,
            "items_pending": pending,
            "completion_pct": round(completed / total * 100, 2) if total else 0,
        }
    finally:
        if lock_fh is not None:
            _unlock_manifest(lock_fh)


def print_status(config):
    """Pretty-print the progress report."""
    s = get_status(config)
    if "error" in s:
        print(s["error"])
        return

    print(f"\n{'='*50}")
    print(f"JOB STATUS: {s['task_name']}")
    print(f"{'='*50}")
    print(f"Items: {s['items_completed']} done / {s['items_in_progress']} active / {s['items_failed']} failed / {s['items_pending']} pending  (total: {s['items_total']})")
    print(f"Progress: {s['completion_pct']}%")
    print(f"{'='*50}\n")


# ============= RESET =============

def reset_items(config, item_filename=None):
    """Reset failed and stale in-progress items back to pending."""
    jobs_folder = Path(config["jobs_folder"])

    manifest, lock_fh = _acquire_manifest(jobs_folder)
    try:
        reset_count = 0
        items_to_check = [item_filename] if item_filename else manifest["item_queue"]

        for filename in items_to_check:
            if filename not in manifest["items"]:
                continue
            item_info = manifest["items"][filename]
            if item_info["status"] in ("failed", "in_progress"):
                item_info["status"] = "pending"
                item_info["claimed_at"] = None
                item_info["worker_id"] = None
                item_info.pop("error", None)
                item_info.pop("failed_at", None)
                reset_count += 1

        _release_manifest(jobs_folder, manifest, lock_fh)
        lock_fh = None
        print(f"Reset {reset_count} items to pending.")
    finally:
        if lock_fh is not None:
            _unlock_manifest(lock_fh)


# ============= CLI =============

def main():
    parser = argparse.ArgumentParser(description="Generic agent job queue")
    parser.add_argument("--task", required=True, help="Task name (matches tasks/<name>.json)")
    # `--variant` is global so every subcommand consults the same resolved
    # config (init / next / submit / status / reset all read it). Empty
    # → legacy flat layout. A task JSON with no `{variant}` placeholder
    # ignores it harmlessly.
    # Default falls back to AGENT_QUEUE_VARIANT so the launcher can inject
    # the variant into the agent subprocess environment without each agent
    # needing to learn the flag (their hardcoded `--task X` commands then
    # pick up the variant transparently).
    parser.add_argument(
        "--variant", default=os.environ.get("AGENT_QUEUE_VARIANT") or None,
        help="Path subfolder; substituted into {variant} in task JSON paths.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # init
    subparsers.add_parser("init", help="Initialize job queue from input folder")

    # next
    p_next = subparsers.add_parser("next", help="Claim next item (JSON to stdout)")
    p_next.add_argument("--worker-id", default=None)

    # submit
    p_submit = subparsers.add_parser("submit", help="Submit completed item result")
    p_submit.add_argument("--item", required=True, help="Item filename")
    p_submit.add_argument("--file", required=True, help="Path to result file")

    # fail
    p_fail = subparsers.add_parser("fail", help="Mark item as failed")
    p_fail.add_argument("--item", required=True, help="Item filename")
    p_fail.add_argument("--error", default="", help="Error description")

    # status
    p_status = subparsers.add_parser("status", help="Print progress report")
    p_status.add_argument("--json", action="store_true")

    # reset
    p_reset = subparsers.add_parser("reset", help="Reset failed/stale items")
    p_reset.add_argument("--item", default=None, help="Specific item to reset")

    # paths — prints resolved folder paths as one JSON line. Used by the
    # launcher so it shares the variant-substitution logic instead of
    # re-implementing it in bash.
    subparsers.add_parser("paths", help="Print resolved input/output/jobs paths as JSON")

    args = parser.parse_args()
    config = load_config(args.task, variant=args.variant)

    if args.command == "init":
        init_jobs(config)

    elif args.command == "next":
        item = get_next_item(config, args.worker_id)
        if item:
            print(json.dumps(item, ensure_ascii=False, indent=2))
        else:
            print('{"done": true}')

    elif args.command == "submit":
        result = submit_item(config, args.item, args.file)
        print(json.dumps(result))

    elif args.command == "fail":
        fail_item(config, args.item, args.error)

    elif args.command == "status":
        if args.json:
            print(json.dumps(get_status(config), indent=2))
        else:
            print_status(config)

    elif args.command == "reset":
        reset_items(config, args.item)

    elif args.command == "paths":
        print(json.dumps({
            "input_folder": str(config["input_folder"]) if config["input_folder"] else "",
            "output_folder": str(config["output_folder"]) if config["output_folder"] else "",
            "jobs_folder": str(config["jobs_folder"]) if config["jobs_folder"] else "",
            "agent": config["agent"],
            "model": config["model"],
            "variant": config.get("variant", ""),
        }))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
