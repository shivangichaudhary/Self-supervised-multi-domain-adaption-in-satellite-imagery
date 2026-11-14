"""Run the Objective 3 / 4 experiment queue unattended.

Every job is retried (up to MAX_TRIES) if it exits non-zero or is killed. The experiment scripts reuse
finished generator checkpoints (Objective 3) and finished seeds (Objective 4), so a retry resumes
instead of starting over. A job may name a file it has to wait for (e.g. a converted dataset that is
still being built). Progress is appended to results/queue_status.txt.

    python experiments/run_queue.py
"""
import datetime
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_TRIES = 6
# the queue itself is started with pythonw.exe (no console); the jobs use the console build of the same interpreter
PYTHON = os.path.join(os.path.dirname(sys.executable), "python.exe") if sys.executable.lower().endswith("pythonw.exe") else sys.executable
CKPT = "results/objective3/ckpt"
# (log file, command, file that must exist before the job starts or None)
JOBS = [
    # Objective 3: generators + fidelity only (leave-one-pair-out); seeds 0 and 1 are already finished
    # (seed 1 finished 2026-11-12 15:27, exit 0, so it is no longer queued)
    # (2026-11-14: Objective 3 now runs first because the user asked to complete it; Objective 4 seed 2 follows)
    ("results/objective3/log.txt", ["experiments/objective3.py", "--seeds", "2", "--no_downstream"], None),
    # zero-shot SpaceNet 6 with every "full" generator (+ the published Objective 2 one)
    ("results/objective3/log_zero_shot_sn6.txt",
     ["experiments/zero_shot_sn6.py", "--sn6", "data/spacenet6_dataset", "--gens", "published=outputs_pastis_fixed/generator.pth",
      f"full_seed0={CKPT}/seed0_full/generator.pth", f"full_seed1={CKPT}/seed1_full/generator.pth",
      f"full_seed2={CKPT}/seed2_full/generator.pth"], "data/spacenet6_dataset/patches.json"),
    # Objective 4: downstream segmentation on the growing-season dataset (see data/convert_pastis_season.py);
    # seeds 0 and 1 are finished and are skipped, so this only runs seed 2
    ("results/objective4/log.txt", ["experiments/objective4.py", "--seeds", "0", "1", "2", "--epochs", "40",
                                    "--data_dir", "data/france_dataset_season"], "data/france_dataset_season/READY"),
]


def status(msg):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(os.path.join(ROOT, "results", "queue_status.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    if sys.stdout is None or sys.stderr is None:   # pythonw.exe has no console, so give print() a file
        sys.stdout = sys.stderr = open(os.path.join(ROOT, "results", "queue_stdout.txt"), "a", encoding="utf-8", buffering=1)
    for log_rel, argv, wait_for in JOBS:
        log = os.path.join(ROOT, log_rel)
        os.makedirs(os.path.dirname(log), exist_ok=True)
        name = " ".join(argv)[:160]
        if wait_for and not os.path.exists(os.path.join(ROOT, wait_for)):
            status(f"WAIT  for {wait_for} before {name}")
            while not os.path.exists(os.path.join(ROOT, wait_for)):
                time.sleep(60)
        for attempt in range(1, MAX_TRIES + 1):
            status(f"START {name} (attempt {attempt})")
            with open(log, "a", encoding="utf-8") as lf:
                lf.write(f"\n##### {datetime.datetime.now():%Y-%m-%d %H:%M:%S} attempt {attempt}: {name}\n")
                lf.flush()
                # The jobs run with no console at all (DETACHED_PROCESS) and in their own process group, so neither a
                # Ctrl+C nor a "window closed" event aimed at a console can reach them (that killed the queue twice
                # on 2026-11-12: "^C" at 20:54, "forrtl: window-CLOSE event" at 23:02).
                flags = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0))
                rc = subprocess.run([PYTHON, "-u", *argv], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=lf,
                                    stderr=subprocess.STDOUT, creationflags=flags).returncode
            status(f"END   {name} exit={rc}")
            if rc == 0:
                break
        else:
            status(f"GIVE UP on {name} after {MAX_TRIES} attempts; continuing with the next job")
    status("QUEUE DONE")


if __name__ == "__main__":
    main()
