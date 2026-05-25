#!/usr/bin/env python3
"""Closed-loop feeding orchestrator.

Loop:
  1. run_scoop.py      — scoop sequence
  2. click_pointer.py  — click-to-feed (close window with Q / ESC to continue)
  3. 10-second pause
  Repeat until Ctrl+C on THIS script.
"""
import subprocess
import sys
import os
import signal
import time

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SCOOP_CMD = [sys.executable, os.path.join(BASE_DIR, 'run_scoop.py')]
CLICK_CMD = [sys.executable,
             os.path.join(BASE_DIR, 'src/click_pointer/scripts/click_pointer.py')]

PAUSE_SEC = 10
_stop     = False
_proc     = None


def _sigint(sig, frame):
    global _stop
    _stop = True
    print('\n[loop_feed] Ctrl+C caught — will stop after current step.')
    if _proc is not None:
        try:
            _proc.terminate()
        except Exception:
            pass


signal.signal(signal.SIGINT, _sigint)


def run(cmd, label):
    global _proc
    if _stop:
        return
    print(f'\n{"="*60}')
    print(f'[loop_feed] Starting: {label}')
    print(f'{"="*60}', flush=True)

    # start_new_session=True isolates the child from this process group
    # so Ctrl+C on the child does NOT propagate to loop_feed.py
    proc = subprocess.Popen(cmd, cwd=BASE_DIR, start_new_session=True)
    _proc = proc
    proc.wait()
    _proc = None


def pause():
    if _stop:
        return
    print(f'\n[loop_feed] Next cycle in {PAUSE_SEC}s  (Ctrl+C to stop)...', flush=True)
    for i in range(PAUSE_SEC, 0, -1):
        if _stop:
            break
        print(f'\r[loop_feed]  {i:2d}s ', end='', flush=True)
        time.sleep(1)
    print()


def main():
    cycle = 0
    print('[loop_feed] Closed-loop feeding started. Ctrl+C to stop.')
    while not _stop:
        cycle += 1
        print(f'\n[loop_feed] ══ Cycle {cycle} ══')
        run(SCOOP_CMD, 'run_scoop.py')
        if _stop:
            break
        run(CLICK_CMD, 'click_pointer.py')
        if _stop:
            break
        pause()

    print('[loop_feed] Stopped.')


if __name__ == '__main__':
    main()
