"""Wait for idle GPUs and save teacher slot acceptance artifacts under one output root."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / 'training/verl'))
from verl.workers.multi_gpu.cpu_state_store import CPUStateStore, ENV_NAME

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--model-root')
parser.add_argument('--closed-loop', action='store_true')
args = parser.parse_args()
root = Path(args.output)
for name in ('logs', 'reports', 'profiles'):
    (root / name).mkdir(parents=True, exist_ok=True)
if (root / 'logs/test.log').exists():
    raise FileExistsError(root / 'logs/test.log')
since = None
with (root / 'profiles/idle_gate.log').open('w') as log:
    while True:
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True).strip().splitlines()
        idle = not apps and len(rows) == 4 and all(int(r.split(',')[0]) < 256 and int(r.split(',')[1]) < 5 for r in rows)
        since = (since or time.monotonic()) if idle else None
        elapsed = time.monotonic() - since if since else 0
        line = f'{time.time()} idle={elapsed:.0f}s apps={apps!r} GPUs={rows!r}'
        log.write(line + '\n')
        log.flush()
        print(line, flush=True)
        if elapsed >= 60:
            break
        time.sleep(10)
owner = CPUStateStore.create()
command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
           str(repo / 'training/verl/tests/multi_gpu/test_teacher_slots.py'),
           '--output', str(root / 'reports/metrics.json')]
if args.model_root:
    command += ['--model-root', args.model_root]
if args.closed_loop:
    command = [sys.executable, str(repo / 'training/verl/tests/multi_gpu/test_shared_cpu_state.py'),
               '--gpu', '--rollout', '--mopd']
(root / 'reports/config.txt').write_text(repr(command) + '\n')
try:
    with (root / 'logs/test.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   env=dict(os.environ, **{ENV_NAME: owner.namespace}, OMP_NUM_THREADS='1'))
        try:
            code = process.wait(timeout=1800)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            raise
    print(f'teacher slots exit={code}', flush=True)
    raise SystemExit(code)
finally:
    owner.unlink()
