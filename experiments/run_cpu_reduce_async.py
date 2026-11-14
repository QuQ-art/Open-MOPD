"""Correctness ladder and paired benchmark, only after a stable four-GPU idle window."""
import argparse
import json
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
parser.add_argument('--benchmark-only', action='store_true')
parser.add_argument('--closed-loop-only', action='store_true')
args = parser.parse_args()
root = Path(args.output)
for name in ('logs', 'reports', 'profiles'):
    (root / name).mkdir(parents=True, exist_ok=True)
if (root / 'logs/benchmark.log').exists():
    raise FileExistsError(root)
since = None
with (root / 'profiles/idle_gate.log').open('w') as fp:
    while True:
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True).strip().splitlines()
        idle = not apps and len(rows) == 4 and all(int(r.split(',')[0]) < 256 and int(r.split(',')[1]) < 5 for r in rows)
        since = (since or time.monotonic()) if idle else None
        elapsed = time.monotonic() - since if since else 0
        text = f'idle={elapsed:.0f}s apps={apps!r} gpus={rows}'
        print(text, flush=True)
        fp.write(text + '\n'); fp.flush()
        if elapsed >= 60:
            break
        time.sleep(10)
results = []

def run(name, command, env):
    with (root / f'logs/{name}.log').open('w') as fp:
        process = subprocess.Popen(command, env=env, stdout=fp, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=2400)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            raise
    results.append(dict(name=name, exit_code=code, command=command))
    (root / 'reports/processes.json').write_text(json.dumps(results, indent=2))
    print(f'{name}: {code}', flush=True)
    if code:
        raise SystemExit(code)

if args.closed_loop_only:
    run('closed_loop', [sys.executable, str(repo / 'training/verl/tests/multi_gpu/test_shared_cpu_state.py'),
                        '--gpu', '--rollout', '--mopd'], dict(os.environ, OPENMOPD_CPU_REDUCE_MODE='async'))
    raise SystemExit(0)

if not args.benchmark_only:
    for mode in ('serial', 'double_buffer', 'async'):
        run(mode, [sys.executable, str(repo / 'training/verl/tests/multi_gpu/test_shared_cpu_state.py'), '--gpu'],
            dict(os.environ, OPENMOPD_CPU_REDUCE_MODE=mode))
owner = CPUStateStore.create()
try:
    run('benchmark', [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                     str(repo / 'training/verl/tests/multi_gpu/benchmark_cpu_reduce.py'),
                     '--output', str(root / 'reports/benchmark.json')],
        dict(os.environ, **{ENV_NAME: owner.namespace}, OMP_NUM_THREADS='8'))
finally:
    owner.unlink()
