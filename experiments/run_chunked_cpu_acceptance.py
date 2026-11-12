"""Run chunked CPU reduce acceptance only after all four GPUs are idle."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(sys.argv[1])
repo = Path(__file__).resolve().parents[1]
(root / 'logs').mkdir(parents=True, exist_ok=True)

def idle_gate():
    since = None
    while True:
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True).strip().splitlines()
        idle = not apps and len(rows) == 4 and all(int(r.split(',')[0]) < 256 and int(r.split(',')[1]) < 5 for r in rows)
        since = (since or time.monotonic()) if idle else None
        elapsed = time.monotonic() - since if since else 0
        print(f'idle gate {elapsed:.0f}s; apps={apps!r}', flush=True)
        if elapsed >= 60:
            return
        time.sleep(10)

tests = [] if '--production-only' in sys.argv[2:] else [('chunk_boundaries', ['--gpu']), ('mopd', ['--gpu', '--rollout', '--mopd'])]
for name, flags in tests:
    idle_gate()
    with (root / f'logs/{name}.log').open('w') as fp:
        result = subprocess.run([sys.executable, str(repo / 'training/verl/tests/multi_gpu/test_shared_cpu_state.py'),
                                 *flags], stdout=fp, stderr=subprocess.STDOUT)
    print(json.dumps({'test': name, 'exit_code': result.returncode}), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)

for phase, label in [('train', 'train4096_shared'), ('resume', 'resume4096_shared')]:
    command = [sys.executable, str(repo / 'experiments/run_phase4_continuous.py'), '--output', str(root),
               '--phase', phase, '--label', label, '--checkpoint-label', 'train4096_shared',
               '--kv-mib', '2048', '--entropy-chunk-size', '256', '--gpu-memory-utilization', '0.30', '--share-weights']
    subprocess.run(command, check=True)
