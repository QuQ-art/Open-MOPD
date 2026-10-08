#!/usr/bin/env python3
"""Run the four teacher-prefetch / optimizer-offload combinations sequentially."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import threading
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
SOURCES = [
    'scripts/local/profile_native_single_gpu.sh', 'scripts/local/profile_overlap_matrix.py',
    'training/verl/verl/trainer/ppo/ray_trainer.py',
    'training/verl/verl/trainer/config/rollout/rollout.yaml',
    'training/verl/verl/workers/config/rollout.py',
    'training/verl/verl/workers/fsdp_workers.py',
    'training/verl/verl/workers/actor/dp_actor.py',
    'training/verl/verl/workers/teacher_param_prefetch.py',
    'training/verl/verl/workers/attention_metadata.py',
    'training/verl/verl/workers/optimizer_offload_overlap.py',
    'training/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py',
]
CONFIG = dict(
    CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_DEVICE_MAX_CONNECTIONS='8',
    PROFILE_BATCH_SIZE='4', PROFILE_DATA_SEED='42', PROFILE_RAY_CPUS='8',
    TEACHER_PARAM_PREFETCH_HANDLES='37', TEACHER_PARAM_PREFETCH_MAX_MB='6144', CODE_TEACHER_PARAM_PREFETCH_HANDLES='37',
    CODE_TEACHER_PARAM_PREFETCH_MAX_MB='6144', TEACHER_ATTENTION_METADATA_CACHE='true',
    OPTIMIZER_OFFLOAD_OVERLAP_CHUNK_MB='32', ROLLOUT_GPU_MEMORY_UTILIZATION='0.35',
    ROLLOUT_IGNORE_EOS='false', SHARE_STUDENT_WEIGHTS='false', BF16_STUDENT_WEIGHTS='true',
    OPENMOPD_VAL_BEFORE_TRAIN='false', OPENMOPD_PROFILE_STEP='3',
    OPENMOPD_PROFILE_STEP_COUNT='2', OPENMOPD_NUM_STEPS='4',
)
CASES = [('00_off_off', 'false', 'false'), ('10_teacher_only', 'true', 'false'),
         ('01_optimizer_only', 'false', 'true'), ('11_both', 'true', 'true')]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def gpu_sample():
    result = subprocess.run(
        ['nvidia-smi', '--id=0', '--query-gpu=memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    memory, utilization = map(int, result.stdout.strip().split(','))
    return dict(time=time.time(), memory_MiB=memory, utilization_pct=utilization)


def wait_idle(seconds):
    deadline = time.monotonic() + seconds
    while True:
        sample = gpu_sample()
        if sample['memory_MiB'] < 1024 and sample['utilization_pct'] < 5:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f'GPU 0 is busy; no workload was stopped: {sample}')
        print(f'Waiting for GPU 0: {sample}', flush=True)
        time.sleep(min(10, max(0, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='New output directory; existing results are never overwritten')
    parser.add_argument('--wait-seconds', type=int, default=0, help='Wait for GPU 0 before each case')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CONFIG)
    for key in ('CUDA_DEVICE_MAX_COPY_CONNECTIONS', 'CUDA_LAUNCH_BLOCKING',
                'TEACHER_FORWARD_OVERLAP', 'CODE_TEACHER_PARAM_PREFETCH', 'CODE_TEACHER_PARAM_PREFETCH_TRIGGER'):
        env.pop(key, None)
    hashes = {}
    for name in SOURCES:
        source = REPO / name
        target = root / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = dict(status='waiting_for_gpu', common_config=CONFIG, source_sha256=hashes, cases=[])
    save(root / 'manifest.json', manifest)
    try:
        for name, teacher, optimizer in CASES:
            wait_idle(args.wait_seconds)
            if any(hashlib.sha256((REPO / p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
                raise RuntimeError('Source changed since snapshot; refusing to mix implementations')
            folder = root / name
            folder.mkdir()
            overrides = dict(TEACHER_PARAM_PREFETCH=teacher, OPTIMIZER_OFFLOAD_OVERLAP=optimizer)
            case = dict(name=name, switches=overrides, status='running')
            manifest['cases'].append(case)
            manifest['status'] = 'running'
            save(root / 'manifest.json', manifest)
            print(f'Starting {name}: {overrides}; log: {folder}/launcher.log', flush=True)
            with (folder / 'launcher.log').open('w') as log, (folder / 'gpu_memory.jsonl').open('w') as memory:
                process = subprocess.Popen(
                    ['bash', 'scripts/local/profile_native_single_gpu.sh', str(folder)],
                    cwd=REPO, env={**env, **overrides}, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
                def forward_log():
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end='', flush=True)
                reader = threading.Thread(target=forward_log, daemon=True)
                reader.start()
                started = time.monotonic()
                next_progress = started
                try:
                    while process.poll() is None:
                        sample = gpu_sample()
                        memory.write(json.dumps(sample) + '\n')
                        if time.monotonic() >= next_progress:
                            print(f"[{name}] elapsed={time.monotonic() - started:.0f}s "
                                  f"GPU0={sample['memory_MiB']}MiB/{sample['utilization_pct']}%", flush=True)
                            next_progress = time.monotonic() + 30
                        memory.flush()
                        time.sleep(2)
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    case.update(status='interrupted', exit_code=process.returncode)
                    raise
                finally:
                    reader.join(timeout=10)
            reports = list((folder / 'traces').glob('*.nsys-rep'))
            unchanged = all(hashlib.sha256((REPO / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
            case.update(exit_code=process.returncode, source_unchanged=unchanged,
                        reports=[str(p.relative_to(root)) for p in reports], status='completed')
            if process.returncode or not reports or not unchanged:
                case['status'] = 'failed'
                raise RuntimeError(f'{name} failed validation; see {folder}/launcher.log')
            save(folder / 'manifest.json', {**case, 'config': {**CONFIG, **overrides}, 'source_sha256': hashes})
            save(root / 'manifest.json', manifest)
            print(f'Completed {name}', flush=True)
        manifest['status'] = 'completed'
    except BaseException as error:
        manifest.update(status='incomplete', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save(root / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
