"""Production 4096-token continuous/resume acceptance with sampled GPU memory."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import signal
import psutil
import threading
import time
from zoneinfo import ZoneInfo

parser = argparse.ArgumentParser()
parser.add_argument('--output')
parser.add_argument('--label', default='train4096')
parser.add_argument('--phase', choices=['train', 'resume'], default='train')
parser.add_argument('--checkpoint-label', default='train4096')
parser.add_argument('--kv-mib', type=int, default=4096)
parser.add_argument('--remove-padding', action='store_true')
parser.add_argument('--entropy-chunk-size', type=int, default=0)
parser.add_argument('--gpu-memory-utilization', type=float, default=0.5)
parser.add_argument('--share-weights', action='store_true')
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
root = Path(args.output) if args.output else Path('/mnt/sda1/zst/RONDO/experiments') / (
    'phase4_continuous_' + datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S'))
for folder in ['logs', 'reports', 'profiles', 'checkpoints']:
    (root / folder).mkdir(parents=True, exist_ok=True)
print(root, flush=True)
label = args.label
log_path = root / f'logs/{label}.log'
if log_path.exists():
    raise FileExistsError(log_path)
ray_dir = root / f'logs/ray_{label}'
ray_dir.mkdir()
ray_link = Path('/tmp') / f'p4c_{os.getpid()}'
ray_link.symlink_to(ray_dir, target_is_directory=True)
checkpoint = root / 'checkpoints' / args.checkpoint_label
metrics_path = root / f'reports/{label}.jsonl'
env = dict(os.environ, OPENMOPD_PYTHON='/home/zst/miniconda3/envs/open_mopd/bin/python',
    OPENMOPD_MODEL_ROOT='/home/xxf/Distill/models', OPENMOPD_N_GPUS_PER_NODE='4',
    OPENMOPD_TRAIN_BATCH_SIZE='4', OPENMOPD_NUM_STEPS='3' if args.phase == 'train' else '5',
    OPENMOPD_KV_CACHE_MB=str(args.kv_mib), OPENMOPD_MAX_NUM_SEQS='1',
    OPENMOPD_VAL_BEFORE_TRAIN='false', ENFORCE_EAGER='true', STEP_MEMORY_ENABLED='false',
    CUDA_VISIBLE_DEVICES='0,1,2,3', VERL_FILE_LOGGER_PATH=str(metrics_path),
    HYDRA_FULL_ERROR='1', OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
overrides = ['+ray_kwargs.ray_init._temp_dir=' + str(ray_link),
    'actor_rollout_ref.rollout.gpu_memory_utilization=' + str(args.gpu_memory_utilization),
    'data.max_response_length=4096', 'actor_rollout_ref.rollout.max_model_len=5120',
    'actor_rollout_ref.rollout.max_num_batched_tokens=5120',
    'trainer.save_freq=3', 'trainer.test_freq=0',
    'trainer.resume_mode=' + ('disable' if args.phase == 'train' else 'resume_path'),
    'trainer.default_local_dir=' + str(checkpoint), 'trainer.experiment_name=' + root.name + '_' + label,
    'trainer.logger=[console,file]', 'global_profiler.save_path=' + str(root / 'profiles'),
    'hydra.run.dir=' + str(root / f'logs/hydra_{label}')]
if args.share_weights:
    env['SHARE_STUDENT_WEIGHTS'] = 'true'
if args.phase == 'resume':
    overrides += ['trainer.resume_from_path=' + str(checkpoint / 'global_step_3')]
if args.entropy_chunk_size:
    overrides += ['actor_rollout_ref.actor.entropy_from_logits_with_chunking=true',
                  '+actor_rollout_ref.actor.entropy_chunk_size=' + str(args.entropy_chunk_size),
                  'actor_rollout_ref.actor.use_torch_compile=false']
if args.remove_padding:
    overrides += ['actor_rollout_ref.model.use_remove_padding=true', 'reward_model.model.use_remove_padding=true']
base = ['bash', str(repo / 'r2.sh')]
with (root / f'reports/{label}_config.txt').open('w') as fp:
    fp.write(json.dumps(vars(args), indent=2) + '\n'); fp.flush()
    subprocess.run(base + ['--dry-run'] + overrides, env=env, cwd=repo, stdout=fp, stderr=subprocess.STDOUT, check=True)
    fp.write('\nResolved config:\n'); fp.flush()
    subprocess.run(base + overrides + ['--cfg', 'job', '--resolve'], env=env, cwd=repo,
                   stdout=fp, stderr=subprocess.STDOUT, check=True)
# Do not overlap another user's GPU job. Require a stable idle window immediately before launch.
idle_since = None
with (root / f'profiles/{label}_idle_gate.jsonl').open('w') as idle_log:
    while True:
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                        '--format=csv,noheader,nounits'], text=True).strip()
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True)
        readings = [[int(x.strip()) for x in line.split(',')] for line in raw.strip().splitlines()]
        idle = not apps and len(readings) == 4 and all(mem < 256 and util < 5 for mem, util in readings)
        idle_since = (idle_since if idle_since is not None else time.monotonic()) if idle else None
        elapsed = time.monotonic() - idle_since if idle_since is not None else 0
        idle_log.write(json.dumps({'timestamp':time.time(), 'apps':apps, 'gpus':readings,
                                   'idle_seconds':elapsed}) + '\n'); idle_log.flush()
        print(f'GPU idle gate: idle={idle}, stable_seconds={elapsed:.0f}', flush=True)
        if elapsed >= 60:
            break
        time.sleep(10)
stop = threading.Event()
process = None
interference = []

def monitor():
    known_ours = {os.getpid()}
    with (root / f'profiles/{label}_gpu.jsonl').open('w') as fp:
        while not stop.is_set():
            try:
                raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True, timeout=10)
                apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                                '--format=csv,noheader,nounits'], text=True, timeout=10)
                known_ours.update(p.pid for p in psutil.Process().children(recursive=True))
                ours = known_ours
                foreign = sorted({int(pid.strip()) for pid in apps.splitlines() if pid.strip().isdigit()} - ours)
                if foreign and process is not None and process.poll() is None:
                    interference.append({'timestamp':time.time(), 'pids':foreign})
                    print(f'External GPU processes detected: {foreign}; interrupting this experiment only', flush=True)
                    process.send_signal(signal.SIGINT)
                row = {'timestamp':time.time(), 'foreign_pids':foreign, 'gpus':[[int(x.strip()) for x in line.split(',')]
                       for line in raw.strip().splitlines()]}
                fp.write(json.dumps(row) + '\n'); fp.flush()
            except Exception as error:
                fp.write(json.dumps({'timestamp':time.time(), 'error':str(error)}) + '\n'); fp.flush()
            stop.wait(2)
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
start = time.time()
try:
    with log_path.open('w') as fp:
        process = subprocess.Popen(base + overrides, env=env, cwd=repo, stdout=fp, stderr=subprocess.STDOUT)
        (root / f'reports/{label}_process.json').write_text(json.dumps({'pid':process.pid, 'args':vars(args)}))
        code = process.wait()
finally:
    stop.set(); thread.join(timeout=15)
rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line] if metrics_path.exists() else []
report = {'exit_code':code, 'elapsed_seconds':time.time()-start, 'options':vars(args), 'metrics':rows, 'external_gpu_interference':interference}
(root / f'reports/{label}_metrics.json').write_text(json.dumps(report, indent=2))
print(json.dumps({'exit_code':code, 'steps':[r['step'] for r in rows], 'root':str(root)}), flush=True)
raise SystemExit(code)
