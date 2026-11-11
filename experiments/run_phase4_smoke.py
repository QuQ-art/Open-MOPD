"""Launch the production one-step smoke test and preserve all reports."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

repo = Path(__file__).resolve().parents[1]
name = 'phase4_production_' + datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/mnt/sda1/zst/RONDO/experiments') / name
name = output.name
for part in ('logs', 'checkpoints', 'reports', 'profiles'):
    (output / part).mkdir(parents=True, exist_ok=True)
print(output, flush=True)
if (output / 'logs/train.log').exists():
    attempt = len(list((output / 'logs').glob('train.attempt*.log'))) + 1
    (output / 'logs/train.log').rename(output / f'logs/train.attempt{attempt}.log')
    for file in ('config.txt', 'metrics.json', 'summary.md'):
        source = output / 'reports' / file
        if source.exists():
            source.rename(source.with_name(f'attempt{attempt}.' + file))
# Short path avoids AF_UNIX socket length limits; files live in the requested output tree.
ray_link = Path('/tmp') / ('p4ray_' + str(os.getpid()))
(output / 'logs/ray').mkdir(exist_ok=True)
ray_link.symlink_to(output / 'logs/ray', target_is_directory=True)
env = dict(os.environ, OPENMOPD_PYTHON='/home/zst/miniconda3/envs/open_mopd/bin/python',
           OPENMOPD_MODEL_ROOT='/home/xxf/Distill/models', OPENMOPD_N_GPUS_PER_NODE='4',
           OPENMOPD_TRAIN_BATCH_SIZE='4', OPENMOPD_NUM_STEPS='1', OPENMOPD_KV_CACHE_MB='2048',
           OPENMOPD_MAX_NUM_SEQS='1', OPENMOPD_VAL_BEFORE_TRAIN='false',
           ENFORCE_EAGER='true', STEP_MEMORY_ENABLED='false', CUDA_VISIBLE_DEVICES='0,1,2,3',
           VERL_FILE_LOGGER_PATH=str(output / 'reports/metrics.jsonl'),
           HYDRA_FULL_ERROR='1', OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
overrides = ['+ray_kwargs.ray_init._temp_dir=' + str(ray_link),
             'data.max_response_length=512', 'actor_rollout_ref.rollout.max_model_len=1536',
             'actor_rollout_ref.rollout.max_num_batched_tokens=1536',
             'trainer.save_freq=0', 'trainer.test_freq=0', 'trainer.resume_mode=disable',
             'trainer.default_local_dir=' + str(output / 'checkpoints'),
             'trainer.experiment_name=' + name, 'trainer.logger=[console,file]',
             'global_profiler.save_path=' + str(output / 'profiles'),
             'hydra.run.dir=' + str(output / 'logs/hydra')]
base = ['bash', str(repo / 'r2.sh')]
with (output / 'reports/config.txt').open('w') as fp:
    fp.write('Environment overrides:\n' + json.dumps({k:v for k,v in env.items() if os.environ.get(k) != v}, indent=2))
    fp.write('\n\nCommand:\n'); fp.flush()
    subprocess.run(base + ['--dry-run'] + overrides, env=env, cwd=repo, stdout=fp, stderr=subprocess.STDOUT, check=True)
    fp.write('\nResolved Hydra config:\n'); fp.flush()
    subprocess.run(base + overrides + ['--cfg', 'job', '--resolve'], env=env, cwd=repo,
                   stdout=fp, stderr=subprocess.STDOUT, check=True)
start = datetime.datetime.now(datetime.timezone.utc)
with (output / 'logs/train.log').open('w') as fp:
    process = subprocess.Popen(base + overrides, env=env, cwd=repo, stdout=fp, stderr=subprocess.STDOUT)
    (output / 'reports/process.json').write_text(json.dumps({'pid':process.pid, 'output':str(output)}))
    result = process.wait()
metrics_path = output / 'reports/metrics.jsonl'
rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line] if metrics_path.exists() else []
report = {'exit_code':result, 'elapsed_seconds':(datetime.datetime.now(datetime.timezone.utc)-start).total_seconds(),
          'training_metrics':rows, 'output':str(output)}
(output / 'reports/metrics.json').write_text(json.dumps(report, indent=2))
(output / 'reports/summary.md').write_text('# Production smoke test\n\n'
    + f'Exit code: {result}\n\nStudent: MixSFT. Teachers: Math, Code, IF (sequential).\n\n'
    + '4 GPU replicated DP, shared CPU Adam, global batch 4, requested optimizer steps 1.\n'
    + 'Response limit 512; KV cache 2048 MiB. Checkpoint saving disabled for this smoke test.\n'
    + f'\nMetrics records: {len(rows)}. See train.log for stage evidence.\n')
print(json.dumps(report), flush=True)
sys.exit(result)
