"""Summarize continuous production training and inspect saved optimizer metadata."""
import json
import math
from pathlib import Path
import sys
import torch

root = Path(sys.argv[1])
train_label, resume_label = sys.argv[2:4]
train = json.loads((root / f'reports/{train_label}_metrics.json').read_text())
resume = json.loads((root / f'reports/{resume_label}_metrics.json').read_text())
rows = train['metrics'] + resume['metrics']
checks = {
    'both_processes_exit_zero': train['exit_code'] == resume['exit_code'] == 0,
    'continuous_three_steps': [r['step'] for r in train['metrics']] == [1,2,3],
    'new_process_resumed_steps_four_five': [r['step'] for r in resume['metrics']] == [4,5],
    'optimizer_steps_and_publication_versions': all(
        all(r['data'].get(key) == r['step'] for key in ['multi_gpu/optimizer_step_min',
            'multi_gpu/optimizer_step_max','multi_gpu/published_version']) for r in rows),
    'finite_gradients': all(math.isfinite(r['data'].get('actor/grad_norm',float('nan'))) for r in rows),
    'no_external_gpu_interference': not train['external_gpu_interference'] and not resume['external_gpu_interference'],
}
for phase, run in [('train', train), ('resume', resume)]:
    values = [r['data']['perf/current_memory_allocated_gb'] for r in run['metrics']]
    checks[phase + '_live_gpu_allocated_range_below_256mib'] = bool(values) and max(values) - min(values) < 0.25
for label in [train_label, resume_label]:
    gate = [json.loads(line) for line in (root / f'profiles/{label}_idle_gate.jsonl').read_text().splitlines()]
    checks[label + '_idle_gate_passed'] = gate[-1]['idle_seconds'] >= 60 and not gate[-1]['apps']
checkpoint_root = root / 'checkpoints' / train['options']['checkpoint_label']
checkpoints = []
for step in [3,5]:
    actor = checkpoint_root / f'global_step_{step}/actor'
    files = sorted(actor.glob('optim_world_size_4_rank_*.pt'))
    assert len(files) == 1, files
    state = torch.load(files[0], mmap=True, map_location='cpu', weights_only=False)
    steps = [float(s['step']) for s in state['state'].values()]
    version = state.get('published_version')
    checkpoints.append({'step':step, 'optimizer_files':len(files), 'optimizer_bytes':files[0].stat().st_size,
        'state_step_min':min(steps), 'state_step_max':max(steps), 'published_version':version,
        'model_files':len(list(actor.glob('model_world_size_4_rank_*.pt'))),
        'extra_state_files':len(list(actor.glob('extra_state_world_size_4_rank_*.pt'))),
        'path':str(actor)})
    checks[f'checkpoint_{step}_state_complete'] = min(steps) == max(steps) == version == step and checkpoints[-1]['model_files'] == checkpoints[-1]['extra_state_files'] == 4
    del state
peaks = {}
for label in [train_label, resume_label]:
    samples = [json.loads(l) for l in (root / f'profiles/{label}_gpu.jsonl').read_text().splitlines() if l]
    peaks[label] = {str(i):max((g[1] for s in samples for g in s.get('gpus',[]) if g[0]==i), default=0) for i in range(4)}
step_metrics = []
for row in rows:
    m = row['data']
    step_metrics.append({'step':row['step'], **{k:m.get(k) for k in ['actor/grad_norm',
        'multi_gpu/published_version','response_length/mean','response_length/max','timing_s/step',
        'perf/current_memory_allocated_gb','perf/current_memory_reserved_gb','perf/max_memory_allocated_gb',
        'perf/max_memory_reserved_gb','perf/cpu_memory_used_gb']}})
report = {'passed':all(checks.values()), 'checks':checks, 'step_metrics':step_metrics,
    'gpu_sampled_peak_mib':peaks, 'checkpoints':checkpoints, 'train':train,'resume':resume}
(root / 'reports/metrics.json').write_text(json.dumps(report,indent=2))
lines = ['# 4096 response 连续训练与 checkpoint 恢复验收','',f"结果：{'PASS' if report['passed'] else 'FAIL'}",'',
    '正式 MixSFT + Math/Code/IF；rl_prompt_mix/train.parquet；4 GPU、TP=1、global batch=4。',
    '连续训练 3 步后保存，退出并启动新进程恢复，再训练第 4、5 步。',
    '最终 response_length=4096，全程未缩短。Teacher 顺序评分，所有 overlap 关闭。','',
    '## 显式记录的显存调整','',
    '- KV Cache：4096 → 2048 MiB。',
    '- entropy：按 256 token 分块；关闭该路径 Torch compile；数值与梯度对照测试通过。',
    '- vLLM gpu_memory_utilization：0.5 → 0.30；显式 KV 字节数仍决定 KV 大小。',
    '- vLLM 初始化前清理空闲 PyTorch 缓存。',
    '- 启用每 rank Student/vLLM 权重共享；保持 FSDP NO_SHARD、Shared CPU Adam。',
    '- 每次启动前要求全部四卡无计算进程、低显存/低利用率持续 60 秒。','',
    '## 验收断言','']
lines += [f'- {k}: {v}' for k,v in checks.items()]
lines += ['', '## 每步指标','', '| Step | 发布版本 | Grad norm | Response max | Step 秒 | 步末 allocated GiB | 步末 reserved GiB |',
    '|---|---|---|---|---|---|---|']
for r in step_metrics:
    lines.append('| ' + ' | '.join(str(r[k]) for k in ['step','multi_gpu/published_version','actor/grad_norm',
        'response_length/max','timing_s/step','perf/current_memory_allocated_gb','perf/current_memory_reserved_gb']) + ' |')
lines += ['', 'PyTorch 指标是各 rank 的汇总值（trainer 默认平均），不是四卡中的最大值。',
    '整卡显存采用每 2 秒 nvidia-smi 采样，可能漏掉短暂尖峰：', '', json.dumps(peaks,indent=2), '',
    '仅 3 步连续训练 + 2 步恢复训练，不将此结果外推为长期无泄漏。', '', '## Checkpoint','']
lines += [f"- Step {c['step']}: `{c['path']}`；4 个模型文件、4 个 extra 文件、1 个共享 optimizer 文件；保存的版本和 Adam step 均为 {c['step']}。" for c in checkpoints]
lines += ['', '共享 Adam 的 master/moments/student 原位恢复到新建具名映射；publisher 版本从 checkpoint 恢复。',
    'optimizer 仅 owner 写一份，读取使用 mmap，避免四份完整 optimizer CPU 副本。','', '## 历史尝试','',
    '- train4096：Student entropy softmax OOM，额外申请约 1.96 GiB。',
    '- train4096_kv2048_chunk256：vLLM 启动空闲内存门槛未通过；未完成训练。',
    '- train4096_memory：模型初始化被外部进程（占 GPU 0 约 18.41 GiB）干扰，不用于判断配置是否可行。',
    '- train4096_idle：启动空闲检查通过，无外部干扰时 backward 超出 PyTorch 配额；未完成 step。',
    '  退出期间监控曾将已脱离父进程的本次 Ray worker 误认为外部进程，已改为保留本次 worker PID 记录。',
    '- 各次配置、日志与 GPU 采样均保留在 reports/、logs/、profiles/。', '']
(root / 'reports/summary.md').write_text('\n'.join(lines))
print(json.dumps({'passed':report['passed'], 'checks':checks,'gpu_sampled_peak_mib':peaks},indent=2))
raise SystemExit(0 if report['passed'] else 1)
