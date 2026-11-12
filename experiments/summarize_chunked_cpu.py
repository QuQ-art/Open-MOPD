"""Compare bounded-workspace CPU reduce with recorded full-buffer CPU reduce."""
import json
from pathlib import Path
import re
import subprocess
import sys

root = Path(sys.argv[1])
old_cpu = Path('/mnt/sda1/zst/RONDO/experiments/phase4_cpu_reduce_20261111_run1')
old_gpu = Path('/mnt/sda1/zst/RONDO/experiments/phase4_continuous_20261111_114025')
subprocess.run([sys.executable, str(Path(__file__).with_name('summarize_phase4_continuous.py')), str(root),
                'train4096_shared', 'resume4096_shared', str(old_gpu)], check=True)
report = json.loads((root / 'reports/metrics.json').read_text())
baseline = json.loads((old_cpu / 'reports/metrics.json').read_text())
rows = report['train']['metrics'] + report['resume']['metrics']
old_rows = baseline['train']['metrics'] + baseline['resume']['metrics']
comparison = []
for old, new in zip(old_rows, rows, strict=True):
    assert old['step'] == new['step']
    comparison.append({'step':new['step'], **{key:{'full_buffer':old['data'][key], 'chunked':new['data'][key]}
        for key in ['timing_s/update_actor', 'timing_s/step', 'response_length/mean', 'perf/cpu_memory_used_gb']}})
averages = {}
for key in ['timing_s/update_actor', 'timing_s/step', 'response_length/mean', 'perf/cpu_memory_used_gb']:
    old = sum(row[key]['full_buffer'] for row in comparison) / len(rows)
    new = sum(row[key]['chunked'] for row in comparison) / len(rows)
    averages[key] = {'full_buffer':old, 'chunked':new, 'change_pct':100 * (new / old - 1)}
shm = {}
allocations = {}
for label in ['train4096_shared', 'resume4096_shared']:
    samples = [json.loads(line) for line in (root / f'profiles/{label}_gpu.jsonl').read_text().splitlines()]
    values = [s['shm_used_bytes'] for s in samples if 'shm_used_bytes' in s]
    shm[label] = {'peak_bytes':max(values), 'initial_bytes':values[0], 'peak_minus_initial_bytes':max(values)-values[0],
                  'last_bytes':values[-1], 'sample_interval_s':2}
    text = (root / f'logs/{label}.log').read_text()
    matches = re.findall(r'CPU_REDUCE_WORKSPACE chunk_bytes_per_rank=(\d+) workspace_bytes=(\d+) reduced_gradient_bytes=(\d+) chunks=(\d+)', text)
    assert matches, 'missing allocation evidence'
    chunk, workspace, reduced, chunks = map(int, matches[0])
    allocations[label] = {'chunk_bytes_per_rank':chunk, 'workspace_bytes':workspace,
                         'reduced_gradient_bytes':reduced, 'chunks':chunks,
                         'new_gradient_bytes':workspace+reduced, 'old_gradient_bytes_calculated':reduced*5}
    report['checks'][label + '_bounded_workspace'] = chunk == 256 * 1024**2 and workspace == 4 * chunk
for name, marker in [('chunk_boundaries', 'GPU_SHARED_CPU_PASS'), ('mopd', 'GPU_ROLLOUT_PASS')]:
    report['checks'][name] = marker in (root / f'logs/{name}.log').read_text()
report.update(cpu_reduce_baseline=str(old_cpu), cpu_reduce_comparison=comparison,
              cpu_reduce_averages=averages, shm_measurements=shm, gradient_allocations=allocations)
report['passed'] = all(report['checks'].values())
(root / 'reports/metrics.json').write_text(json.dumps(report, indent=2))
lines = ['# Chunked CPU reduce 验收', '', f"结果：{'PASS' if report['passed'] else 'FAIL'}", '',
    '256 MiB/rank 固定 workspace，阻塞 D2H；全部 chunk 汇总后计算全局 norm、裁剪并执行一次 AdamW。',
    '正式模型与历史配置一致：MixSFT + Math/Code/IF、4 GPU、TP=1、global batch=4、response=4096、KV=2048 MiB。',
    '连续 3 步保存后，新进程恢复执行第 4、5 步。GPU 空闲检查和外部进程监控启用。', '',
    '## 分块正确性', '',
    '28-byte workspace 小模型覆盖参数内切分、跨参数打包、尾块，与 GPU all-reduce 逐块对照。',
    '检查全局 norm、裁剪、AdamW 状态、四卡参数一致、NaN/末块 Inf 整步跳过、checkpoint 恢复后继续更新。',
    '256 MiB/rank MOPD 测试验证本地梯度、workspace 内容、CPU 均值、裁剪和真实 vLLM 权重发布。', '',
    '## CPU 梯度内存', '', json.dumps(allocations, indent=2), '',
    '旧梯度占用由相同参数量与旧实现的四份 rank_slots + 一份 reduced 计算；不是旧运行的 /dev/shm 实测。',
    '本轮 /dev/shm 总占用实测（包含模型、教师、优化器等共享状态，2 秒采样）：',
    'initial_bytes 是启动前已有占用；peak_minus_initial_bytes 是相对增量，不能视作单独的梯度内存。', '', json.dumps(shm, indent=2), '',
    '## 性能对比', '', f'完整缓冲区 CPU reduce 基线：{old_cpu}', '',
    '| Step | 原 update_actor 秒 | Chunked update_actor 秒 | 原整步秒 | Chunked整步秒 |', '|---|---|---|---|---|']
for row in comparison:
    lines.append('| ' + ' | '.join(str(v) for v in [row['step'], row['timing_s/update_actor']['full_buffer'],
        row['timing_s/update_actor']['chunked'], row['timing_s/step']['full_buffer'], row['timing_s/step']['chunked']]) + ' |')
lines += ['', '五步平均：', '', json.dumps(averages, indent=2), '',
          '这是历史运行对比，非交错重复基准；生成长度与系统负载可能不同，不能将全部耗时变化归因于分块同步。',
          '整步耗时不包含 checkpoint 保存。完整配置差异、检查结果及 checkpoint 信息见 metrics.json。', '',
          '## 检查结果', '']
lines += [f'- {key}: {value}' for key,value in report['checks'].items()]
(root / 'reports/summary.md').write_text('\n'.join(lines))
print(json.dumps({'passed':report['passed'], 'averages':averages, 'shm':shm, 'allocations':allocations}, indent=2))
raise SystemExit(0 if report['passed'] else 1)
