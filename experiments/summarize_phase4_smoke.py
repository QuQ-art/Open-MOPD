"""Validate the completed production smoke run and write its reviewable summary."""
import json
import math
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
report_path = root / 'reports/metrics.json'
report = json.loads(report_path.read_text())
rows = report['training_metrics']
metrics = rows[-1]['data'] if rows else {}
checks = {
    'process_exit_zero': report['exit_code'] == 0,
    'one_training_record': len(rows) == 1 and rows[0]['step'] == 1,
    'global_step_one': metrics.get('training/global_step') == 1,
    'shared_adam_all_steps_one': metrics.get('multi_gpu/optimizer_step_min') == 1
        and metrics.get('multi_gpu/optimizer_step_max') == 1,
    'published_version_one': metrics.get('multi_gpu/published_version') == 1,
    'finite_nonzero_grad_norm': math.isfinite(metrics.get('actor/grad_norm', float('nan')))
        and metrics.get('actor/grad_norm', 0) > 0,
    'four_samples': sum(metrics.get(f'mt_opd/domain/{d}/prompt_count', 0) for d in ['math', 'code', 'if']) == 4,
    'all_compute_stages_completed': all(metrics.get('timing_s/' + stage, 0) > 0 for stage in
        ['gen', 'compute_log_prob', 'compute_rm_score', 'compute_mt_rm_1_score',
         'compute_mt_rm_2_score', 'compute_distillation_reward', 'update_actor']),
}
report['acceptance'] = checks
report['passed'] = all(checks.values())
report['attempts'] = 1 + len(list((root / 'logs').glob('train.attempt*.log')))
report['checkpoint_saved'] = False
report_path.write_text(json.dumps(report, indent=2))
gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used',
                                    '--format=csv'], text=True)
(root / 'profiles/gpu_inventory.csv').write_text(gpu_info)
lines = [
    '# 正式模型 1-step Smoke Test', '',
    '**结果：' + ('PASS' if report['passed'] else 'FAIL') + '**', '',
    'Student 为正式 MixSFT；Teacher 为正式 Math、Code、IF，均为约 3.34B SmolLM3。',
    '数据直接读取 `/home/xxf/Distill/models/data/rl_prompt_mix/train.parquet`。',
    '通过 r2.sh → main_ppo → Ray trainer/worker 完整启动，未替换模型、RPC 或计算方法。', '',
    '- 4 GPU Replicated DP，TP=1，global batch=4，micro batch/GPU=1，PPO epoch=1。',
    '- Shared CPU Adam；FSDP NO_SHARD 自动同步梯度；三个 Teacher 顺序评分；overlap 关闭。',
    '- max_response_length=512；max_prompt_length=1024；KV Cache=2048 MiB；eager rollout。',
    '- 关闭训练前验证、验证周期、checkpoint 保存和额外 profiler，以检查单步计算闭环。',
    '- GPU 实际容量见 profiles/gpu_inventory.csv：两张约 24 GiB、两张约 48 GiB。', '',
    '## 验收', '',
]
lines.extend(f'- {key}: {value}' for key, value in checks.items())
lines += ['', '## 实测指标', '', '| 指标 | 值 |', '|---|---|']
keys = ['actor/grad_norm', 'actor/pg_loss', 'multi_gpu/optimizer_step_min',
        'multi_gpu/optimizer_step_max', 'multi_gpu/published_version', 'response_length/mean',
        'response_length/max', 'timing_s/gen', 'timing_s/compute_rm_score',
        'timing_s/compute_mt_rm_1_score', 'timing_s/compute_mt_rm_2_score',
        'timing_s/update_actor', 'timing_s/step', 'perf/max_memory_allocated_gb',
        'perf/max_memory_reserved_gb', 'perf/cpu_memory_used_gb']
lines.extend(f'| {key} | {metrics.get(key)} |' for key in keys)
lines += ['', f"本次成功进程用时（含数据准备、模型初始化）：{report['elapsed_seconds']:.2f} 秒。",
          '', '## 修复与历史尝试', '',
          '1. Ray 共享内存命名空间传递补充 os 导入。',
          '2. 内存诊断兼容 PyTorch 缺失 active_bytes.current，显示 nan，不影响训练。',
          '3. 移除误加到 Student log-prob 方法的 Teacher 专用共享权重检查。',
          '4. FileLogger 支持 Tensor/NumPy 指标 JSON 序列化并及时 flush。',
          '', '此前失败尝试分别保存在 logs/train.attempt*.log 和 reports/attempt* 文件中。',
          '第四次尝试已完成计算更新，但指标落盘失败；最后一次正常退出并通过上述验收。',
          '', '## 文件', '',
          '- logs/train.log：最终完整训练日志；logs/ray：Ray 各次会话日志。',
          '- reports/config.txt：环境覆盖、完整命令、解析后的 Hydra 配置。',
          '- reports/metrics.json：完整训练指标、退出状态与验收断言；metrics.jsonl：原始指标。',
          '- checkpoints/：保留目录，本次未保存 checkpoint，更新后的参数仅用于本次进程。',
          '- profiles/gpu_inventory.csv：GPU 清单；本次未启用 Nsight/PyTorch profiler。',
          '', '本次是正式模型训练链路的 smoke 验收，不代表训练质量或长时间稳定性评估。', '']
(root / 'reports/summary.md').write_text('\n'.join(lines))
print(json.dumps({'passed':report['passed'], 'checks':checks, 'summary':str(root / 'reports/summary.md')}, indent=2))
if not report['passed']:
    raise SystemExit(1)
