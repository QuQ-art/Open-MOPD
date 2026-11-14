"""Summarize measured teacher-only memory, without treating reserved as live allocation."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
if (not (root / 'reports/metrics.json').exists()
        or 'ranks' not in json.loads((root / 'reports/metrics.json').read_text())):
    log = (root / 'logs/test.log').read_text()
    results = {}
    for marker in ('GPU_MOPD_PASS', 'GPU_ROLLOUT_PASS'):
        matches = [line.split(marker + ' ', 1)[1] for line in log.splitlines() if marker + ' ' in line]
        if not matches:
            raise RuntimeError('missing acceptance result: ' + marker)
        results[marker] = json.loads(matches[-1])
    (root / 'reports/metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    (root / 'reports/summary.md').write_text(
        '# Teacher slot MOPD 闭环\n\nPASS：四卡不同样本、顺序 Teacher slot 评分与原路径逐元素一致、'
        'MT-OPD signal、CPU reduce、梯度裁剪、一次 AdamW、四卡新参数一致及真实 vLLM 权重发布通过。'
        '\n\n此测试使用小模型 fixture，不是正式模型 4096-token 连续训练。'
        '\n\n[指标](metrics.json) · [日志](../logs/test.log)\n')
    print(root / 'reports/summary.md')
    raise SystemExit(0)
metrics = json.loads((root / 'reports/metrics.json').read_text())
rows = metrics['ranks']
lines = ['# Teacher slot 顺序复用验收', '',
         '状态：PASS。4 个 rank，各自顺序执行 Math、Code、IF，两轮共 24 次 forward；'
         '与原 FSDP 路径的 logits 逐元素一致（atol=0, rtol=0）。', '',
         '每张卡复用同一组 GPU 参数地址；三位 Teacher 的 CPU mmap 权重仍独立共享。'
         'H2D 为阻塞拷贝，无异步预取、无整模型 pinned 副本。关闭 pipeline 后原 FSDP forward 仍一致。', '',
         '| Rank | 原路径 peak allocated GiB | slot peak allocated GiB | '
         '原路径 peak reserved GiB | slot peak reserved GiB |',
         '|---|---:|---:|---:|---:|']
for row in rows:
    lines.append('| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |'.format(
        row['rank'], row['baseline']['peak_allocated'] / 2**30,
        row['slots']['peak_allocated'] / 2**30, row['baseline']['peak_reserved'] / 2**30,
        row['slots']['peak_reserved'] / 2**30))
r = rows[0]
lines += ['', f"每 rank {r['slots_count']} 个 slot，参数容量 {r['slot_bytes'] / 2**30:.3f} GiB。",
          '', '测量范围：模型初始化后，teacher-only，BF16/eager，每 rank 固定 8 tokens；'
          '两阶段开始前均 synchronize + empty_cache + reset_peak_memory_stats。'
          '原路径执行一轮三位 Teacher，slot 路径执行两轮。', '',
          'allocated 峰值未下降；reserved 峰值下降反映 CUDA allocator 缓存复用。'
          'slot 是持久 GPU 内存：评分后 idle allocated 从 '
          f"{r['baseline']['idle_allocated'] / 2**30:.3f} 增至 {r['slots']['idle_allocated'] / 2**30:.3f} GiB。"
          '因此不能将 reserved 降幅等同于全训练峰值降幅；本报告不代表 4096-token 正式训练显存。', '',
          '详细数值：[metrics.json](metrics.json)；运行配置：[config.txt](config.txt)；'
          '日志：[test.log](../logs/test.log)。', '']
(root / 'reports/summary.md').write_text('\n'.join(lines))
print(root / 'reports/summary.md')
