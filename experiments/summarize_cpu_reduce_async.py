"""Summarize paired measurements; overlapping stage times must not be added."""
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
raw = json.loads((root / 'reports/benchmark.json').read_text())
summary = {}
for mode in ('serial', 'double_buffer', 'async'):
    records = [r for r in raw['records'] if r['mode'] == mode and r['round'] >= 0]
    rounds = sorted({r['round'] for r in records})
    values = {}
    for key in ('elapsed_s', 'transport_reduce_s', 'd2h_s', 'd2h_wait_s', 'staging_copy_s', 'cpu_reduce_s', 'barrier_s'):
        samples = [max(r[key] for r in records if r['round'] == i) for i in rounds]
        values[key] = dict(mean=statistics.mean(samples), min=min(samples), max=max(samples), samples=samples)
    summary[mode] = values
improvement = 100 * (1 - summary['async']['transport_reduce_s']['mean'] / summary['serial']['transport_reduce_s']['mean'])
summary['async_transport_improvement_percent'] = improvement
(root / 'reports/metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
lines = ['# CPU reduce 双缓冲与异步 D2H', '',
    '四卡，BF16 参数与梯度，参数量 3,075,098,624（与 MixSFT 相同量级/数量），FP32 reduce；'
    '256 MiB/rank/chunk。使用一个大参数模拟同等总量（46 chunks），相同 GPU 梯度重复使用，不含 rollout、backward、AdamW 或权重发布。', '',
    '每种模式预热一次，测量三轮；每轮轮换模式顺序。每项先取该轮四 rank 最大值，再求三轮平均；'
    'CPU reduce 仅 owner 执行。elapsed 包含全局 norm/clip；transport 不含 norm/clip。', '',
    '| 模式 | D2H+复制+reduce+同步 s | 含 norm/clip s | D2H s | 等待 D2H s | pinned→shared s | owner reduce s |',
    '|---|---:|---:|---:|---:|---:|---:|']
for mode in ('serial', 'double_buffer', 'async'):
    v = summary[mode]
    keys = ('transport_reduce_s', 'elapsed_s', 'd2h_s', 'd2h_wait_s', 'staging_copy_s', 'cpu_reduce_s')
    lines.append('| ' + mode + ' | ' + ' | '.join(f"{v[k]['mean']:.4f}" for k in keys) + ' |')
lines += ['', f'async 对 serial 的 transport 时间改善：{improvement:.2f}%（负数表示变慢）。', '',
    'D2H 指标：serial 使用阻塞 copy 的 CPU 墙钟时间；buffered 模式使用 CUDA events 的传输区间时间。'
    '两种计时口径不同；BF16→FP32 copy 的 blocking/nonblocking 路径也可能改变 dtype 转换位置，不能将全部收益归因于 overlap 或 PCIe 带宽提升。overlap 下各分项不是可直接相加的互斥阶段。', '',
    '固定 workspace：serial 1 GiB shared；双缓冲 2 GiB shared + 每 rank 512 MiB pinned（四卡合计 2 GiB）。'
    '两者均保留一份完整 FP32 reduced gradient。配对 benchmark 为复用同一 optimizer，三模式都保留双缓冲分配；'
    'serial 实际只访问 workspace 0、不使用 pinned staging。', '',
    '正确性：serial/double_buffer/async 分别通过四卡梯度均值、global norm、AdamW moments、四卡参数、'
    'NaN/Inf 整步跳过、checkpoint 原位恢复后继续更新、非默认 stream 延迟梯度与缺失梯度检查。', '',
    'CUDA event 在 staging CPU 读取前 synchronize；ready barrier 在共享 workspace 发布后；'
    'consumed barrier 在 owner reduce 后。下一个 chunk 的 D2H 在 owner reduce 前提交，复用由双槽轮转保护。', '',
    '该结果是正式参数量的 transport microbenchmark，不代表完整训练 update_actor 的相同比例收益。', '',
    '[原始逐 rank 结果](benchmark.json) · [统计](metrics.json) · [进程退出状态](processes.json)', '']
(root / 'reports/summary.md').write_text('\n'.join(lines))
print(json.dumps(summary, indent=2))
