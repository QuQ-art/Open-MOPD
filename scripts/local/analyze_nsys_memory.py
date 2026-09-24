#!/usr/bin/env python3
"""Summarize exposed memory-transfer time from an Open-MOPD Nsight SQLite export."""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path


Interval = tuple[int, int]


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def duration_ns(intervals: list[Interval]) -> int:
    return sum(end - start for start, end in merge_intervals(intervals))


def intersection_ns(left: list[Interval], right: list[Interval]) -> int:
    left = merge_intervals(left)
    right = merge_intervals(right)
    total = 0
    i = j = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def clip(interval: Interval, bounds: Interval) -> Interval | None:
    clipped = max(interval[0], bounds[0]), min(interval[1], bounds[1])
    return clipped if clipped[1] > clipped[0] else None


def ms(value_ns: int | float) -> float:
    return value_ns / 1_000_000


def pct(value_ns: int | float, total_ns: int) -> float:
    return value_ns * 100 / total_ns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path, help="Nsight Systems SQLite export for the GPU worker")
    args = parser.parse_args()

    connection = sqlite3.connect(args.sqlite)
    connection.row_factory = sqlite3.Row

    nvtx = []
    for row in connection.execute(
        """
        SELECT e.start, e.end, COALESCE(e.text, s.value) AS name
        FROM NVTX_EVENTS e
        LEFT JOIN StringIds s ON e.textId = s.id
        WHERE e.end IS NOT NULL AND COALESCE(e.text, s.value) LIKE 'openmopd::%'
        ORDER BY e.start
        """
    ):
        nvtx.append((row["start"], row["end"], row["name"]))

    steps = sorted(
        [(start, end, name) for start, end, name in nvtx if name.startswith("openmopd::step::")]
    )
    if not steps:
        raise SystemExit("expected at least one openmopd step range, found none")
    capture = steps[0][0], steps[-1][1]
    capture_ns = capture[1] - capture[0]

    copy_kinds = {
        row["id"]: row["label"] for row in connection.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER")
    }
    copies = []
    for row in connection.execute(
        "SELECT start, end, bytes, copyKind, correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"
    ):
        clipped = clip((row["start"], row["end"]), capture)
        if clipped:
            copies.append(
                (
                    *clipped,
                    row["bytes"],
                    copy_kinds.get(row["copyKind"], str(row["copyKind"])),
                    row["correlationId"],
                )
            )

    kernels = []
    for row in connection.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"):
        clipped = clip((row["start"], row["end"]), capture)
        if clipped:
            kernels.append(clipped)

    runtime = []
    for row in connection.execute(
        """
        SELECT r.start, r.end, s.value AS name, r.correlationId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        ORDER BY r.start
        """
    ):
        clipped = clip((row["start"], row["end"]), capture)
        if clipped:
            runtime.append((*clipped, row["name"], row["correlationId"]))

    memcpy_intervals = [(start, end) for start, end, _, _, _ in copies]
    memcpy_union_ns = duration_ns(memcpy_intervals)
    memcpy_kernel_overlap_ns = intersection_ns(memcpy_intervals, kernels)
    memcpy_exposed_ns = memcpy_union_ns - memcpy_kernel_overlap_ns
    kernel_union_ns = duration_ns(kernels)

    host_alloc = [(start, end) for start, end, name, _ in runtime if name.startswith("cudaHostAlloc")]
    copy_api = [(start, end) for start, end, name, _ in runtime if "Memcpy" in name]
    memory_map_api = [
        (start, end)
        for start, end, name, _ in runtime
        if name.startswith(("cuMemMap", "cuMemUnmap", "cuMemCreate", "cuMemRelease", "cuMemSetAccess"))
    ]
    memory_service = host_alloc + copy_api + memory_map_api + memcpy_intervals
    memory_service_ns = duration_ns(memory_service)

    io_ranges = [
        clipped
        for start, end, name in nvtx
        if "::io::" in name
        if (clipped := clip((start, end), capture))
    ]
    sync_ranges = [
        clipped
        for start, end, name in nvtx
        if "::weight_sync::" in name
        if (clipped := clip((start, end), capture))
    ]
    lifecycle_ranges = [
        clipped
        for start, end, name in nvtx
        if "::memory::" in name
        if (clipped := clip((start, end), capture))
    ]
    semantic_memory_ns = duration_ns(io_ranges + sync_ranges + lifecycle_ranges)

    copy_bytes: dict[str, int] = defaultdict(int)
    copy_sum_ns: dict[str, int] = defaultdict(int)
    copy_count: dict[str, int] = defaultdict(int)
    for start, end, size, kind, _ in copies:
        copy_bytes[kind] += size
        copy_sum_ns[kind] += end - start
        copy_count[kind] += 1

    capture_name = " -> ".join(name.removeprefix("openmopd::") for _, _, name in steps)
    print(f"# Nsight memory analysis: {capture_name}\n")
    print(f"- Capture wall time: **{ms(capture_ns):.3f} ms** across {len(steps)} step(s)")
    print(
        f"- Strict exposed GPU DMA time (copy engine active while no GPU kernel runs): "
        f"**{ms(memcpy_exposed_ns):.3f} ms ({pct(memcpy_exposed_ns, capture_ns):.2f}%)**"
    )
    print(
        f"- GPU DMA busy union: **{ms(memcpy_union_ns):.3f} ms ({pct(memcpy_union_ns, capture_ns):.2f}%)**; "
        f"overlapped with kernels: {ms(memcpy_kernel_overlap_ns):.3f} ms"
    )
    print(
        f"- End-to-end memory-service union (pinned-host allocation + copy API + GPU DMA + map/unmap): "
        f"**{ms(memory_service_ns):.3f} ms ({pct(memory_service_ns, capture_ns):.2f}%)**"
    )
    print(
        f"- Explicit Open-MOPD I/O/weight-sync/sleep-wake phase union: "
        f"**{ms(semantic_memory_ns):.3f} ms ({pct(semantic_memory_ns, capture_ns):.2f}%)**"
    )
    print(
        f"- GPU kernel busy union: **{ms(kernel_union_ns):.3f} ms "
        f"({pct(kernel_union_ns, capture_ns):.2f}%)**"
    )
    print(f"- `cudaHostAlloc` blocking union: **{ms(duration_ns(host_alloc)):.3f} ms**")

    print("\n## Transfers\n")
    print("| Direction | Count | GiB | Summed GPU time (ms) |")
    print("|---|---:|---:|---:|")
    for kind in sorted(copy_bytes):
        print(
            f"| {kind} | {copy_count[kind]} | {copy_bytes[kind] / 2**30:.3f} | "
            f"{ms(copy_sum_ns[kind]):.3f} |"
        )

    print("\n## Per-step GPU activity\n")
    print("| Step | Wall (ms) | H2D GiB | D2H GiB | D2D GiB | DMA union (ms) | DMA/kernel overlap (ms) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for step_start, step_end, step_name in steps:
        bounds = step_start, step_end
        step_copies = []
        for copy_start, copy_end, size, kind, correlation_id in copies:
            clipped = clip((copy_start, copy_end), bounds)
            if clipped:
                fraction = (clipped[1] - clipped[0]) / (copy_end - copy_start)
                step_copies.append((*clipped, size * fraction, kind, correlation_id))
        step_kernels = [clipped for event in kernels if (clipped := clip(event, bounds))]
        step_bytes: dict[str, float] = defaultdict(float)
        for _, _, size, kind, _ in step_copies:
            step_bytes[kind] += size
        step_dma = [(start, end) for start, end, _, _, _ in step_copies]
        print(
            f"| `{step_name.removeprefix('openmopd::')}` | {ms(step_end - step_start):.3f} | "
            f"{step_bytes['Host-to-Device'] / 2**30:.3f} | "
            f"{step_bytes['Device-to-Host'] / 2**30:.3f} | "
            f"{step_bytes['Device-to-Device'] / 2**30:.3f} | "
            f"{ms(duration_ns(step_dma)):.3f} | {ms(intersection_ns(step_dma, step_kernels)):.3f} |"
        )

    if any(name == "openmopd::io::h2d::teacher_prefetch" for _, _, name in nvtx):
        print("\n## Teacher parameter prefetch\n")
        print("GPU copy intervals are linked to the prefetch API range by CUDA correlation ID.\n")
        print("| Step | H2D MiB | GPU copy (ms) | Within student phase (ms) | With student kernels (ms) |")
        print("|---|---:|---:|---:|---:|")
        for step_start, step_end, step_name in steps:
            prefetch_ranges = [
                (start, end)
                for start, end, name in nvtx
                if name == "openmopd::io::h2d::teacher_prefetch"
                and step_start <= start <= end <= step_end
            ]
            copy_ids = {
                correlation_id
                for start, end, api_name, correlation_id in runtime
                if correlation_id is not None
                and "Memcpy" in api_name
                and any(a <= start <= end <= b for a, b in prefetch_ranges)
            }
            prefetch_copies = [
                copy for copy in copies if copy[3] == "Host-to-Device" and copy[4] in copy_ids
            ]
            prefetch_intervals = [(start, end) for start, end, _, _, _ in prefetch_copies]
            student_ranges = [
                (start, end)
                for start, end, name in nvtx
                if name == "openmopd::compute::student_log_prob"
                and step_start <= start <= end <= step_end
            ]
            student_kernels = [
                clipped
                for kernel in kernels
                for student in student_ranges
                if (clipped := clip(kernel, student))
            ]
            print(
                f"| `{step_name.removeprefix('openmopd::')}` | "
                f"{sum(copy[2] for copy in prefetch_copies) / 2**20:.1f} | "
                f"{ms(duration_ns(prefetch_intervals)):.3f} | "
                f"{ms(intersection_ns(prefetch_intervals, student_ranges)):.3f} | "
                f"{ms(intersection_ns(prefetch_intervals, student_kernels)):.3f} |"
            )

    if len(steps) > 1:
        print("\n## Cross-step optimizer-offload overlap\n")
        print(
            "CUDA memcpy correlation IDs associate asynchronous D2H copies with the "
            "`io::d2h::optimizer_state` API calls that submitted them.\n"
        )
        print(
            "| Transition | Optimizer D2H (GiB) | Optimizer D2H GPU time (ms) | "
            "Inside next rollout (ms) | Concurrent with next rollout kernels (ms) |"
        )
        print("|---|---:|---:|---:|---:|")
        for current_step, next_step in zip(steps, steps[1:]):
            current_start, current_end, current_name = current_step
            next_start, next_end, next_name = next_step
            optimizer_ranges = [
                (start, end)
                for start, end, name in nvtx
                if name == "openmopd::io::d2h::optimizer_state"
                and start >= current_start
                and end <= current_end
            ]
            optimizer_copy_ids = {
                correlation_id
                for start, end, api_name, correlation_id in runtime
                if correlation_id is not None
                and "Memcpy" in api_name
                and any(start >= range_start and end <= range_end for range_start, range_end in optimizer_ranges)
            }
            optimizer_copies = [
                copy
                for copy in copies
                if copy[3] == "Device-to-Host" and copy[4] in optimizer_copy_ids
            ]
            optimizer_copy_intervals = [(start, end) for start, end, _, _, _ in optimizer_copies]
            next_rollouts = [
                (start, end)
                for start, end, name in nvtx
                if name == "openmopd::compute::student_rollout"
                and start >= next_start
                and end <= next_end
            ]
            next_rollout_kernels = [
                clipped
                for kernel in kernels
                for rollout in next_rollouts
                if (clipped := clip(kernel, rollout))
            ]
            print(
                f"| `{current_name.rsplit('::', 1)[-1]} -> {next_name.rsplit('::', 1)[-1]}` | "
                f"{sum(copy[2] for copy in optimizer_copies) / 2**30:.3f} | "
                f"{ms(duration_ns(optimizer_copy_intervals)):.3f} | "
                f"{ms(intersection_ns(optimizer_copy_intervals, next_rollouts)):.3f} | "
                f"{ms(intersection_ns(optimizer_copy_intervals, next_rollout_kernels)):.3f} |"
            )

    print("\n## Instrumented phases\n")
    print("| Step | Phase | Wall (ms) | H2D GiB | D2H GiB | D2D GiB | GPU copy (ms) | Host alloc (ms) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for range_start, range_end, name in nvtx:
        if name.startswith("openmopd::step::"):
            continue
        phase_step = next(
            (
                step_name.rsplit("::", 1)[-1]
                for step_start, step_end, step_name in steps
                if range_start >= step_start and range_end <= step_end
            ),
            None,
        )
        if phase_step is None:
            continue
        phase_copies = [copy for copy in copies if copy[0] >= range_start and copy[1] <= range_end]
        phase_runtime = [event for event in runtime if event[0] >= range_start and event[1] <= range_end]
        bytes_by_kind: dict[str, int] = defaultdict(int)
        for _, _, size, kind, _ in phase_copies:
            bytes_by_kind[kind] += size
        phase_copy_ns = duration_ns([(start, end) for start, end, _, _, _ in phase_copies])
        phase_host_alloc_ns = duration_ns(
            [
                (start, end)
                for start, end, api_name, _ in phase_runtime
                if api_name.startswith("cudaHostAlloc")
            ]
        )
        print(
            f"| {phase_step} | `{name.removeprefix('openmopd::')}` | {ms(range_end - range_start):.3f} | "
            f"{bytes_by_kind['Host-to-Device'] / 2**30:.3f} | "
            f"{bytes_by_kind['Device-to-Host'] / 2**30:.3f} | "
            f"{bytes_by_kind['Device-to-Device'] / 2**30:.3f} | "
            f"{ms(phase_copy_ns):.3f} | {ms(phase_host_alloc_ns):.3f} |"
        )


if __name__ == "__main__":
    main()
