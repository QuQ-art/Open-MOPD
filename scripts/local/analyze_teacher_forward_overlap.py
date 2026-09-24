#!/usr/bin/env python3
"""Compare serial/joint scoring using thread-aware CUDA correlation IDs.

Usage: python scripts/local/analyze_teacher_forward_overlap.py output/run_a output/run_b
CPU NVTX duration, GPU busy time, and real DMA/kernel overlap are reported separately.
"""

import argparse
import json
import sqlite3
from pathlib import Path

from analyze_nsys_memory import duration_ns, intersection_ns, ms


def analyze(run: Path) -> list[dict]:
    database = run / "traces" / "worker.sqlite"
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        nvtx = list(connection.execute(
            "SELECT e.start,e.end,e.globalTid,coalesce(e.text,s.value) name FROM NVTX_EVENTS e "
            "LEFT JOIN StringIds s ON e.textId=s.id WHERE e.end IS NOT NULL "
            "AND coalesce(e.text,s.value) LIKE 'openmopd::%' ORDER BY start"
        ))
        apis = list(connection.execute(
            "SELECT a.*,s.value name FROM CUPTI_ACTIVITY_KIND_RUNTIME a "
            "JOIN StringIds s ON a.nameId=s.id ORDER BY a.start"
        ))
        kernels = list(connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"))
        copies = list(connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"))
    finally:
        connection.close()

    def owned(ranges, activities):
        # Time alone misattributes teacher kernels to the overlapping student range.
        correlations = {
            api["correlationId"] for api in apis
            if any(r["globalTid"] == api["globalTid"]
                   and r["start"] <= api["start"] <= api["end"] <= r["end"] for r in ranges)
        }
        return [a for a in activities if a["correlationId"] in correlations]

    def intervals(rows):
        return [(r["start"], r["end"]) for r in rows]

    def busy(rows):
        return ms(duration_ns(intervals(rows)))

    results = []
    for step in (r for r in nvtx if r["name"].startswith("openmopd::step::")):
        ranges = [r for r in nvtx if step["start"] <= r["start"] <= r["end"] <= step["end"]]

        def named(name):
            return [r for r in ranges if r["name"] == "openmopd::" + name]

        students = named("compute::student_log_prob")
        if len(students) != 1:
            raise ValueError(f"expected one student range in {run}/{step['name']}, got {len(students)}")
        student = students[0]
        model = named("compute::teacher_model_forward::Math")
        post = named("compute::teacher_postprocess::Math")
        teacher = model + post if model else named("compute::teacher_forward::Math")
        if not teacher or (model and not post):
            raise ValueError(f"missing Math teacher ranges in {run}/{step['name']}")
        sk = owned(students, kernels)
        tk = owned(teacher, kernels)
        student_apis = [
            a for a in apis if a["globalTid"] == student["globalTid"]
            and student["start"] <= a["start"] <= a["end"] <= student["end"]
        ]
        student_sync = [a for a in student_apis if a["name"].startswith("cudaStreamSynchronize")]
        ht = [r for r in owned(teacher, copies) if r["copyKind"] == 1]
        prefetch = [r for r in owned(named("io::h2d::teacher_prefetch"), copies) if r["copyKind"] == 1]
        all_ht = ht + prefetch
        ht_ns = duration_ns(intervals(all_ht))
        overlap_ns = intersection_ns(intervals(all_ht), intervals(sk))
        prefetch_ns = duration_ns(intervals(prefetch))
        prefetch_overlap_ns = intersection_ns(intervals(prefetch), intervals(sk))
        end = max(r["end"] for r in teacher)
        joint = named("compute::student_teacher_scoring")
        results.append({
            "run": run.name, "step": step["name"].split("::")[-1],
            "step_ms": ms(step["end"] - step["start"]),
            "student_wall_ms": ms(student["end"] - student["start"]),
            "student_kernel_count": len(sk), "student_kernel_ms": busy(sk),
            "student_kernel_span_ms": ms(max(r["end"] for r in sk) - min(r["start"] for r in sk)),
            "student_cuda_api_union_ms": busy(student_apis),
            "student_stream_sync_count": len(student_sync), "student_stream_sync_ms": busy(student_sync),
            "student_small_dtoh_count": sum(
                r["copyKind"] == 2 and r["bytes"] == 4 for r in owned(students, copies)
            ),
            "teacher_kernel_count": len(tk), "teacher_kernel_ms": busy(tk),
            "teacher_htod_ms": busy(ht), "teacher_htod_MiB": sum(r["bytes"] for r in ht) / 2**20,
            "prefetch_htod_ms": busy(prefetch),
            "prefetch_MiB": sum(r["bytes"] for r in prefetch) / 2**20,
            "prefetch_student_kernel_overlap_ms": ms(prefetch_overlap_ns),
            "prefetch_student_kernel_overlap_percent": (
                100 * prefetch_overlap_ns / prefetch_ns if prefetch_ns else None
            ),
            "teacher_htod_student_kernel_overlap_ms": ms(overlap_ns),
            "teacher_htod_student_kernel_overlap_percent": 100 * overlap_ns / ht_ns if ht_ns else None,
            "student_teacher_kernel_overlap_ms": ms(intersection_ns(intervals(sk), intervals(tk))),
            "student_to_teacher_end_ms": ms(end - student["start"]),
            "student_to_teacher_gpu_done_ms": ms(max([end] + [r["end"] for r in sk + tk]) - student["start"]),
            "joint_range_ms": busy(joint) if joint else None,
            "teacher_model_cpu_ms": busy(model) if model else None,
            "teacher_postprocess_ms": busy(post) if post else None,
            "join_cpu_ms": busy(named("compute::teacher_logits_join")) if joint else None,
            "student_streams": sorted({r["streamId"] for r in sk}),
            "teacher_streams": sorted({r["streamId"] for r in tk}),
            "contexts": sorted({r["contextId"] for r in sk + tk}),
        })
    if not results:
        raise ValueError(f"no Open-MOPD step ranges in {database}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([row for run in args.runs for row in analyze(run)], indent=2))
