# Fork provenance - sglang

Single-commit snapshot fork of **sgl-project/sglang** (https://github.com/sgl-project/sglang),
as baked into the Docker image
`lmsysorg/sglang-rocm:v0.5.12.post1-rocm700-mi30x-20260602` (tree at /sgl-workspace/sglang).

- Upstream repo: https://github.com/sgl-project/sglang
- Upstream commit baked into image: 22043b917bd2e37a427ad334d78b59cd7e3a8079
  (branch main, git describe: gateway-v0.3.1-4720-g22043b917, commit date 2026-06-02)
- Source image: lmsysorg/sglang-rocm:v0.5.12.post1-rocm700-mi30x-20260602

On top of upstream we apply AMD-internal ROCTX request-stats + mori.map attribution
instrumentation. Modified files (5, all under python/sglang/srt/):

- observability/sglang_roctx.py   - ROCTX marker helpers / reqstats emission (new file).
- observability/req_time_stats.py - per-request time-stats accounting.
- disaggregation/prefill.py        - prefill-side ROCTX reqstats wiring.
- managers/schedule_batch.py       - batch-level ROCTX instrumentation hooks.
- disaggregation/mori/conn.py      - mori.map rid->uid attribution emit (_mori_emit_map / mori.map).

Instrumentation gated by SGLANG_ROCTX (default OFF) -> base behavior unless enabled.
Upstream git history is intentionally not included; this is a single clean source snapshot.

Build artifacts excluded via .gitignore: sgl-model-gateway/bindings/python/target/
(Rust compiled output, ~3.3 GB), plus __pycache__/ and *.pyc.
