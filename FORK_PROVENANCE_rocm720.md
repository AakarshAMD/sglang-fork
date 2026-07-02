# ROCm7.2 instrumented sglang fork (branch rocm720-sglang-v0.5.14)
Base: sgl-project/sglang @ v0.5.14 (contains PR #25404 DP disagg-decode idle-batch
all_gather deadlock fix: scheduler_components/dp_attn.py forward_mode.is_idle()).
Overlay (ADDITIVE, env-gated SGLANG_ROCTX; default OFF):
  observability/sglang_roctx.py      (new)  sdk-roctx mark/range helpers
  observability/req_time_stats.py    (+roctx) _roctx() emit at every ReqTimeStats stamp
                                             + roctx_id + set_prefill_kv_transfer_start_time
  disaggregation/prefill.py          (+4)   stamp prefill_kv_transfer_start on first KV send
  managers/schedule_batch.py         (+10)  tag time_stats.roctx_id = bootstrap_room
  disaggregation/mori/conn.py        (+81)  mori.map room<->uid rid attribution marks
Baked into image rocmshared/pytorch-private:sglang_mori_roctx_cq_bytes_rocm720_20260702.
