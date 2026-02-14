# PAPER_MEMORY_MAPPING

## Scope
- Branch: `v3-online-memory`
- Target path: `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v3_online.py`
- Target mode: `train_mode: l2_memory_ablation` (`memory_off` vs `memory_on`)
- Reference file: `/home/jerry/.openclaw/media/inbound/dee63f08-7672-497a-8d7c-3840e9166c60.pdf`

## Paper -> Implementation Mapping
1. Gate-in memory write
- Paper intent: write only useful experiences (progress/success).
- Implementation: in `train_and_eval_online_v3_ff`, writes are accepted only when:
  - `progress_delta > memory_progress_eps`, or
  - current step is terminal success (`done && info["success"]`).
- Additional successful-segment handling: on successful episode end, last `memory_success_segment_len` step candidates are written with success bonus.

2. Retrieval score = similarity * quality_score
- Paper intent: retrieval should combine state match and sample usefulness.
- Implementation:
  - Similarity: `sim = 1 / (1 + euclidean_distance)`.
  - Quality score: positive-progress based quality with floor and success bonus.
  - Retrieval score: `score = sim * quality`.
  - Memory action retrieval: top-`k` by `score`, weighted average by normalized scores.

3. Gate-out eviction when over capacity
- Paper intent: discard weak memories, not oldest by default.
- Implementation:
  - Memory bank is a scored list (`MemorySample`).
  - On insertion when full, evict the lowest-quality sample first.
  - Track `memory_eviction_count` diagnostics.

4. Diagnostics
- `memory_write_accept_rate`: accepted writes / write attempts.
- `memory_retrieval_score_stats`: count, mean, std, min, max, p50, p90 over retrieval calls.
- `memory_eviction_count`: number of quality-based evictions.

## Assumptions
- The provided PDF appears image-scanned in this environment (no local OCR/pdf text tool available), so alignment is implemented against the requested mechanism constraints and practical RL memory design patterns.
- `progress_delta` uses environment-reported `info["distance"]` between consecutive steps.
- Quality score is non-negative and composed as:
  - base = `max(memory_quality_min, memory_quality_progress_scale * max(progress_delta, 0))`
  - plus `memory_quality_success_bonus` on success writes.
- Successful-segment writing duplicates are allowed intentionally to amplify successful local trajectories.

## Config Knobs (Added/Used)
- `memory_progress_eps` (default `1e-4`)
- `memory_quality_progress_scale` (default `1.0`)
- `memory_quality_success_bonus` (default `0.5`)
- `memory_quality_min` (default `0.05`)
- `memory_success_segment_len` (default `8`)
- Existing: `memory_bucket_quota`, `memory_k`
