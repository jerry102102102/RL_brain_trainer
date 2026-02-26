# WP0 DoD Checklist (Mapped to `wp0_report.json`)

This file mirrors the non-negotiable WP0 checklist and maps each item to evidence fields/paths produced by `artifacts/wp0/wp0_report.json`.

## 1. Camera contract topic/type/encoding/res/fps
- Status: `sections.camera_contract.status`
- Expected config lock:
  - `config.path`
  - `sections.camera_contract.evidence.per_camera.<camera>.expected.topic`
  - `sections.camera_contract.evidence.per_camera.<camera>.expected.type`
  - `sections.camera_contract.evidence.per_camera.<camera>.expected.encoding`
  - `sections.camera_contract.evidence.per_camera.<camera>.expected.width`
  - `sections.camera_contract.evidence.per_camera.<camera>.expected.height`
  - `sections.camera_contract.evidence.per_camera.<camera>.expected.fps`
- Live observed evidence (if available):
  - `sections.camera_contract.evidence.per_camera.<camera>.observed.type`
  - `sections.camera_contract.evidence.per_camera.<camera>.observed.encoding`
  - `sections.camera_contract.evidence.per_camera.<camera>.observed.width`
  - `sections.camera_contract.evidence.per_camera.<camera>.observed.height`
  - `sections.camera_contract.evidence.per_camera.<camera>.observed.fps`
  - `sections.camera_contract.numeric_evidence.<camera>_latency_p95_ms`

## 2. TF contract + `frames.pdf` path under `artifacts/wp0/`
- Status: `sections.tf_contract.status`
- Required pairs: `sections.tf_contract.evidence.required_pairs`
- `frames.pdf` path: `artifacts.frames_pdf`
- `frames.pdf` existence check: `sections.tf_contract.evidence.frames_pdf_exists`
- Runtime check evidence: `sections.tf_contract.evidence.checks`

## 3. Approx sync (`slop=50ms`, `queue_size=10`, success rate `>95%`)
- Status: `sections.approx_sync.status`
- Slop lock: `sections.approx_sync.numeric_evidence.configured_slop_ms`
- Queue size lock: `sections.approx_sync.numeric_evidence.configured_queue_size`
- Runtime evidence:
  - `sections.approx_sync.numeric_evidence.success_rate`
  - `sections.approx_sync.numeric_evidence.pairs`
  - `sections.approx_sync.numeric_evidence.observed_slop_ms`
  - `sections.approx_sync.numeric_evidence.observed_queue_size`

## 4. Tray jitter + ID switch + missing rate
- Status: `sections.tray_stability.status`
- Jitter threshold / evidence:
  - `sections.tray_stability.numeric_evidence.jitter_std_limit_m`
  - `sections.tray_stability.numeric_evidence.jitter_std_x_m`
  - `sections.tray_stability.numeric_evidence.jitter_std_y_m`
  - `sections.tray_stability.numeric_evidence.jitter_std_z_m`
- ID switch / missing rate evidence:
  - `sections.tray_stability.numeric_evidence.id_switch_rate`
  - `sections.tray_stability.numeric_evidence.id_switch_rate_limit`
  - `sections.tray_stability.numeric_evidence.missing_rate`

## 5. State topic latency P95 `<80ms` (`recv-now - header.stamp`, same clock domain)
- Status: `sections.state_latency.status`
- Numeric evidence:
  - `sections.state_latency.numeric_evidence.p95_ms`
  - `sections.state_latency.numeric_evidence.p95_limit_ms`
  - `sections.state_latency.numeric_evidence.sample_count`
- Same clock-domain statement evidence:
  - `sections.state_latency.evidence.metrics.latency_definition`

## 6. Rosbag record/replay checks + replay image latency P95 `<120ms`
- Status: `sections.rosbag_replay.status`
- Record/replay command evidence:
  - `sections.rosbag_replay.evidence.commands.record`
  - `sections.rosbag_replay.evidence.commands.replay`
- Replay latency evidence:
  - `sections.rosbag_replay.numeric_evidence.replay_image_latency_p95_ms`
  - `sections.rosbag_replay.numeric_evidence.replay_image_latency_p95_limit_ms`
  - `sections.rosbag_replay.evidence.replay_image_topics`

## 7. Overall result + failure reasons / suggested fixes
- Overall PASS/FAIL: `overall.result`
- Section counts: `overall.counts`
- Reasons + fixes: `issues[]`

