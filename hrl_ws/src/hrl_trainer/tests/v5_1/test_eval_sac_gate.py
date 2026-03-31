from pathlib import Path

from hrl_trainer.v5_1.eval_sac_gate import evaluate
from hrl_trainer.v5_1.train_loop_sac import run_train


def test_eval_gate_outputs_summary_and_gate_file(tmp_path: Path):
    train_root = tmp_path / "train"
    ckpt = run_train(episodes=3, seed=11, artifact_root=train_root)
    payload, code = evaluate(
        ckpt,
        episodes=5,
        seed=11,
        policy_mode="sac",
        enforce_gates=True,
        output_dir=tmp_path / "e2e",
    )
    assert "summary" in payload
    assert "gate_result" in payload
    assert (tmp_path / "e2e" / "gate_result.json").exists()
    assert code in (0, 2)
