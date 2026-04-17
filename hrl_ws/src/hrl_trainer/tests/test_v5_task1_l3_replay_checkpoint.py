import json
import tempfile
import unittest
from pathlib import Path

from hrl_trainer.v5.task1_train import run_task1_training


class TestTask1L3ReplayCheckpoint(unittest.TestCase):
    def test_training_emits_replay_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "task1_ckpt.json"
            rows = run_task1_training(episodes=2, reward_mode="heuristic", checkpoint_path=str(ckpt))

            self.assertEqual(len(rows), 2)
            self.assertTrue(ckpt.exists())
            self.assertIn("replay", rows[0])
            self.assertGreater(len(rows[0]["replay"]), 0)
            self.assertIn("l2_gain_after_update", rows[-1])

            payload = json.loads(ckpt.read_text(encoding="utf-8"))
            self.assertIn("l2", payload)
            self.assertIn("gain", payload["l2"])
            self.assertEqual(len(payload["episodes"]), 2)


if __name__ == "__main__":
    unittest.main()
