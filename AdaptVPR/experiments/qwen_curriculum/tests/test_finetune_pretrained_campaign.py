from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.qwen_curriculum.finetune_pretrained import training_command


class FineTuneCommandTests(unittest.TestCase):
    def config(self):
        return {"real_data": "/real", "cities": ["City"], "accepted_manifest": "/accepted.jsonl",
                "training_dir": "/output", "epochs": 10, "backbone_repo": "/repo",
                "initial_checkpoint": "/original/dino_salad.ckpt"}

    def test_fresh_training_uses_complete_original_checkpoint_and_only_accepted_pool(self):
        cfg = self.config()
        command = training_command(cfg)
        self.assertEqual(command[command.index("--init-checkpoint") + 1], cfg["initial_checkpoint"])
        self.assertEqual(command[command.index("--synthetic-manifest") + 1], cfg["accepted_manifest"])
        self.assertNotIn("--backbone-weights", command)
        self.assertFalse(any("reliability" in flag for flag in command))
        self.assertNotIn("--synthetic-places-only", command)
        self.assertNotIn("--max-batches-per-epoch", command)

    def test_resume_keeps_full_optimizer_checkpoint_and_original_epoch_schedule(self):
        command = training_command(self.config(), resume=True)
        self.assertNotIn("--init-checkpoint", command)
        self.assertEqual(command[command.index("--resume") + 1], "/output/checkpoint.pt")
        self.assertEqual(command[command.index("--epochs") + 1], "10")


if __name__ == "__main__":
    unittest.main()
