import unittest
from pathlib import Path

import yaml

from starVLA.dataloader.gr00t_lerobot.mixtures import (
    DATASET_NAMED_MIXTURES,
    ROBOTWIN_CLEAN_TASKS,
)


class RoboTwinContractTest(unittest.TestCase):
    def test_clean_mixture_and_model_shape(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / "examples/Robotwin/train_files/robotwin_clean_method.yaml").read_text())
        tasks = DATASET_NAMED_MIXTURES["robotwin_clean_50"]
        self.assertEqual(len(ROBOTWIN_CLEAN_TASKS), 50)
        self.assertEqual(tasks, [(f"Clean/{name}", 1.0, "robotwin50") for name in ROBOTWIN_CLEAN_TASKS])
        action = config["framework"]["action_model"]
        self.assertEqual((action["action_dim"], action["action_horizon"], action["discrete_action_horizon"]), (14, 50, 10))
        self.assertTrue(action["isolate_branch_tokens_before_shared"])
        self.assertEqual(action["diffusion_model_cfg"]["num_shared_layers"], 3)
        self.assertEqual(config["framework"]["qwenvl"]["num_vl_layers"], 32)
        self.assertEqual(config["datasets"]["vla_data"]["data_mix"], "robotwin_clean_50")
        self.assertEqual(config["trainer"]["max_train_steps"], 60000)


if __name__ == "__main__":
    unittest.main()
