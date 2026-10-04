from pathlib import Path

import cv2
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.tools import read_mode_config


class ModelClient:
    def __init__(self, policy_ckpt_path, host="127.0.0.1", port=5694,
                 unnorm_key="new_embodiment", action_mode="abs"):
        if action_mode != "abs":
            raise ValueError("This RoboTwin release supports absolute joint actions only.")
        self.client = WebsocketClientPolicy(host, int(port))
        config, statistics = read_mode_config(Path(policy_ckpt_path))
        self.action_chunk_size = int(config["framework"]["action_model"]["action_horizon"])
        self.action_stats = statistics[unnorm_key]["action"]
        self.reset("")

    def reset(self, task_description):
        self.task_description = task_description
        self.raw_actions = None

    def step(self, example, step):
        instruction = example["lang"]
        if instruction != self.task_description:
            self.reset(instruction)
        if self.raw_actions is None or step % self.action_chunk_size == 0:
            images = [cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                      for image in example["image"]]
            response = self.client.predict_action({
                "examples": [{"lang": instruction, "image": images}],
                "do_sample": False,
                "use_ddim": True,
            })
            normalized = np.asarray(response["data"]["normalized_actions"])[0]
            if normalized.shape != (self.action_chunk_size, 14):
                raise ValueError(f"Expected {self.action_chunk_size}x14 actions, got {normalized.shape}")
            mask = np.asarray(self.action_stats["mask"], dtype=bool)
            low = np.asarray(self.action_stats["min"])
            high = np.asarray(self.action_stats["max"])
            normalized = np.clip(normalized, -1, 1)
            self.raw_actions = np.where(
                mask,
                0.5 * (normalized + 1) * (high - low) + low,
                (normalized >= 0.49).astype(normalized.dtype),
            )
        action = self.raw_actions[min(step % self.action_chunk_size, len(self.raw_actions) - 1)]
        return action[[0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]


def get_model(usr_args):
    return ModelClient(
        policy_ckpt_path=usr_args["policy_ckpt_path"],
        host=usr_args.get("host", "127.0.0.1"),
        port=usr_args.get("port", 5694),
        unnorm_key=usr_args.get("unnorm_key", "new_embodiment"),
        action_mode=usr_args.get("action_mode", "abs"),
    )


def reset_model(model):
    model.reset("")


def eval(TASK_ENV, model, observation):
    images = [
        observation["observation"][camera]["rgb"]
        for camera in ("head_camera", "left_camera", "right_camera")
    ]
    action = model.step(
        {"lang": str(TASK_ENV.get_instruction()), "image": images},
        step=TASK_ENV.take_action_cnt,
    )
    TASK_ENV.take_action(action)
