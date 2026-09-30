# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import argparse
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework
import torch, os


def main(args) -> None:
    vla = baseframework.from_pretrained(
        args.ckpt_path,
    )

    if args.num_inference_steps is not None:
        if args.num_inference_steps <= 0:
            raise ValueError("--num_inference_steps must be a positive integer")
        if not hasattr(vla, "action_model") or not hasattr(vla.action_model, "num_inference_timesteps"):
            raise AttributeError("Loaded policy does not expose action_model.num_inference_timesteps")
        vla.action_model.num_inference_timesteps = args.num_inference_steps
    if args.num_discrete_steps is not None:
        if args.num_discrete_steps <= 0:
            raise ValueError("--num_discrete_steps must be a positive integer")
        vla.action_model.default_num_discrete_steps = args.num_discrete_steps
    if args.num_continuous_steps is not None:
        if args.num_continuous_steps <= 0:
            raise ValueError("--num_continuous_steps must be a positive integer")
        vla.action_model.default_num_continuous_steps = args.num_continuous_steps
    vla.action_model.inference_branch_mode = args.inference_branch_mode
    effective_discrete_steps = vla.action_model.default_num_discrete_steps
    effective_continuous_steps = vla.action_model.default_num_continuous_steps
    logging.info(
        "Using inference mode %s: %s discrete + %s continuous steps",
        args.inference_branch_mode,
        effective_discrete_steps,
        effective_continuous_steps,
    )

    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()

    # start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "libero"},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout" , type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--num_discrete_steps", type=int, default=None)
    parser.add_argument("--num_continuous_steps", type=int, default=None)
    parser.add_argument(
        "--inference_branch_mode",
        choices=("hybrid",),
        default="hybrid",
    )
    return parser




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
