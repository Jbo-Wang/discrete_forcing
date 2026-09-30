"""Small dependency/config check; --initialize additionally loads real training inputs."""
import argparse
import os
import tempfile
from pathlib import Path
from omegaconf import OmegaConf

def main():
    for source in Path('.').rglob('*.py'):
        if 'third_party' not in source.parts:
            compile(source.read_text(encoding='utf-8'), str(source), 'exec')
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--model-path')
    parser.add_argument('--data-root')
    args = parser.parse_args()
    if args.evaluation:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                from examples.LIBERO.eval_files.eval_libero import eval_libero
            finally:
                os.chdir(previous)
        print('Evaluation imports OK')
        return
    from starVLA.training.train_starvla import VLATrainer
    from starVLA.model.framework import build_framework
    from starVLA.model.framework.QwenPILF_v3 import QwenPILFv3
    from starVLA.dataloader import build_dataloader
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    from deployment.model_server.server_policy import build_argparser
    cfg = OmegaConf.load('examples/LIBERO/train_files/libero_method.yaml')
    assert cfg.framework.action_model.action_horizon == 8
    assert cfg.framework.action_model.isolate_branch_tokens_before_shared
    print('Training/server imports and configuration OK')
    if not args.initialize:
        return
    import torch.distributed as dist
    if args.model_path:
        cfg.framework.qwenvl.base_vlm = args.model_path
    if args.data_root:
        cfg.datasets.vla_data.data_root_dir = args.data_root
    with tempfile.TemporaryDirectory() as temporary:
        cfg.output_dir = temporary
        dist.init_process_group('gloo', init_method=Path(temporary, 'dist').as_uri(), rank=0, world_size=1)
        try:
            model = build_framework(cfg)
            loader = build_dataloader(cfg)
            batch = next(iter(loader))
            assert len(batch) == cfg.datasets.vla_data.per_device_batch_size
            print('Model and dataloader initialized; one real batch loaded')
        finally:
            dist.destroy_process_group()

if __name__ == '__main__':
    main()
