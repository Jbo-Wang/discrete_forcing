from pathlib import Path

import torch.distributed as dist
from torch.utils.data import DataLoader


def build_dataloader(cfg, dataset_py="lerobot_datasets"):
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    if dataset_py != "lerobot_datasets":
        raise ValueError("This package uses the LIBERO LeRobot loader.")
    vla_dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.datasets.vla_data.per_device_batch_size,
        collate_fn=collate_fn,
        num_workers=4,
    )
    if dist.get_rank() == 0:
        vla_dataset.save_dataset_statistics(Path(cfg.output_dir) / "dataset_statistics.json")
    return dataloader
