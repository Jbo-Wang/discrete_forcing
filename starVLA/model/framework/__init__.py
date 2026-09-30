from starVLA.model.tools import FRAMEWORK_REGISTRY


def build_framework(cfg):
    from starVLA.model.framework.QwenPILF_v3 import QwenPILFv3

    if cfg.framework.name != "QwenPILF_v3":
        raise ValueError("This package provides the LIBERO QwenPILF_v3 method.")
    return QwenPILFv3(cfg)
