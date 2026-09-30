# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum


class EmbodimentTag(Enum):
    NEW_EMBODIMENT = "new_embodiment"
    FRANKA = "franka"


EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.NEW_EMBODIMENT.value: 31,
    EmbodimentTag.FRANKA.value: 25,
}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {"libero_franka": EmbodimentTag.FRANKA}
