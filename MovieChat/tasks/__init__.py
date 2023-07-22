"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from MovieChat.common.registry import registry
from MovieChat.tasks.base_task import BaseTask
from MovieChat.tasks.image_text_pretrain import ImageTextPretrainTask
from MovieChat.tasks.video_text_pretrain import VideoTextPretrainTask


def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "BaseTask",
    "ImageTextPretrainTask",
    "VideoTextPretrainTask"
]
