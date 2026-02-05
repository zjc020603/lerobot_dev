#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from pprint import pformat

import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_mixture import WeightedDatasetMixture
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def _make_single_dataset(cfg: TrainPipelineConfig, dataset_cfg: DatasetConfig) -> LeRobotDataset:
    image_transforms = (
        ImageTransforms(dataset_cfg.image_transforms)
        if dataset_cfg.image_transforms.enable
        else None
    )

    if isinstance(dataset_cfg.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            dataset_cfg.repo_id, root=dataset_cfg.root, revision=dataset_cfg.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        if not dataset_cfg.streaming:
            dataset = LeRobotDataset(
                dataset_cfg.repo_id,
                root=dataset_cfg.root,
                episodes=dataset_cfg.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=dataset_cfg.revision,
                video_backend=dataset_cfg.video_backend,
                tolerance_s=cfg.tolerance_s,
                policy_cfg=cfg.policy,
            )
        else:
            dataset = StreamingLeRobotDataset(
                dataset_cfg.repo_id,
                root=dataset_cfg.root,
                episodes=dataset_cfg.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=dataset_cfg.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")

    if dataset_cfg.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            stats_key = key
            if stats_key not in dataset.meta.stats:
                if stats_key.startswith("observation.images."):
                    stats_key = stats_key.replace("observation.images.", "", 1)
                elif stats_key.startswith("observation.image."):
                    stats_key = stats_key.replace("observation.image.", "", 1)

            if stats_key not in dataset.meta.stats:
                logging.warning("Skipping ImageNet stats for missing key: %s", key)
                continue

            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[stats_key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset | WeightedDatasetMixture:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    if cfg.dataset_mixture is not None:
        datasets = [_make_single_dataset(cfg, ds_cfg) for ds_cfg in cfg.dataset_mixture.datasets]
        target_fps = cfg.dataset_mixture.action_freq
        for ds in datasets:
            if getattr(ds.meta, "fps", None) != target_fps:
                raise ValueError(
                    "Dataset fps mismatch for mixture: expected action_freq={} Hz, got {} Hz for repo_id={}."
                    " Please resample or align fps before training."
                    .format(target_fps, ds.meta.fps, getattr(ds.meta, "repo_id", "unknown"))
                )
        return WeightedDatasetMixture(datasets, cfg.dataset_mixture.weights)

    if cfg.dataset is None:
        raise ValueError("`dataset` is required when dataset_mixture is not set.")

    return _make_single_dataset(cfg, cfg.dataset)
