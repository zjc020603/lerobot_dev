#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
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
"""Compute advantage sidecar and percentiles for a dataset.

Usage example:

lerobot-get-advantage \
    --policy.path=/data/zjc/workspace/lerobot_dev/outputs/value_training/checkpoints/001000/pretrained_model \
    --dataset.repo_id=lerobot/libero \
    --dataset.root=/data/zjc/workspace/lerobot_dev/data \
    --batch_size=32 \
    --policy.device=cuda
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import IMAGENET_STATS, resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.reward import calculate_n_step_return
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.utils import ADVANTAGES_PATH
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


def _ensure_scalar(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    return value


_default0 = defaultdict(int)


@dataclass
class AdvantageConfig:
    dataset: DatasetConfig
    policy: PreTrainedConfig | None = None
    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int | None = None
    seed: int | None = 1000
    tolerance_s: float = 1e-4
    rename_map: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)

        if self.policy is None:
            raise ValueError("Policy is not configured. Please specify --policy.path=...")
        if self.policy.type != "value":
            raise ValueError(f"This script only supports value policy, got: {self.policy.type}")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def _make_dataset(cfg: AdvantageConfig) -> LeRobotDataset:
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )
    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        tolerance_s=cfg.tolerance_s,
    )

    if cfg.dataset.use_imagenet_stats:
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


@parser.wrap()
def main(cfg: AdvantageConfig):
    cfg.validate()
    init_logging()
    register_third_party_plugins()

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = get_safe_torch_device(cfg.policy.device, log=True)

    dataset = _make_dataset(cfg)

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    processor_kwargs: dict[str, object] = {
        "dataset_stats": dataset.meta.stats,
        "preprocessor_overrides": {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    }

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )

    policy = policy.to(device)
    policy.eval()

    advantages = []
    values = {}
    ds_advantage = {}

    with torch.inference_mode():
        # First pass: compute v0 and reward per step
        for batch in dataloader:
            batch = preprocessor(batch)
            v0_batch = policy.predict_value(batch)

            episode_indices = batch["episode_index"]
            current_indices = batch["index"]
            timestamps = batch["timestamp"]

            for ep_idx, current_idx, ts, v0 in zip(
                episode_indices, current_indices, timestamps, v0_batch, strict=True
            ):
                ep_idx = int(_ensure_scalar(ep_idx))
                current_idx = int(_ensure_scalar(current_idx))
                ts = float(_ensure_scalar(ts))
                v0 = float(_ensure_scalar(v0))

                ep = dataset.meta.episodes[ep_idx]
                if "dataset_to_index" in ep:
                    episode_end_idx = ep["dataset_to_index"] - 1
                elif "dataset_from_index" in ep and "length" in ep:
                    episode_end_idx = ep["dataset_from_index"] + ep["length"] - 1
                else:
                    raise ValueError("Episode metadata missing dataset_to_index or length for return computation.")

                success = ep.get("success", True)
                reward = calculate_n_step_return(
                    success=bool(success),
                    n_steps_look_ahead=cfg.policy.reward_config.N_steps_look_ahead,
                    episode_end_idx=episode_end_idx,
                    reward_normalizer=cfg.policy.reward_config.reward_normalizer,
                    current_idx=current_idx,
                    c_neg=cfg.policy.reward_config.C_neg,
                )

                values[(ep_idx, current_idx)] = {
                    "v0": v0,
                    "reward": reward,
                }
                ds_advantage[(ep_idx, ts)] = None

        # Second pass: compute advantages
        for batch in dataloader:
            episode_indices = batch["episode_index"]
            current_indices = batch["index"]
            timestamps = batch["timestamp"]

            for ep_idx, current_idx, ts in zip(
                episode_indices, current_indices, timestamps, strict=True
            ):
                ep_idx = int(_ensure_scalar(ep_idx))
                current_idx = int(_ensure_scalar(current_idx))
                ts = float(_ensure_scalar(ts))

                look_ahead_idx = current_idx + cfg.policy.reward_config.N_steps_look_ahead
                vn = values.get((ep_idx, look_ahead_idx), _default0)["v0"]
                reward = values.get((ep_idx, current_idx), _default0)["reward"]
                v0 = values.get((ep_idx, current_idx), _default0)["v0"]

                advantage = reward + vn - v0
                advantages.append(advantage)
                ds_advantage[(ep_idx, ts)] = advantage

    # Write sidecar file
    advantage_json = {f"{ep_idx},{ts}": val for (ep_idx, ts), val in ds_advantage.items()}
    out_path = Path(dataset.root) / ADVANTAGES_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(advantage_json, f, indent=2)

    # Percentiles for threshold selection
    percentiles = list(range(0, 101, 5))
    advantage_percentiles = np.percentile(np.array(advantages), percentiles)

    print("Advantage percentiles for deciding epsilon threshold:")
    for p, val in zip(percentiles, advantage_percentiles, strict=False):
        print(f"  {p:03d}th percentile: {val:.6f}")


if __name__ == "__main__":
    main()
