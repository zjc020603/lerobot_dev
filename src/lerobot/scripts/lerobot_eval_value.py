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
"""Offline evaluation for Value policy.

Usage example:

lerobot-eval-value \
    --policy.path=/data/zjc/workspace/lerobot_dev/outputs/value_training/checkpoints/001000/pretrained_model \
    --dataset.repo_id=lerobot/libero \
    --dataset.root=/data/zjc/workspace/lerobot_dev/data \
    --batch_size=32 \
    --max_batches=200 \
    --policy.device=cuda
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import IMAGENET_STATS, resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


@dataclass
class ValueEvalConfig:
    dataset: DatasetConfig
    policy: PreTrainedConfig | None = None
    batch_size: int = 32
    num_workers: int = 4
    max_batches: int | None = None
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
        """Enable --policy.path style arguments."""
        return ["policy"]


def _make_dataset(cfg: ValueEvalConfig):
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

    if not cfg.dataset.streaming:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
            tolerance_s=cfg.tolerance_s,
            policy_cfg=cfg.policy,
        )
    else:
        dataset = StreamingLeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=cfg.dataset.revision,
            max_num_shards=cfg.num_workers,
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
def eval_value(cfg: ValueEvalConfig):
    cfg.validate()
    init_logging()

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = get_safe_torch_device(cfg.policy.device, log=True)

    dataset = _make_dataset(cfg)

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    processor_kwargs: dict[str, Any] = {}
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    policy = policy.to(device)
    policy.eval()

    total_ce = 0.0
    total_l1 = 0.0
    total_acc = 0.0
    total_count = 0

    # Online stats for Pearson correlation
    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    sum_xy = 0.0

    max_batches = cfg.max_batches

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = preprocessor(batch)

            images, img_masks = policy.prepare_images(batch)
            lang_tokens = batch[OBS_LANGUAGE_TOKENS]
            lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
            state = batch.get("state")
            if state is None:
                state = batch.get(OBS_STATE)

            logits = policy.model.forward(images, img_masks, lang_tokens, lang_masks, state)
            logits = logits.to(dtype=torch.float32)

            values = policy.calculate_value(logits)
            target_bin = batch["return_bin_idx"].to(dtype=torch.long)
            target_value = batch["return_continuous"].to(dtype=torch.float32)

            ce = F.cross_entropy(logits, target_bin, reduction="mean")
            l1 = F.l1_loss(values, target_value, reduction="mean")
            acc = (logits.argmax(dim=-1) == target_bin).float().mean()

            bsz = target_bin.shape[0]
            total_ce += ce.item() * bsz
            total_l1 += l1.item() * bsz
            total_acc += acc.item() * bsz
            total_count += bsz

            x = values.detach().to(dtype=torch.float64)
            y = target_value.detach().to(dtype=torch.float64)
            sum_x += x.sum().item()
            sum_y += y.sum().item()
            sum_x2 += (x * x).sum().item()
            sum_y2 += (y * y).sum().item()
            sum_xy += (x * y).sum().item()

    if total_count == 0:
        raise RuntimeError("No samples were processed in evaluation.")

    mean_ce = total_ce / total_count
    mean_l1 = total_l1 / total_count
    mean_acc = total_acc / total_count

    n = float(total_count)
    cov = sum_xy - (sum_x * sum_y / n)
    var_x = sum_x2 - (sum_x * sum_x / n)
    var_y = sum_y2 - (sum_y * sum_y / n)
    if var_x <= 0 or var_y <= 0:
        corr = float("nan")
    else:
        corr = cov / (var_x * var_y) ** 0.5

    logging.info("Value eval results:")
    logging.info("  samples: %d", total_count)
    logging.info("  CE loss: %.6f", mean_ce)
    logging.info("  L1 loss: %.6f", mean_l1)
    logging.info("  Accuracy: %.4f", mean_acc)
    logging.info("  Pearson corr(value, return): %.4f", corr)


def main():
    register_third_party_plugins()
    eval_value()


if __name__ == "__main__":
    main()
