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

"""Weighted dataset mixture for combining multiple datasets with controlled sampling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


@dataclass
class DatasetMixtureMetadata:
    """Minimal metadata wrapper for dataset mixtures."""

    stats: dict[str, dict[str, np.ndarray]] | None
    camera_keys: list[str]
    tasks: object
    episodes: object
    features: dict
    fps: int
    total_frames: int
    total_episodes: int


def _aggregate_stats(
    stats_list: list[dict[str, dict[str, np.ndarray]]],
    weights: list[float],
) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats across datasets with a simple weighted strategy.

    - mean/std: weighted average
    - min/max: global min/max
    - quantiles: weighted average (approx)
    """

    if not stats_list:
        return {}

    weights_arr = np.array(weights, dtype=np.float64)
    weights_arr = weights_arr / weights_arr.sum()

    all_keys = set.intersection(*(set(s.keys()) for s in stats_list))
    aggregated: dict[str, dict[str, np.ndarray]] = {}

    for key in all_keys:
        per_key_stats = [s[key] for s in stats_list]
        stat_names = set.intersection(*(set(ps.keys()) for ps in per_key_stats))
        aggregated[key] = {}
        for stat_name in stat_names:
            values = [ps[stat_name] for ps in per_key_stats]
            if stat_name == "min":
                aggregated[key][stat_name] = np.min(values, axis=0)
            elif stat_name == "max":
                aggregated[key][stat_name] = np.max(values, axis=0)
            else:
                stacked = np.stack(values, axis=0)
                reshape_dims = (len(weights_arr),) + (1,) * (stacked.ndim - 1)
                w = weights_arr.reshape(reshape_dims)
                aggregated[key][stat_name] = (stacked * w).sum(axis=0)

    return aggregated


class HierarchicalSampler(Sampler[tuple[int, int]]):
    """Two-level weighted sampling: dataset index then within-dataset index."""

    def __init__(self, lengths: Iterable[int], weights: Iterable[float], num_samples: int | None = None):
        self.lengths = list(lengths)
        if any(l <= 0 for l in self.lengths):
            raise ValueError("All datasets must be non-empty for sampling.")
        weights_arr = np.array(list(weights), dtype=np.float64)
        if weights_arr.sum() <= 0:
            raise ValueError("Weights must sum to a positive value.")
        self.weights = torch.tensor(weights_arr / weights_arr.sum(), dtype=torch.float32)
        self.num_samples = num_samples or int(sum(self.lengths))

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        for _ in range(self.num_samples):
            dataset_idx = int(torch.multinomial(self.weights, 1).item())
            sample_idx = int(torch.randint(0, self.lengths[dataset_idx], (1,)).item())
            yield (dataset_idx, sample_idx)


class WeightedDatasetMixture(Dataset):
    """Dataset wrapper that samples from multiple datasets using a sampler."""

    def __init__(self, datasets: list[Dataset], weights: list[float]):
        if len(datasets) != len(weights):
            raise ValueError("datasets and weights must have the same length")
        self.datasets = datasets
        self.weights = weights
        self.lengths = [len(ds) for ds in datasets]

        meta0 = getattr(datasets[0], "meta", None)
        if meta0 is None:
            raise ValueError("Datasets must expose a .meta attribute")

        stats_list = [getattr(ds, "meta").stats for ds in datasets]
        aggregated_stats = None
        if all(s is not None for s in stats_list):
            aggregated_stats = _aggregate_stats(stats_list, weights)
        else:
            logging.warning("Some datasets missing stats; using stats from the first dataset.")
            aggregated_stats = meta0.stats

        self.meta = DatasetMixtureMetadata(
            stats=aggregated_stats,
            camera_keys=getattr(meta0, "camera_keys", []),
            tasks=getattr(meta0, "tasks", None),
            episodes=getattr(meta0, "episodes", None),
            features=getattr(meta0, "features", {}),
            fps=getattr(meta0, "fps", 0),
            total_frames=sum(getattr(ds, "num_frames", len(ds)) for ds in datasets),
            total_episodes=sum(getattr(ds, "num_episodes", 0) for ds in datasets),
        )

        self.num_frames = self.meta.total_frames
        self.num_episodes = self.meta.total_episodes

    def __len__(self) -> int:
        return int(sum(self.lengths))

    def __getitem__(self, index):
        if isinstance(index, tuple):
            dataset_idx, sample_idx = index
            return self.datasets[dataset_idx][sample_idx]

        # Fallback: treat as concatenated dataset
        idx = int(index)
        for ds_idx, ds_len in enumerate(self.lengths):
            if idx < ds_len:
                return self.datasets[ds_idx][idx]
            idx -= ds_len
        raise IndexError("Index out of range")

    def get_sampler(self) -> HierarchicalSampler:
        return HierarchicalSampler(self.lengths, self.weights, num_samples=len(self))
