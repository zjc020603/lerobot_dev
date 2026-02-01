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
"""Reward calculation utilities for dataset-side value labels."""


def calculate_return_bins_with_equal_width(
    success: bool,
    b: int,
    episode_end_idx: int,
    reward_normalizer: int,
    current_idx: int,
    c_neg: float = -100.0,
) -> tuple[int, float]:
    """Sparse reward function to train value function network."""
    return_value = current_idx - episode_end_idx + 1
    if not success:
        return_value += c_neg

    return_normalized = return_value / reward_normalizer
    bin_idx = int((return_normalized + 1) * (b - 1))
    return bin_idx, return_normalized


def calculate_n_step_return(
    success: bool,
    n_steps_look_ahead: int,
    episode_end_idx: int,
    reward_normalizer: int,
    current_idx: int,
    c_neg: float = -100.0,
) -> float:
    """Sparse reward function to calculate advantage."""
    return_value = max(current_idx - episode_end_idx + 1, -1 * n_steps_look_ahead)
    if not success and current_idx + n_steps_look_ahead >= episode_end_idx:
        return_value += c_neg

    return_normalized = return_value / reward_normalizer
    return return_normalized
