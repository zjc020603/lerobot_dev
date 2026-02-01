#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""Value Function Model using SIGLIP and Gemma 3 270M

A value function model that estimates state values for reinforcement learning.
Uses SIGLIP for vision encoding and Gemma 3 270M for language processing.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.value.configuration_value import ValueConfig
from lerobot.policies.value.siglip_gemma import (
    SiglipGemmaValueConfig,
    SiglipGemmaValueModel,
)
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def make_att_2d_masks(pad_masks, att_masks):
    """Creates a 2-D attention mask given padding and 1-D attention masks."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    """Resizes an image while preserving aspect ratio and padding to target dimensions."""
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


class ValueFunction(PreTrainedPolicy):
    """Wrapper class around Value Function model to train and run inference within LeRobot."""

    config_class = ValueConfig
    name = "value"

    def __init__(
        self,
        config: ValueConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        dataset_meta: object | None = None,
        **kwargs,
    ):
        """Initializes the ValueFunction policy."""

        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ValueModel(config)

    def reset(self):
        pass

    def get_optim_params(self) -> dict:
        return self.parameters()

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("Value functions do not select actions. Use predict_value() instead.")

    def sample_actions(self, batch: dict[str, Tensor], noise: Tensor = None):
        raise NotImplementedError("Value functions do not sample actions. Use predict_value() instead.")

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("Value functions do not predict actions. Use predict_value() instead.")

    def calculate_value(self, logits: Tensor) -> Tensor:
        start_idx = torch.linspace(
            -1,
            -1 / self.config.reward_config.number_of_bins,
            self.config.reward_config.number_of_bins,
            device=logits.device,
        )
        end_idx = torch.linspace(
            -1 + 1 / self.config.reward_config.number_of_bins,
            0,
            self.config.reward_config.number_of_bins,
            device=logits.device,
        )

        mid_idx = rearrange((start_idx + end_idx) / 2, "n -> 1 n")

        value = torch.softmax(logits, dim=-1).to(dtype=torch.float32) @ mid_idx.T

        return rearrange(value, "b 1 -> b")

    @torch.no_grad()
    def predict_value(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        #这里比opentau少了normalization，放到了preprocessor中，更符合lerobot的设计理念
        images, img_masks = self.prepare_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        state = batch.get("state")
        if state is None:
            state = batch.get(OBS_STATE)

        logits = self.model.forward(images, img_masks, lang_tokens, lang_masks, state)
        return self.calculate_value(logits)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor] | None]:
        images, img_masks = self.prepare_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        state = batch.get("state")
        if state is None:
            state = batch.get(OBS_STATE)

        logits = self.model.forward(images, img_masks, lang_tokens, lang_masks, state)
        
        #debug：加入断言以判断因idx越界导致的问题
        assert batch["return_bin_idx"].dtype == torch.long
        assert batch["return_bin_idx"].min().item() >= 0
        assert batch["return_bin_idx"].max().item() < logits.shape[-1], (
            f"return_bin_idx out of range: "
            f"[{batch['return_bin_idx'].min().item()}, "
            f"{batch['return_bin_idx'].max().item()}], "
            f"num_bins={logits.shape[-1]}"
        )
        
        values = self.calculate_value(logits)
        logits = logits.to(dtype=torch.float32)
        batch["return_bin_idx"] = batch["return_bin_idx"].to(dtype=torch.long)
        loss = F.cross_entropy(logits, batch["return_bin_idx"])

        l1_loss = F.l1_loss(values, batch["return_continuous"])
        accuracy = (logits.argmax(dim=-1) == batch["return_bin_idx"]).float().mean()


        #opentau的版本中policy.forward的返回值是一个dict
        #在我们的lerobot版本中是一个tuple(loss, dict)
        output_dict = {
            "CE": loss.detach(),
            "L1": l1_loss.detach(),
            "Accuracy": accuracy.detach(),
        }

        return loss, output_dict
         

        # return {
        #     "MSE": torch.zeros_like(loss, requires_grad=False),
        #     "CE": loss,
        #     "L1": l1_loss,
        #     "Accuracy": accuracy,
        # }

    def prepare_images(self, batch):
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key]

            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks


class ValueModel(nn.Module):
    """Value Function Model using SIGLIP and Gemma 3 270M."""

    CLASSIFICATION_TOKEN_ID = 6

    def __init__(self, config):
        super().__init__()
        self.config = config

        siglip_gemma_value_config = SiglipGemmaValueConfig(
            num_value_bins=self.config.reward_config.number_of_bins
        )
        self.siglip_gemma_value = SiglipGemmaValueModel(siglip_gemma_value_config)

        self.state_proj = nn.Linear(self.config.max_state_dim, 640)
        self.multi_modal_proj = nn.Linear(1152, 640)
        self.bins = config.reward_config.number_of_bins
        self.c_neg = config.reward_config.C_neg

    def embed_sequence(self, images, img_masks, lang_tokens, lang_masks, state):
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=False):
            img_emb = self.siglip_gemma_value.embed_image(img)
            img_emb = img_emb.to(dtype=torch.bfloat16)
            img_emb = img_emb.to(dtype=torch.float32)
            img_emb = self.multi_modal_proj(img_emb)
            img_emb = img_emb.to(dtype=torch.bfloat16)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * num_img_embs

        lang_emb = self.siglip_gemma_value.embed_language_tokens(lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state = pad_vector(state, self.config.max_state_dim)
        state_emb = self.state_proj(state)
        state_emb = state_emb.to(dtype=torch.bfloat16)
        embs.append(state_emb[:, None, :])

        state_mask = torch.ones(state_emb.shape[0], 1, dtype=torch.bool, device=state_emb.device)
        pad_masks.append(state_mask)
        att_masks += [0]

        cls_token = torch.full(
            (bsize, 1), self.CLASSIFICATION_TOKEN_ID, device=state_emb.device, dtype=torch.long
        )
        cls_token_emb = self.siglip_gemma_value.gemma.embed_tokens(cls_token)
        embs.append(cls_token_emb)
        pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=state_emb.device))
        att_masks += [0]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def forward(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embs, pad_masks, att_masks = self.embed_sequence(images, img_masks, lang_tokens, lang_masks, state)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        logits = self.siglip_gemma_value.forward(
            inputs_embeds=embs,
            attention_mask=att_2d_masks,
            position_ids=position_ids,
        )

        return logits
