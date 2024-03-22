"""Compositional prompt ensemble for WinCLIP."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as  F
from open_clip.tokenizer import tokenize


NORMAL_STATES = [
    "flawless {}",
    "perfect {}",
    "unblemished {}",
]

ANOMALOUS_STATES = [
    "damaged {}",
    "flaw {}",
    "defect {}"
]

TEMPLATES = [
    "a cropped photo of the {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "a bright photo of a {}.",
    "a bright photo of the {}.",
    "a dark photo of the {}.",
    "a dark photo of a {}.",
    "a jpeg corrupted photo of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of the {}.",
    "a blurry photo of a {}.",
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
]

def create_prompt_ensemble(class_name: str = "object") -> tuple[list[str], list[str]]:
    normal_states = [state.format(class_name) for state in NORMAL_STATES]
    normal_ensemble = [template.format(state) for state in normal_states for template in TEMPLATES]

    anomalous_states = [state.format(class_name) for state in ANOMALOUS_STATES]
    anomalous_ensemble = [template.format(state) for state in anomalous_states for template in TEMPLATES]
    return normal_ensemble, anomalous_ensemble



def text_global_pool(x, text= None, pool_type='argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x
    return pooled, tokens


