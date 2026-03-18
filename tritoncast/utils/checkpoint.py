from collections import OrderedDict

import torch


def _strip_prefix(key, prefixes):
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _extract_state_dict(checkpoint, state_keys):
    if isinstance(checkpoint, dict):
        for state_key in state_keys:
            if state_key in checkpoint and isinstance(checkpoint[state_key], dict):
                return checkpoint[state_key]
    return checkpoint


def load_model_checkpoint(
    model,
    checkpoint_path,
    device=None,
    state_keys=("model_state",),
    strip_prefixes=("module.",),
    skip_keys=("ged",),
    strict=True,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint, state_keys)

    cleaned_state_dict = OrderedDict()
    for key, value in state_dict.items():
        normalized_key = _strip_prefix(key, strip_prefixes)
        if normalized_key in skip_keys:
            continue
        cleaned_state_dict[normalized_key] = value

    model.load_state_dict(cleaned_state_dict, strict=strict)
    model.eval()
    return model

