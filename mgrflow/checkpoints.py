"""Save portable inference weights and load the original full MGRFlow exports."""

from dataclasses import asdict
from pathlib import Path

import torch

from .model import MGRFlow, ModelConfig


def save_weights(
    path: Path, model: MGRFlow, marker_keys: list[str], dataset: str, p99: dict, **info
) -> None:
    payload = {
        "model_config": asdict(model.config),
        "model": model.state_dict(),
        "marker_keys": marker_keys,
        "dataset": dataset,
        "p99": p99,
        **info,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def original_weights(payload: dict) -> dict:
    """Translate parameter names in the original native-256 full-model export."""
    contract = payload["contract"]
    architecture = contract["architecture"]
    if not (
        architecture.get("conditioning_version") == "adaln4_v2"
        and architecture.get("latent_shape") == [4, 32, 32]
        and architecture.get("pathology_feature_mode", "multi_level") == "multi_level"
        and architecture.get("aligned_pathology_injection")
        and architecture.get("relationship_model")
        == "same_position_marker_axis_attention_every_block"
    ):
        raise ValueError("Use a native-256 adaln4_v2 full-model export or an MGRFlow checkpoint.")
    prefixes = {
        "marker_embedding.": "smm.marker_embedding.",
        "time_embedding.": "smm.time_embedding.",
        "condition_fusion.": "smm.condition_fusion.",
        "pathology_adapter.": "mma.",
        "aligned_context_norm.": "mira_input.norm.",
        "aligned_context_projection.": "mira_input.projection.",
    }
    state = {}
    for key, value in payload["model"].items():
        for source, target in prefixes.items():
            if key.startswith(source):
                key = target + key[len(source) :]
                break
        state[key] = value
    return {
        "model": state,
        "model_config": {
            name: architecture[name] for name in ("width", "depth", "heads", "mlp_ratio")
        }
        | {"pathology_width": state["mma.branches.0.1.weight"].shape[1]},
        "marker_keys": contract["marker_registry"]["ordered_marker_keys"],
        "dataset": contract["datasets"][0],
        "p99": None,
    }


def load_weights(path: str | Path, device: torch.device) -> tuple[MGRFlow, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") == "marker_pathology_flow_weights_v1":
        payload = original_weights(payload)
    state = payload.pop("model")
    config = ModelConfig(**payload["model_config"])
    model = MGRFlow(state["smm.marker_embedding.table"], config)
    model.load_state_dict(state)
    return model.eval().to(device), payload
