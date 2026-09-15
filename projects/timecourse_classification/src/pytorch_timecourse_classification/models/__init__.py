"""Model registry for time-course classification experiments."""

from torch import nn

from .attention import AttentionClassifier
from .cnn import CNNClassifier, scaled_cnn_config
from .cnn_attention_pooling import CNNAttentionPoolingClassifier
from .extended_cnns import ExtendedCNNClassifier
from .hierarchical_patch_transformer import HierarchicalPatchTransformerClassifier


MODEL_TYPES: dict[str, type[nn.Module]] = {
    ExtendedCNNClassifier.model_name: ExtendedCNNClassifier,
    CNNClassifier.model_name: CNNClassifier,
    AttentionClassifier.model_name: AttentionClassifier,
    CNNAttentionPoolingClassifier.model_name: CNNAttentionPoolingClassifier,
    HierarchicalPatchTransformerClassifier.model_name: (
        HierarchicalPatchTransformerClassifier
    ),
}


def build_model(model_name: str, **model_config: object) -> nn.Module:
    """Instantiate a registered model from its saved name and configuration."""
    try:
        model_type = MODEL_TYPES[model_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown model {model_name!r}; expected one of {sorted(MODEL_TYPES)}."
        ) from error
    return model_type(**model_config)


__all__ = [
    "AttentionClassifier",
    "CNNClassifier",
    "CNNAttentionPoolingClassifier",
    "ExtendedCNNClassifier",
    "HierarchicalPatchTransformerClassifier",
    "MODEL_TYPES",
    "build_model",
    "scaled_cnn_config",
]
