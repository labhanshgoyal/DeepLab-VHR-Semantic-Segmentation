from models.deeplabv1 import DeepLabLargeFOV
from models.deeplabv2 import DeepLabV2
from models.deeplabv3 import DeepLabV3
from models.deeplabv3plus import DeepLabV3Plus

# default architecture configs for each model
# these match the original paper settings
MODEL_DEFAULTS = {
    "deeplabv1": {},
    "deeplabv2": {
        "n_blocks": [3, 4, 23, 3],
        "atrous_rates": [6, 12, 18, 24],
        "output_stride": 8,
    },
    "deeplabv3": {
        "n_blocks": [3, 4, 23, 3],
        "atrous_rates": [6, 12, 18],
        "multi_grids": [1, 2, 4],
        "output_stride": 8,
    },
    "deeplabv3plus": {
        "n_blocks": [3, 4, 23, 3],
        "atrous_rates": [6, 12, 18],
        "multi_grids": [1, 2, 4],
        "output_stride": 16,
        "atr_sep_conv": True,
    },
}

MODEL_REGISTRY = {
    "deeplabv1": DeepLabLargeFOV,
    "deeplabv2": DeepLabV2,
    "deeplabv3": DeepLabV3,
    "deeplabv3plus": DeepLabV3Plus,
}

def get_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(MODEL_REGISTRY.keys())}")
    # merge defaults with user overrides (user kwargs win)
    config = {**MODEL_DEFAULTS.get(name, {}), **kwargs}
    return MODEL_REGISTRY[name](**config)