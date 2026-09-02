from models.deeplabv1 import DeepLabLargeFOV
from models.deeplabv2 import DeepLabV2
from models.deeplabv3 import DeepLabV3
from models.deeplabv3plus import DeepLabV3Plus

MODEL_REGISTRY = {
    "deeplabv1": DeepLabLargeFOV,
    "deeplabv2": DeepLabV2,
    "deeplabv3": DeepLabV3,
    "deeplabv3plus": DeepLabV3Plus,
}

def get_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)