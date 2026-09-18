import torch
import numpy as np
import random
from torchvision import transforms
from PIL import Image


class JointCompose:
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class JointResize:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image, mask):
        # image is PIL, mask is numpy [H, W, C]
        image = transforms.functional.resize(image, self.size, interpolation=Image.BILINEAR)

        # resize each mask channel with NEAREST (no interpolation artifacts)
        h, w = self.size
        C = mask.shape[2]
        resized = np.zeros((h, w, C), dtype=np.float32)
        for c in range(C):
            # convert single channel to PIL, resize, convert back
            ch = Image.fromarray(mask[:, :, c].astype(np.uint8))
            ch = transforms.functional.resize(ch, self.size, interpolation=Image.NEAREST)
            resized[:, :, c] = np.array(ch).astype(np.float32)
        return image, resized


class JointRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = transforms.functional.hflip(image)
            mask = np.flip(mask, axis=1).copy()  # flip width axis
        return image, mask


class JointRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = transforms.functional.vflip(image)
            mask = np.flip(mask, axis=0).copy()  # flip height axis
        return image, mask


class JointRandomRotation:
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, image, mask):
        angle = random.choice(self.angles)
        if angle != 0:
            image = transforms.functional.rotate(image, angle)
            # np.rot90: k=1 is 90° counterclockwise
            k = angle // 90
            mask = np.rot90(mask, k=k, axes=(0, 1)).copy()
        return image, mask


class JointColorJitter:
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.2):
        self.jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation
        )

    def __call__(self, image, mask):
        # only jitter the image, not the mask
        image = self.jitter(image)
        return image, mask


class JointToTensor:
    def __call__(self, image, mask):
        # image: PIL → tensor [C, H, W]
        image = transforms.functional.to_tensor(image)
        # mask: numpy [H, W, C] → tensor [C, H, W]
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()
        return image, mask


class JointNormalize:
    def __init__(self):
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, image, mask):
        image = self.normalize(image)
        return image, mask


def get_train_transforms(image_size=512):
    return JointCompose([
        JointResize(image_size),
        JointRandomHorizontalFlip(p=0.5),
        JointRandomVerticalFlip(p=0.5),
        JointRandomRotation(),
        JointColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        JointToTensor(),
        JointNormalize(),
    ])


def get_val_transforms(image_size=512):
    return JointCompose([
        JointResize(image_size),
        JointToTensor(),
        JointNormalize(),
    ])